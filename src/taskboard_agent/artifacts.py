from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


class ArtifactError(RuntimeError):
    """Raised when an artifact is missing or fails integrity validation."""


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    kind: str
    byte_count: int
    sha256: str
    source_turn_id: str | None = None
    source_step_id: str | None = None
    label: str | None = None
    fields: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["fields"] = list(self.fields)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactRef:
        return cls(
            artifact_id=str(data["artifact_id"]),
            kind=str(data.get("kind") or "artifact"),
            byte_count=int(data.get("byte_count") or 0),
            sha256=str(data.get("sha256") or data["artifact_id"]),
            source_turn_id=_optional_str(data.get("source_turn_id")),
            source_step_id=_optional_str(data.get("source_step_id")),
            label=_optional_str(data.get("label")),
            fields=tuple(
                str(item) for item in data.get("fields", []) if isinstance(item, str)
            ),
        )


class ArtifactStore(Protocol):
    def put(
        self,
        value: Any,
        *,
        kind: str,
        source_turn_id: str | None = None,
        source_step_id: str | None = None,
        label: str | None = None,
    ) -> ArtifactRef:
        ...

    def get(self, artifact_id: str) -> Any:
        ...


class FileArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def put(
        self,
        value: Any,
        *,
        kind: str,
        source_turn_id: str | None = None,
        source_step_id: str | None = None,
        label: str | None = None,
    ) -> ArtifactRef:
        body = _canonical_json_bytes(value)
        digest = hashlib.sha256(body).hexdigest()
        target = self._path(digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            envelope = _canonical_json_bytes(
                {
                    "artifact_id": digest,
                    "sha256": digest,
                    "content": value,
                }
            )
            handle, temporary_name = tempfile.mkstemp(
                prefix=f".{digest}.", suffix=".tmp", dir=target.parent
            )
            try:
                with os.fdopen(handle, "wb") as temporary:
                    temporary.write(envelope)
                    temporary.flush()
                    os.fsync(temporary.fileno())
                os.replace(temporary_name, target)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
        return _artifact_ref(
            value,
            digest=digest,
            byte_count=len(body),
            kind=kind,
            source_turn_id=source_turn_id,
            source_step_id=source_step_id,
            label=label,
        )

    def get(self, artifact_id: str) -> Any:
        if not _valid_artifact_id(artifact_id):
            raise ArtifactError(f"invalid artifact id: {artifact_id}")
        target = self._path(artifact_id)
        try:
            envelope = json.loads(target.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ArtifactError(f"artifact not found: {artifact_id}") from exc
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"artifact could not be read: {artifact_id}: {exc}") from exc
        if not isinstance(envelope, dict) or "content" not in envelope:
            raise ArtifactError(f"artifact envelope is invalid: {artifact_id}")
        content = envelope["content"]
        actual = hashlib.sha256(_canonical_json_bytes(content)).hexdigest()
        if actual != artifact_id or envelope.get("sha256") != artifact_id:
            raise ArtifactError(f"artifact integrity check failed: {artifact_id}")
        return content

    def _path(self, artifact_id: str) -> Path:
        return self.root / artifact_id[:2] / f"{artifact_id}.json"


class InMemoryArtifactStore:
    def __init__(self) -> None:
        self._items: dict[str, Any] = {}

    def put(
        self,
        value: Any,
        *,
        kind: str,
        source_turn_id: str | None = None,
        source_step_id: str | None = None,
        label: str | None = None,
    ) -> ArtifactRef:
        body = _canonical_json_bytes(value)
        digest = hashlib.sha256(body).hexdigest()
        self._items.setdefault(digest, json.loads(body.decode("utf-8")))
        return _artifact_ref(
            value,
            digest=digest,
            byte_count=len(body),
            kind=kind,
            source_turn_id=source_turn_id,
            source_step_id=source_step_id,
            label=label,
        )

    def get(self, artifact_id: str) -> Any:
        if artifact_id not in self._items:
            raise ArtifactError(f"artifact not found: {artifact_id}")
        return json.loads(json.dumps(self._items[artifact_id], ensure_ascii=False))


def _artifact_ref(
    value: Any,
    *,
    digest: str,
    byte_count: int,
    kind: str,
    source_turn_id: str | None,
    source_step_id: str | None,
    label: str | None,
) -> ArtifactRef:
    fields = tuple(sorted(value)) if isinstance(value, dict) else ()
    return ArtifactRef(
        artifact_id=digest,
        kind=kind,
        byte_count=byte_count,
        sha256=digest,
        source_turn_id=source_turn_id,
        source_step_id=source_step_id,
        label=label,
        fields=fields,
    )


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _valid_artifact_id(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
