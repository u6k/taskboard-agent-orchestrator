from __future__ import annotations

import json
from pathlib import Path

import pytest

from taskboard_agent.artifacts import (
    ArtifactError,
    FileArtifactStore,
    InMemoryArtifactStore,
)


def test_file_artifact_store_deduplicates_and_reads_content(tmp_path: Path) -> None:
    store = FileArtifactStore(tmp_path / "artifacts")
    value = {"section": "第一章", "text": "本文"}

    first = store.put(value, kind="draft", label="section-1")
    second = store.put(value, kind="assistant_turn", label="answer")

    assert first.artifact_id == second.artifact_id
    assert first.byte_count == len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    assert store.get(first.artifact_id) == value
    assert len(list((tmp_path / "artifacts").rglob("*.json"))) == 1


def test_file_artifact_store_detects_tampering(tmp_path: Path) -> None:
    store = FileArtifactStore(tmp_path / "artifacts")
    ref = store.put({"text": "original"}, kind="draft")
    target = tmp_path / "artifacts" / ref.artifact_id[:2] / f"{ref.artifact_id}.json"
    target.write_text(
        json.dumps(
            {
                "artifact_id": ref.artifact_id,
                "sha256": ref.artifact_id,
                "content": {"text": "tampered"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="integrity"):
        store.get(ref.artifact_id)


def test_file_artifact_store_reports_missing_content(tmp_path: Path) -> None:
    store = FileArtifactStore(tmp_path / "artifacts")

    with pytest.raises(ArtifactError, match="not found"):
        store.get("a" * 64)


def test_in_memory_artifact_store_does_not_create_files(tmp_path: Path) -> None:
    store = InMemoryArtifactStore()
    ref = store.put({"text": "dry-run"}, kind="assistant_turn")

    assert store.get(ref.artifact_id) == {"text": "dry-run"}
    assert list(tmp_path.iterdir()) == []
