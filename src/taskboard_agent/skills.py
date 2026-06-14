from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class SkillRegistryError(RuntimeError):
    """Raised when skill files cannot be loaded."""


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    required_tools: tuple[str, ...]
    risk_level: str
    path: Path
    body: str

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "required_tools": list(self.required_tools),
            "risk_level": self.risk_level,
        }


class SkillRegistry:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._skills: dict[str, Skill] | None = None

    def list(self) -> list[Skill]:
        if self._skills is None:
            self._skills = _load_skills(self._root)
        return list(self._skills.values())

    def summaries(self) -> list[dict[str, object]]:
        return [skill.summary() for skill in self.list()]

    def get(self, name: str) -> Skill:
        if self._skills is None:
            self._skills = _load_skills(self._root)
        try:
            return self._skills[name]
        except KeyError as exc:
            raise SkillRegistryError(f"unknown skill: {name}") from exc


def _load_skills(root: Path) -> dict[str, Skill]:
    if not root.exists():
        return {}
    skills: dict[str, Skill] = {}
    for skill_file in sorted(root.glob("*/SKILL.md")):
        skill = _load_skill(skill_file)
        if skill.name in skills:
            raise SkillRegistryError(f"duplicate skill name: {skill.name}")
        skills[skill.name] = skill
    return skills


def _load_skill(path: Path) -> Skill:
    text = path.read_text(encoding="utf-8")
    metadata, body = _split_front_matter(text, path)
    name = metadata.get("name")
    description = metadata.get("description")
    if not isinstance(name, str) or not name:
        raise SkillRegistryError(f"skill missing name: {path}")
    if not isinstance(description, str) or not description:
        raise SkillRegistryError(f"skill missing description: {path}")
    required_tools = metadata.get("required_tools", [])
    if not isinstance(required_tools, list) or not all(
        isinstance(item, str) for item in required_tools
    ):
        raise SkillRegistryError(f"skill required_tools must be a string list: {path}")
    risk_level = metadata.get("risk_level", "read")
    if not isinstance(risk_level, str):
        raise SkillRegistryError(f"skill risk_level must be a string: {path}")
    return Skill(
        name=name,
        description=description,
        required_tools=tuple(required_tools),
        risk_level=risk_level,
        path=path,
        body=body.strip(),
    )


def _split_front_matter(text: str, path: Path) -> tuple[dict[str, object], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillRegistryError(f"skill missing front matter: {path}")
    try:
        end_index = next(
            index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"
        )
    except StopIteration as exc:
        raise SkillRegistryError(f"skill front matter is not closed: {path}") from exc

    metadata = _parse_simple_yaml(lines[1:end_index], path)
    body = "\n".join(lines[end_index + 1 :])
    return metadata, body


def _parse_simple_yaml(lines: list[str], path: Path) -> dict[str, object]:
    metadata: dict[str, object] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if line.startswith(" ") or ":" not in line:
            raise SkillRegistryError(f"unsupported front matter line in {path}: {line}")
        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if value == "":
            items: list[str] = []
            index += 1
            while index < len(lines) and lines[index].startswith("  - "):
                items.append(_unquote(lines[index][4:].strip()))
                index += 1
            metadata[key] = items
            continue
        metadata[key] = _unquote(value)
        index += 1
    return metadata


def _unquote(value: str) -> str:
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value
