from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
import tomllib
from typing import Any

from dotenv import load_dotenv


class ConfigError(ValueError):
    """Raised when required runtime configuration is missing or invalid."""


@dataclass(frozen=True)
class AgentProfileConfig:
    id: str
    redmine_user_id: int
    redmine_api_key: str = field(repr=False)
    llm_model: str
    context_window_tokens: int
    llm_api_key: str = field(repr=False)
    llm_api_base: str | None = None
    llm_timeout_seconds: int | None = None
    system_prompt: str | None = field(default=None, repr=False)
    system_prompt_file: Path | None = None


@dataclass(frozen=True)
class AppConfig:
    redmine_url: str
    redmine_in_progress_status_id: int
    redmine_review_status_id: int
    linkace_url: str
    linkace_api_key: str
    linkace_summarized_list_id: int
    agents: tuple[AgentProfileConfig, ...]
    langgraph_checkpoint_db_path: Path = Path(
        ".taskboard-agent/checkpoints.sqlite3"
    )


def load_config(
    env_file: str | Path = ".env",
    agents_file: str | Path = "agents.toml",
) -> AppConfig:
    load_dotenv(dotenv_path=env_file, override=False)

    redmine_url = _required("REDMINE_URL").rstrip("/")
    redmine_in_progress_status_id = _optional_int("REDMINE_IN_PROGRESS_STATUS_ID", 2)
    redmine_review_status_id = _optional_int("REDMINE_REVIEW_STATUS_ID", 10)
    linkace_url = _required("LINKACE_URL").rstrip("/")
    linkace_api_key = _required("LINKACE_API_KEY")
    linkace_summarized_list_id = _optional_int("LINKACE_SUMMARIZED_LIST_ID", 10)
    langgraph_checkpoint_db_path = Path(
        os.getenv(
            "LANGGRAPH_CHECKPOINT_DB_PATH",
            ".taskboard-agent/checkpoints.sqlite3",
        ).strip()
    )
    agents = _load_agent_profiles(Path(agents_file))

    return AppConfig(
        redmine_url=redmine_url,
        redmine_in_progress_status_id=redmine_in_progress_status_id,
        redmine_review_status_id=redmine_review_status_id,
        linkace_url=linkace_url,
        linkace_api_key=linkace_api_key,
        linkace_summarized_list_id=linkace_summarized_list_id,
        agents=agents,
        langgraph_checkpoint_db_path=langgraph_checkpoint_db_path,
    )


def _load_agent_profiles(path: Path) -> tuple[AgentProfileConfig, ...]:
    try:
        with path.open("rb") as file:
            data = tomllib.load(file)
    except FileNotFoundError as exc:
        raise ConfigError(f"agent config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"agent config file is not valid TOML: {path}: {exc}") from exc

    if data.get("version") != 2:
        raise ConfigError("agent config version must be 2")
    raw_agents = data.get("agents")
    if not isinstance(raw_agents, list) or not raw_agents:
        raise ConfigError("agent config must include at least one [[agents]] entry")

    enabled_profiles: list[AgentProfileConfig] = []
    seen_ids: set[str] = set()
    seen_redmine_user_ids: set[int] = set()
    for index, raw_profile in enumerate(raw_agents, 1):
        if not isinstance(raw_profile, dict):
            raise ConfigError(f"agents[{index}] must be a TOML table")
        profile = _parse_agent_profile(raw_profile, index=index, config_path=path)
        if profile.id in seen_ids:
            raise ConfigError(f"duplicate agent id: {profile.id}")
        if profile.redmine_user_id in seen_redmine_user_ids:
            raise ConfigError(
                f"duplicate agent redmine_user_id: {profile.redmine_user_id}"
            )
        seen_ids.add(profile.id)
        seen_redmine_user_ids.add(profile.redmine_user_id)
        if _optional_bool(raw_profile, "enabled", True, index=index):
            enabled_profiles.append(profile)

    if not enabled_profiles:
        raise ConfigError("agent config must enable at least one agent")
    return tuple(enabled_profiles)


def _parse_agent_profile(
    raw: dict[str, Any],
    *,
    index: int,
    config_path: Path,
) -> AgentProfileConfig:
    agent_id = _required_profile_string(raw, "id", index=index)
    redmine_user_id = _required_profile_int(raw, "redmine_user_id", index=index)
    if redmine_user_id <= 0:
        raise ConfigError(f"agents[{index}].redmine_user_id must be positive")
    redmine_api_key = _required_profile_string(
        raw, "redmine_api_key", index=index
    )
    llm_model = _required_profile_string(raw, "llm_model", index=index)
    context_window_tokens = _required_profile_int(
        raw, "context_window_tokens", index=index
    )
    if context_window_tokens <= 0:
        raise ConfigError(
            f"agents[{index}].context_window_tokens must be a positive integer"
        )
    llm_api_key = _required_profile_string(
        raw,
        "llm_api_key",
        index=index,
        allow_empty=True,
    )
    llm_api_base = _optional_profile_string(raw, "llm_api_base", index=index)
    llm_timeout_seconds = _optional_positive_profile_int(
        raw,
        "llm_timeout_seconds",
        index=index,
    )

    prompt_value = raw.get("system_prompt_file")
    system_prompt_file: Path | None = None
    system_prompt: str | None = None
    if prompt_value is not None:
        if not isinstance(prompt_value, str) or not prompt_value.strip():
            raise ConfigError(
                f"agents[{index}].system_prompt_file must be a non-empty string"
            )
        system_prompt_file = Path(prompt_value.strip())
        if not system_prompt_file.is_absolute():
            system_prompt_file = config_path.parent / system_prompt_file
        try:
            system_prompt = system_prompt_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise ConfigError(
                f"agents[{index}].system_prompt_file not found: {system_prompt_file}"
            ) from exc
        except (OSError, UnicodeError) as exc:
            raise ConfigError(
                f"agents[{index}].system_prompt_file could not be read: "
                f"{system_prompt_file}: {exc}"
            ) from exc
        if not system_prompt:
            raise ConfigError(
                f"agents[{index}].system_prompt_file must not be empty: "
                f"{system_prompt_file}"
            )

    return AgentProfileConfig(
        id=agent_id,
        redmine_user_id=redmine_user_id,
        redmine_api_key=redmine_api_key,
        llm_model=llm_model,
        context_window_tokens=context_window_tokens,
        llm_api_base=llm_api_base,
        llm_api_key=llm_api_key,
        llm_timeout_seconds=llm_timeout_seconds,
        system_prompt=system_prompt,
        system_prompt_file=system_prompt_file,
    )


def _required_profile_string(
    raw: dict[str, Any],
    name: str,
    *,
    index: int,
    allow_empty: bool = False,
) -> str:
    value = raw.get(name)
    if not isinstance(value, str):
        raise ConfigError(f"agents[{index}].{name} must be a string")
    stripped = value.strip()
    if not allow_empty and not stripped:
        raise ConfigError(f"agents[{index}].{name} must not be empty")
    return stripped


def _optional_profile_string(
    raw: dict[str, Any],
    name: str,
    *,
    index: int,
) -> str | None:
    value = raw.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"agents[{index}].{name} must be a string")
    return value.strip() or None


def _required_profile_int(
    raw: dict[str, Any],
    name: str,
    *,
    index: int,
) -> int:
    value = raw.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"agents[{index}].{name} must be an integer")
    return value


def _optional_positive_profile_int(
    raw: dict[str, Any],
    name: str,
    *,
    index: int,
) -> int | None:
    value = raw.get(name)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"agents[{index}].{name} must be a positive integer")
    return value


def _optional_bool(
    raw: dict[str, Any],
    name: str,
    default: bool,
    *,
    index: int,
) -> bool:
    value = raw.get(name, default)
    if not isinstance(value, bool):
        raise ConfigError(f"agents[{index}].{name} must be a boolean")
    return value


def _required(name: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise ConfigError(f"{name} is required")
    return value.strip()


def _optional_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc
