from __future__ import annotations

from dataclasses import replace

from hermes_cli.leadgen_agent_integration import (
    LeadGenAgentAdapter,
    LeadGenAgentSettings,
    refuse_live_execution_message,
)


def cmd_leadgen(args) -> None:  # noqa: ANN001
    mode = getattr(args, "mode", "read_only") or "read_only"
    language = getattr(args, "language", "vi") or "vi"
    limit = int(getattr(args, "limit", 5) or 5)
    config = LeadGenAgentSettings.from_runtime()
    repo_override = getattr(args, "repo", None)
    if repo_override:
        config = replace(config, repo=repo_override)

    if mode == "live" or not config.allow_live:
        # Read-only default: never execute live. This path is a safe refusal,
        # not a hidden subprocess wrapper.
        if mode == "live":
            print(refuse_live_execution_message(language=language))
            return

    adapter = LeadGenAgentAdapter(config)
    command = getattr(args, "leadgen_command", None)
    if command in (None, "show", "summary", "read"):
        print(adapter.summarize_for_user(language=language, limit=limit))
        return
    if command == "status":
        result = adapter.load_exports()
        print(result.to_dict())
        return
    print(adapter.summarize_for_user(language=language, limit=limit))
