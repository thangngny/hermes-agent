from __future__ import annotations

from dataclasses import replace

from hermes_cli.leadgen_agent_integration import (
    LeadGenAgentAdapter,
    LeadGenAgentSettings,
    refuse_live_execution_message,
)
from hermes_cli.leadgen_agent_orchestrator import LeadGenAgentRunner, LeadGenSearchRequest


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
    if command == "plan":
        runner = LeadGenAgentRunner(config)
        requested = int(getattr(args, "max_results", 1) or 1)
        max_results = min(requested, config.max_live_results)
        request = LeadGenSearchRequest(
            query=str(getattr(args, "query", "") or "").strip(),
            country=str(getattr(args, "country", config.default_country) or config.default_country).strip(),
            profile=str(getattr(args, "profile", config.default_profile) or config.default_profile).strip(),
            max_results=max_results,
            spend_limit_usd=float(config.spend_limit_usd),
            requested_max_results=requested,
            capped=requested > max_results,
        )
        result = runner.preflight_search(request)
        print(
            "\n".join(
                [
                    f"query={result.request.query}",
                    f"country={result.request.country}",
                    f"max_results={result.request.max_results}",
                    f"estimate_usd={result.estimate_usd if result.estimate_usd is not None else 'unknown'}",
                    f"ok={result.ok}",
                    result.command_preview,
                ]
            )
        )
        return
    if command == "run-confirmed":
        if not getattr(args, "i_understand_this_may_cost_money", False):
            print("Refusing live run. Re-run with --i-understand-this-may-cost-money after reviewing the preflight.")
            return
        runner = LeadGenAgentRunner(config)
        requested = int(getattr(args, "max_results", 1) or 1)
        max_results = min(requested, config.max_live_results)
        request = LeadGenSearchRequest(
            query=str(getattr(args, "query", "") or "").strip(),
            country=str(getattr(args, "country", config.default_country) or config.default_country).strip(),
            profile=str(getattr(args, "profile", config.default_profile) or config.default_profile).strip(),
            max_results=max_results,
            spend_limit_usd=float(config.spend_limit_usd),
            requested_max_results=requested,
            capped=requested > max_results,
        )
        print(runner.run_live_search(request, confirmation="xác nhận chạy live"))
        return
    if command == "status":
        result = adapter.load_exports()
        print(result.to_dict())
        return
    print(adapter.summarize_for_user(language=language, limit=limit))
