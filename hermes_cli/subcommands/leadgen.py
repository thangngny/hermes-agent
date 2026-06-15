from __future__ import annotations

import argparse
from typing import Callable


def build_leadgen_parser(subparsers, *, cmd_leadgen: Callable) -> None:
    parser = subparsers.add_parser(
        "leadgen",
        help="Read Dukick lead discovery exports from lead-gen-agent",
        description=(
            "Read-only integration with the local lead-gen-agent export files. "
            "This command never runs Apify, OpenAI, or any subprocess in read-only mode."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="Override the lead-gen-agent repo path (default: ~/CongThang/lead-gen-agent)",
    )
    parser.add_argument(
        "--language",
        default="vi",
        help="Summary language (default: vi)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of leads to include in the summary (default: 5)",
    )
    parser.add_argument(
        "--mode",
        choices=["read_only", "confirm_live", "live"],
        default="confirm_live",
        help="Read-only summary by default; confirm_live allows safe preflight and gated live execution.",
    )
    leadgen_subparsers = parser.add_subparsers(dest="leadgen_command")
    show = leadgen_subparsers.add_parser("show", help="Show the current exported leads")
    show.set_defaults(func=cmd_leadgen)
    summary = leadgen_subparsers.add_parser("summary", help="Alias for leadgen show")
    summary.set_defaults(func=cmd_leadgen)
    read = leadgen_subparsers.add_parser("read", help="Alias for leadgen show")
    read.set_defaults(func=cmd_leadgen)
    plan = leadgen_subparsers.add_parser("plan", help="Prepare a safe preflight for a lead search")
    plan.add_argument("--query", required=True, help="Lead query or industry, e.g. spa")
    plan.add_argument("--country", default="VN", help="Target country code (default: VN)")
    plan.add_argument("--max-results", type=int, default=1, help="Max leads to search (default: 1)")
    plan.add_argument("--profile", default="dukick_tvc", help="Lead profile (default: dukick_tvc)")
    plan.set_defaults(func=cmd_leadgen)
    run_confirmed = leadgen_subparsers.add_parser("run-confirmed", help="Run a confirmed live search with an explicit cost flag")
    run_confirmed.add_argument("--query", required=True, help="Lead query or industry, e.g. spa")
    run_confirmed.add_argument("--country", default="VN", help="Target country code (default: VN)")
    run_confirmed.add_argument("--max-results", type=int, default=1, help="Max leads to search (default: 1)")
    run_confirmed.add_argument("--profile", default="dukick_tvc", help="Lead profile (default: dukick_tvc)")
    run_confirmed.add_argument(
        "--i-understand-this-may-cost-money",
        action="store_true",
        help="Required explicit acknowledgement before Hermes runs a live Apify search.",
    )
    run_confirmed.set_defaults(func=cmd_leadgen)
    parser.set_defaults(func=cmd_leadgen)
