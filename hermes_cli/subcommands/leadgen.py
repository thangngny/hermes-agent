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
        choices=["read_only", "live"],
        default="read_only",
        help="Read-only default; live remains a manual-gate placeholder.",
    )
    leadgen_subparsers = parser.add_subparsers(dest="leadgen_command")
    show = leadgen_subparsers.add_parser("show", help="Show the current exported leads")
    show.set_defaults(func=cmd_leadgen)
    summary = leadgen_subparsers.add_parser("summary", help="Alias for leadgen show")
    summary.set_defaults(func=cmd_leadgen)
    read = leadgen_subparsers.add_parser("read", help="Alias for leadgen show")
    read.set_defaults(func=cmd_leadgen)
    parser.set_defaults(func=cmd_leadgen)
