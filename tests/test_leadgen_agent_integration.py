from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pytest

from hermes_cli.leadgen_agent_integration import (
    LeadGenAgentAdapter,
    LeadGenAgentSettings,
    looks_like_live_request,
    refuse_live_execution_message,
    respond_to_leadgen_request,
)
from hermes_cli.leadgen_cmd import cmd_leadgen
from hermes_cli.subcommands.leadgen import build_leadgen_parser


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "leadgen_agent"


def _settings(repo: Path) -> LeadGenAgentSettings:
    return LeadGenAgentSettings(
        repo=str(repo),
        hermes_json="data/output/hermes_leads.json",
        hermes_csv="data/output/hermes_leads.csv",
        demo_summary="data/output/demo_leads.md",
        report="data/output/report.json",
        mode="read_only",
        allow_dry_run=False,
        allow_live=False,
    )


def _copy_fixture_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "lead-gen-agent"
    (repo / "data" / "output").mkdir(parents=True)
    for name in ("hermes_leads.json", "demo_leads.md", "report.json"):
        (repo / "data" / "output" / name).write_text((FIXTURE_DIR / name).read_text(encoding="utf-8"), encoding="utf-8")
    (repo / "data" / "output" / "hermes_leads.csv").write_text(
        "lead_id,company_name\napify-123,The Gym Group\n",
        encoding="utf-8",
    )
    return repo


def test_adapter_loads_fixture_successfully(tmp_path):
    repo = _copy_fixture_repo(tmp_path)
    result = LeadGenAgentAdapter(_settings(repo)).load_exports()

    assert result.ok is True
    assert result.source_repo == str(repo)
    assert result.lead_count == 1
    assert result.report_run_id == "20260613-180000-dukick"
    assert result.report_completed_at == "2026-06-13T18:01:30+07:00"
    assert result.report_status == "completed"
    assert result.profile_name == "dukick_tvc"
    assert result.target_country == "VN"
    assert result.lead_query == "gym"
    assert result.demo_summary_text.strip().startswith("# Dukick Lead Discovery Summary")
    assert result.errors == []
    assert result.warnings in (
        [],
        ["Cảnh báo: report_export có thể đã cũ; hãy chạy lead-gen-agent lại trước khi demo."],
    )

    lead = result.leads[0]
    assert lead.company_name == "The Gym Group"
    assert lead.website == "https://www.thegymgroup.com/"
    assert lead.domain == "thegymgroup.com"
    assert lead.page_url == "https://www.facebook.com/TheGymGroup"
    assert lead.source == "apify"
    assert lead.source_id == "123"
    assert lead.cta_text == "Sign up"
    assert lead.page_category == "Gym"
    assert lead.platforms == ("Facebook", "Instagram", "Threads")
    assert lead.market_fit_decision == "global_or_unknown"
    assert lead.relevance_decision == "review"
    assert lead.qualification == "review"
    assert lead.suggested_video_angle.startswith("Member transformation")
    assert lead.why_dukick_should_care.startswith("Fitness brands")


def test_adapter_handles_missing_report_with_warning(tmp_path):
    repo = tmp_path / "lead-gen-agent"
    (repo / "data" / "output").mkdir(parents=True)
    (repo / "data" / "output" / "hermes_leads.json").write_text(
        (FIXTURE_DIR / "hermes_leads.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (repo / "data" / "output" / "demo_leads.md").write_text(
        (FIXTURE_DIR / "demo_leads.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    result = LeadGenAgentAdapter(_settings(repo)).load_exports()

    assert result.ok is True
    assert result.lead_count == 1
    assert any("report.json" in warning for warning in result.warnings)


@pytest.mark.parametrize(
    "filename, contents, expected",
    [
        ("hermes_leads.json", "[]", "Hiện chưa có lead keep/review trong export."),
        ("hermes_leads.json", "{bad json", "File hermes_leads.json không hợp lệ"),
    ],
)
def test_empty_and_invalid_exports_are_safe(tmp_path, filename, contents, expected):
    repo = tmp_path / "lead-gen-agent"
    (repo / "data" / "output").mkdir(parents=True)
    (repo / "data" / "output" / filename).write_text(contents, encoding="utf-8")
    if filename == "hermes_leads.json":
        (repo / "data" / "output" / "demo_leads.md").write_text("# Demo", encoding="utf-8")

    result = LeadGenAgentAdapter(_settings(repo)).load_exports()
    summary = LeadGenAgentAdapter(_settings(repo)).summarize_for_user()

    if contents == "[]":
        assert result.ok is True
        assert result.lead_count == 0
        assert summary == expected
    else:
        assert result.ok is False
        assert any(expected in error for error in result.errors)
        assert summary.startswith("File hermes_leads.json không hợp lệ")


def test_missing_export_file_is_handled_gracefully(tmp_path):
    repo = tmp_path / "lead-gen-agent"
    repo.mkdir()
    result = LeadGenAgentAdapter(_settings(repo)).load_exports()

    assert result.ok is False
    assert result.lead_count == 0
    assert any("Chưa có file lead export" in error for error in result.errors)
    assert LeadGenAgentAdapter(_settings(repo)).summarize_for_user() == "Chưa có file lead export. Cần chạy lead-gen-agent trước."


def test_summary_includes_vietnamese_fields_and_excludes_raw_payload(tmp_path):
    repo = _copy_fixture_repo(tmp_path)
    text = LeadGenAgentAdapter(_settings(repo)).summarize_for_user()

    assert "The Gym Group" in text
    assert "https://www.thegymgroup.com/" in text
    assert "Gym" in text
    assert "Market fit: global_or_unknown" in text
    assert "Trạng thái: review" in text
    assert "Member transformation" in text
    assert "raw_payload" not in text


def test_summary_redacts_secret_like_strings(tmp_path):
    repo = _copy_fixture_repo(tmp_path)
    payload = json.loads((repo / "data" / "output" / "hermes_leads.json").read_text(encoding="utf-8"))
    payload["leads"][0]["api_token"] = "should-not-appear"
    (repo / "data" / "output" / "hermes_leads.json").write_text(json.dumps(payload), encoding="utf-8")

    text = LeadGenAgentAdapter(_settings(repo)).summarize_for_user()
    assert "should-not-appear" not in text
    assert "api_token" not in text.lower()


def test_read_only_mode_never_invokes_subprocess(monkeypatch, tmp_path):
    repo = _copy_fixture_repo(tmp_path)
    called = False

    def boom(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("subprocess should not be called")

    monkeypatch.setattr(subprocess, "run", boom)
    text = respond_to_leadgen_request("show me the latest lead summary", config=_settings(repo))

    assert not called
    assert "The Gym Group" in text


def test_live_request_returns_manual_confirmation_warning():
    assert looks_like_live_request("run apify live now") is True
    assert looks_like_live_request("show the summary") is False
    warning = refuse_live_execution_message()
    assert "cần xác nhận thủ công" in warning
    assert "Apify" in warning


def test_live_request_handler_refuses_execution(monkeypatch, tmp_path):
    repo = _copy_fixture_repo(tmp_path)
    called = False

    def boom(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("subprocess should not be called")

    monkeypatch.setattr(subprocess, "run", boom)
    text = respond_to_leadgen_request("chạy live tìm lead ngay", config=_settings(repo))

    assert not called
    assert "Live Apify" in text
    assert "xác nhận thủ công" in text


def test_repo_path_expands_tilde():
    settings = LeadGenAgentSettings(repo="~/CongThang/lead-gen-agent")
    assert str(settings.repo_path).endswith("CongThang/lead-gen-agent")


def test_cli_command_uses_adapter_and_parser(tmp_path, capsys):
    repo = _copy_fixture_repo(tmp_path)
    args = argparse.Namespace(leadgen_command="show", repo=str(repo), language="vi", limit=1, mode="read_only")
    cmd_leadgen(args)
    out = capsys.readouterr().out
    assert "The Gym Group" in out

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_leadgen_parser(subparsers, cmd_leadgen=cmd_leadgen)
    assert "leadgen" in subparsers.choices
