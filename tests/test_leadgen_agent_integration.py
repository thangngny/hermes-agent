from __future__ import annotations

import argparse
import json
import subprocess
from types import SimpleNamespace
from pathlib import Path

import pytest

from agent.conversation_loop import _maybe_route_sales_lead_request
from hermes_cli.leadgen_agent_integration import (
    LeadGenAgentAdapter,
    LeadGenAgentSettings,
    looks_like_live_request,
    looks_like_lead_discovery_request,
    looks_like_explicit_public_web_search_request,
    refuse_live_execution_message,
    respond_to_leadgen_request,
    route_sales_lead_request,
)
from hermes_cli.leadgen_agent_orchestrator import (
    LeadGenAgentRunner,
    LeadGenPendingRequest,
    LeadGenSearchRequest,
    load_pending_search,
    looks_like_confirmation_message,
    looks_like_read_latest_request,
    orchestrate_sales_lead_request,
    parse_lead_search_request,
    save_pending_search,
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
        mode="confirm_live",
        allow_dry_run=True,
        allow_live=False,
        max_live_results=3,
        default_country="VN",
        default_profile="dukick_tvc",
        spend_limit_usd=1.0,
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
        assert "Chưa có lead đủ điều kiện export." in summary
        assert "Exported leads: 0" in summary
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
    assert "Trạng thái: Review" in text
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
    assert looks_like_lead_discovery_request("khảo sát thêm lead trên mạng giúp tôi") is True
    assert looks_like_lead_discovery_request("tìm khách hàng tiềm năng cho Dukick") is True
    assert looks_like_explicit_public_web_search_request("hãy dùng public web search") is True
    assert looks_like_confirmation_message("xác nhận chạy") is True
    assert looks_like_read_latest_request("đọc lead mới nhất") is True
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


@pytest.mark.parametrize(
    ("user_text", "expected_query", "expected_country", "expected_max"),
    [
        ("tìm lead spa ở Việt Nam cho Dukick", "spa", "VN", 1),
        ("tìm 3 lead gym", "gym", "VN", 3),
        ("tìm lead nha khoa Mỹ", "nha khoa", "US", 1),
        ("Tìm lead dự án căn hộ ở Việt Nam cho Dukick, tối đa 3 lead", "dự án căn hộ", "VN", 3),
        ("Tìm lead chung cư mở bán ở Việt Nam cho Dukick, tối đa 3 lead", "chung cư mở bán", "VN", 3),
        ("Tìm lead bất động sản ở Việt Nam cho Dukick, tối đa 1 lead", "bất động sản", "VN", 1),
    ],
)
def test_parse_lead_search_request(user_text, expected_query, expected_country, expected_max, tmp_path):
    request = parse_lead_search_request(user_text, config=_settings(_copy_fixture_repo(tmp_path)))

    assert request is not None
    assert request.query == expected_query
    assert request.country == expected_country
    assert request.max_results == expected_max


def test_parse_lead_search_caps_results(tmp_path):
    request = parse_lead_search_request("tìm 100 lead spa", config=_settings(_copy_fixture_repo(tmp_path)))

    assert request is not None
    assert request.requested_max_results == 100
    assert request.max_results == 3
    assert request.capped is True


def test_parse_lead_search_strips_control_phrases_from_query(tmp_path):
    request = parse_lead_search_request(
        "Tìm lead chung cư mở bán ở Việt Nam cho Dukick, tối đa 3 lead giúp tôi",
        config=_settings(_copy_fixture_repo(tmp_path)),
    )

    assert request is not None
    assert request.query == "chung cư mở bán"
    assert "tối đa" not in request.query
    assert "dukick" not in request.query
    assert "việt nam" not in request.query


def test_preflight_uses_subprocess_with_dry_run_true(monkeypatch, tmp_path):
    repo = _copy_fixture_repo(tmp_path)
    runner = LeadGenAgentRunner(_settings(repo))
    request = LeadGenSearchRequest(
        query="spa",
        country="VN",
        profile="dukick_tvc",
        max_results=1,
        spend_limit_usd=1.0,
        requested_max_results=1,
    )
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        payload = {
            "request": {"request_id": "20260615-req-1"},
            "apify": {
                "ok": True,
                "estimated_cost_usd": 0.02,
                "reasons": [],
                "details": {"planned_queries": ["spa", "showroom nội thất"]},
            },
        }
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = runner.preflight_search(request)

    assert result.ok is True
    assert result.estimate_usd == 0.02
    assert result.request_id == "20260615-req-1"
    assert result.planned_queries == ("spa", "showroom nội thất")
    assert len(calls) == 2
    assert calls[0][0] == ["leadgen-agent", "smoke-plan", "--source", "apify"]
    assert calls[1][0] == ["leadgen-agent", "preflight", "--source", "apify"]
    for _cmd, kwargs in calls:
        env = kwargs["env"]
        assert env["DRY_RUN"] == "true"
        assert env["OPENAI_ENABLED"] == "false"
        assert env["SEND_ENABLED"] == "false"
        assert env["LEAD_QUERY"] == "spa"
        assert "APIFY_API_TOKEN" not in result.command_preview


def test_preflight_does_not_mutate_dotenv(monkeypatch, tmp_path):
    repo = _copy_fixture_repo(tmp_path)
    dotenv = repo / ".env"
    original = "DRY_RUN=true\n"
    dotenv.write_text(original, encoding="utf-8")
    runner = LeadGenAgentRunner(_settings(repo))
    request = LeadGenSearchRequest("spa", "VN", "dukick_tvc", 1, 1.0, 1)

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"apify": {"ok": True}}), stderr=""),
    )
    runner.preflight_search(request)

    assert dotenv.read_text(encoding="utf-8") == original


def test_live_run_refused_without_confirmation(tmp_path):
    runner = LeadGenAgentRunner(_settings(_copy_fixture_repo(tmp_path)))
    request = LeadGenSearchRequest("spa", "VN", "dukick_tvc", 1, 1.0, 1)

    with pytest.raises(RuntimeError, match="missing_confirmation"):
        runner.run_live_search(request, confirmation="chưa xác nhận")


def test_live_run_refused_when_allow_live_false(tmp_path):
    runner = LeadGenAgentRunner(_settings(_copy_fixture_repo(tmp_path)))
    request = LeadGenSearchRequest("spa", "VN", "dukick_tvc", 1, 1.0, 1)

    with pytest.raises(RuntimeError, match="live_disabled"):
        runner.run_live_search(request, confirmation="xác nhận chạy")


def test_live_run_allowed_only_with_confirmation_and_allow_live_true(monkeypatch, tmp_path):
    settings = _settings(_copy_fixture_repo(tmp_path))
    settings = LeadGenAgentSettings(**{**settings.__dict__, "allow_live": True})
    runner = LeadGenAgentRunner(settings)
    request = LeadGenSearchRequest("spa", "VN", "dukick_tvc", 1, 1.0, 1)
    seen = []

    def fake_run(cmd, **kwargs):
        seen.append((cmd, kwargs["env"]["DRY_RUN"]))
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"ok": True}), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    text = runner.run_live_search(request, confirmation="xác nhận chạy live 1 lead")

    assert "Đã chạy xong" in text
    assert seen == [
        (["leadgen-agent", "search-leads", "--profile", "dukick_tvc", "--source", "apify", "--country", "VN", "--query", "spa", "--max-results", "1", "--live"], "false"),
        (["leadgen-agent", "demo-summary"], "false"),
    ]


def test_live_run_uses_accurate_zero_export_message(monkeypatch, tmp_path):
    repo = _copy_fixture_repo(tmp_path)
    (repo / "data" / "output" / "hermes_leads.json").write_text("[]", encoding="utf-8")
    settings = LeadGenAgentSettings(**{**_settings(repo).__dict__, "allow_live": True})
    runner = LeadGenAgentRunner(settings)
    request = LeadGenSearchRequest("bất động sản", "VN", "dukick_tvc", 1, 1.0, 1)

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"ok": True}), stderr=""),
    )
    text = runner.run_live_search(request, confirmation="xác nhận chạy live 1 lead")

    assert "Đã chạy xong search mới, nhưng chưa có lead nào đủ điều kiện export cho Hermes." in text


def test_live_run_uses_accurate_positive_export_message(monkeypatch, tmp_path):
    repo = _copy_fixture_repo(tmp_path)
    settings = LeadGenAgentSettings(**{**_settings(repo).__dict__, "allow_live": True})
    runner = LeadGenAgentRunner(settings)
    request = LeadGenSearchRequest("spa", "VN", "dukick_tvc", 1, 1.0, 1)

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"ok": True}), stderr=""),
    )
    text = runner.run_live_search(request, confirmation="xác nhận chạy live 1 lead")

    assert "Đã chạy xong search mới. Tìm được 1 lead đủ điều kiện export cho Hermes." in text


def test_sales_read_latest_still_uses_read_only_summary(tmp_path):
    repo = _copy_fixture_repo(tmp_path)

    text = route_sales_lead_request(
        "đọc lead mới nhất từ lead-gen-agent",
        profile_name="sales",
        config=_settings(repo),
    )

    assert "Đã đọc kết quả từ lead-gen-agent" in text
    assert "The Gym Group" in text


def test_sales_search_routes_to_preflight_not_summary(monkeypatch, tmp_path):
    repo = _copy_fixture_repo(tmp_path)
    settings = LeadGenAgentSettings(**{**_settings(repo).__dict__, "allow_live": True})

    monkeypatch.setattr(
        "subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(
                {
                    "request": {"request_id": "20260615-req-3"},
                    "apify": {
                        "ok": True,
                        "estimated_cost_usd": 0.02,
                        "reasons": [],
                        "details": {"planned_queries": ["spa", "showroom nội thất"]},
                    },
                }
            ),
            stderr="",
        ),
    )
    text = route_sales_lead_request(
        "tìm lead spa ở Việt Nam cho Dukick",
        profile_name="sales",
        config=settings,
    )

    assert "Em đã chuẩn bị tìm lead bằng Apify" in text
    assert "* Ngành/query: spa" in text
    assert "Dự kiến sẽ thử các query sau" in text
    assert "showroom nội thất" in text
    assert "Mã plan/run dự kiến: 20260615-req-3" in text
    assert "Anh xác nhận `chạy live 1 lead`" in text


def test_preflight_falls_back_when_planned_queries_missing(monkeypatch, tmp_path):
    repo = _copy_fixture_repo(tmp_path)
    settings = LeadGenAgentSettings(**{**_settings(repo).__dict__, "allow_live": True})

    monkeypatch.setattr(
        "subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps({"apify": {"ok": True, "estimated_cost_usd": 0.02, "reasons": []}}),
            stderr="",
        ),
    )
    text = route_sales_lead_request(
        "tìm lead spa ở Việt Nam cho Dukick",
        profile_name="sales",
        config=settings,
    )

    assert text is not None
    assert "Em đã chuẩn bị tìm lead bằng Apify" in text
    assert "Dự kiến sẽ thử các query sau" not in text
    assert "Anh xác nhận `chạy live 1 lead`" in text


def test_sales_profile_missing_export_returns_safe_message_when_reading_latest(tmp_path):
    repo = tmp_path / "lead-gen-agent"
    repo.mkdir()

    text = route_sales_lead_request(
        "đọc lead mới nhất",
        profile_name="sales",
        config=_settings(repo),
    )

    assert text == "Chưa có file lead export. Cần chạy lead-gen-agent trước."


def test_explicit_public_web_search_only_bypasses_when_export_missing(tmp_path):
    repo = tmp_path / "lead-gen-agent"
    repo.mkdir()

    text = route_sales_lead_request(
        "hãy dùng public web search để tìm lead cho Dukick",
        profile_name="sales",
        config=_settings(repo),
    )

    assert text is None


def test_explicit_public_web_search_still_prefers_export_when_available(tmp_path):
    repo = _copy_fixture_repo(tmp_path)

    text = route_sales_lead_request(
        "hãy dùng public web search để đọc lead cho Dukick",
        profile_name="sales",
        config=_settings(repo),
    )

    assert text is not None
    assert "The Gym Group" in text


def test_non_sales_profile_does_not_intercept_lead_requests(tmp_path):
    repo = _copy_fixture_repo(tmp_path)

    text = route_sales_lead_request(
        "tìm lead trên mạng giúp tôi",
        profile_name="work",
        config=_settings(repo),
    )

    assert text is None


def test_pending_search_state_round_trip(monkeypatch):
    session_row = {"model_config": "{}", "model": "x"}
    agent = SimpleNamespace(
        session_id="sess-1",
        _session_db=SimpleNamespace(
            get_session=lambda _sid: session_row,
            update_session_meta=lambda _sid, meta, model=None: session_row.update({"model_config": meta, "model": model}),
        ),
    )
    pending = LeadGenPendingRequest(
        request=LeadGenSearchRequest("spa", "VN", "dukick_tvc", 1, 1.0, 1),
        estimate_usd=0.02,
        preflight_ok=True,
        command_preview="leadgen-agent search-leads ...",
        created_at=1.0,
    )

    save_pending_search(agent, pending)
    restored = load_pending_search(SimpleNamespace(session_id="sess-1", _session_db=agent._session_db))

    assert restored is not None
    assert restored.request.query == "spa"
    assert json.loads(session_row["model_config"])["_leadgen_pending"]["request"]["country"] == "VN"


def test_orchestrator_ok_chay_runs_pending_only_if_allowed(monkeypatch, tmp_path):
    repo = _copy_fixture_repo(tmp_path)
    settings = LeadGenAgentSettings(**{**_settings(repo).__dict__, "allow_live": True})
    pending = LeadGenPendingRequest(
        request=LeadGenSearchRequest("spa", "VN", "dukick_tvc", 1, 1.0, 1),
        estimate_usd=0.02,
        preflight_ok=True,
        command_preview="leadgen-agent search-leads ...",
        created_at=1.0,
    )
    seen = []

    def fake_run(cmd, **kwargs):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"ok": True}), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    decision = orchestrate_sales_lead_request(
        "ok chạy",
        profile_name="sales",
        config=settings,
        pending=pending,
    )

    assert decision is not None
    assert decision.clear_pending is True
    assert seen[0][1] == "search-leads"


def test_orchestrator_search_creates_pending_request(monkeypatch, tmp_path):
    repo = _copy_fixture_repo(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps({"apify": {"ok": True, "estimated_cost_usd": 0.02, "reasons": []}}),
            stderr="",
        ),
    )

    decision = orchestrate_sales_lead_request(
        "tìm lead spa ở Việt Nam cho Dukick",
        profile_name="sales",
        config=_settings(repo),
        pending=None,
    )

    assert decision is not None
    assert decision.pending is not None
    assert decision.pending.request.query == "spa"
    assert "Estimate: khoảng $0.02" in decision.reply


def test_orchestrator_follow_up_updates_query_and_replans(monkeypatch, tmp_path):
    repo = _copy_fixture_repo(tmp_path)
    pending = LeadGenPendingRequest(
        request=LeadGenSearchRequest("spa", "VN", "dukick_tvc", 1, 1.0, 1),
        estimate_usd=0.02,
        preflight_ok=True,
        command_preview="leadgen-agent search-leads ...",
        created_at=1.0,
    )
    seen = []

    def fake_run(cmd, **kwargs):
        seen.append(cmd)
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps({"apify": {"ok": True, "estimated_cost_usd": 0.03, "reasons": []}}),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    decision = orchestrate_sales_lead_request(
        "ngành bất động sản đi",
        profile_name="sales",
        config=_settings(repo),
        pending=pending,
    )

    assert decision is not None
    assert decision.pending is not None
    assert decision.pending.request.query == "bất động sản"
    assert decision.pending.request.country == "VN"
    assert decision.pending.request.max_results == 1
    assert "Em đã đổi ngành sang bất động sản" in decision.reply
    assert seen == [
        ["leadgen-agent", "smoke-plan", "--source", "apify"],
        ["leadgen-agent", "preflight", "--source", "apify"],
    ]


def test_full_new_request_replaces_pending_request(monkeypatch, tmp_path):
    repo = _copy_fixture_repo(tmp_path)
    pending = LeadGenPendingRequest(
        request=LeadGenSearchRequest("spa", "VN", "dukick_tvc", 1, 1.0, 1),
        estimate_usd=0.02,
        preflight_ok=True,
        command_preview="leadgen-agent search-leads ...",
        created_at=1.0,
    )
    seen_env = []

    def fake_run(cmd, **kwargs):
        seen_env.append(kwargs["env"]["LEAD_QUERY"])
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps({"apify": {"ok": True, "estimated_cost_usd": 0.02, "reasons": []}}),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    decision = orchestrate_sales_lead_request(
        "Tìm lead chung cư mở bán ở Việt Nam cho Dukick, tối đa 3 lead",
        profile_name="sales",
        config=_settings(repo),
        pending=pending,
    )

    assert decision is not None
    assert decision.pending is not None
    assert decision.pending.request.query == "chung cư mở bán"
    assert decision.pending.request.max_results == 3
    assert seen_env == ["chung cư mở bán", "chung cư mở bán"]


@pytest.mark.parametrize(
    ("follow_up", "expected_query", "expected_country", "expected_max"),
    [
        ("đổi sang gym", "gym", "VN", 1),
        ("tối đa 3 lead", "spa", "VN", 3),
        ("ở US", "spa", "US", 1),
    ],
)
def test_orchestrator_follow_up_updates_pending_request_fields(
    monkeypatch,
    tmp_path,
    follow_up,
    expected_query,
    expected_country,
    expected_max,
):
    repo = _copy_fixture_repo(tmp_path)
    pending = LeadGenPendingRequest(
        request=LeadGenSearchRequest("spa", "VN", "dukick_tvc", 1, 1.0, 1),
        estimate_usd=0.02,
        preflight_ok=True,
        command_preview="leadgen-agent search-leads ...",
        created_at=1.0,
    )

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps({"apify": {"ok": True, "estimated_cost_usd": 0.02, "reasons": []}}),
            stderr="",
        ),
    )
    decision = orchestrate_sales_lead_request(
        follow_up,
        profile_name="sales",
        config=_settings(repo),
        pending=pending,
    )

    assert decision is not None
    assert decision.pending is not None
    assert decision.pending.request.query == expected_query
    assert decision.pending.request.country == expected_country
    assert decision.pending.request.max_results == expected_max


def test_confirmation_after_follow_up_uses_latest_pending_request(monkeypatch, tmp_path):
    repo = _copy_fixture_repo(tmp_path)
    settings = LeadGenAgentSettings(**{**_settings(repo).__dict__, "allow_live": True})
    pending = LeadGenPendingRequest(
        request=LeadGenSearchRequest("spa", "VN", "dukick_tvc", 1, 1.0, 1),
        estimate_usd=0.02,
        preflight_ok=True,
        command_preview="leadgen-agent search-leads ...",
        created_at=1.0,
    )
    seen = []

    def fake_run(cmd, **kwargs):
        seen.append((cmd, kwargs["env"].get("LEAD_QUERY"), kwargs["env"].get("TARGET_COUNTRY"), kwargs["env"].get("APIFY_MAX_RESULTS")))
        if cmd[1] in {"smoke-plan", "preflight"}:
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"apify": {"ok": True, "estimated_cost_usd": 0.02, "reasons": []}}), stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"ok": True}), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    updated = orchestrate_sales_lead_request(
        "ngành bất động sản đi",
        profile_name="sales",
        config=settings,
        pending=pending,
    )
    assert updated is not None and updated.pending is not None

    decision = orchestrate_sales_lead_request(
        "xác nhận chạy live 1 lead",
        profile_name="sales",
        config=settings,
        pending=updated.pending,
    )

    assert decision is not None
    assert decision.clear_pending is True
    assert ["leadgen-agent", "search-leads", "--profile", "dukick_tvc", "--source", "apify", "--country", "VN", "--query", "bất động sản", "--max-results", "1", "--live"] == seen[2][0]
    assert seen[2][1:] == ("bất động sản", "VN", "1")


def test_confirmation_after_full_new_request_uses_latest_clean_query(monkeypatch, tmp_path):
    repo = _copy_fixture_repo(tmp_path)
    settings = LeadGenAgentSettings(**{**_settings(repo).__dict__, "allow_live": True})
    pending = LeadGenPendingRequest(
        request=LeadGenSearchRequest("spa", "VN", "dukick_tvc", 1, 1.0, 1),
        estimate_usd=0.02,
        preflight_ok=True,
        command_preview="leadgen-agent search-leads ...",
        created_at=1.0,
    )
    seen = []

    def fake_run(cmd, **kwargs):
        seen.append((cmd, kwargs["env"]["LEAD_QUERY"]))
        if cmd[1] in {"smoke-plan", "preflight"}:
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"apify": {"ok": True, "estimated_cost_usd": 0.02, "reasons": []}}), stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"ok": True}), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    updated = orchestrate_sales_lead_request(
        "Tìm lead dự án căn hộ ở Việt Nam cho Dukick, tối đa 3 lead",
        profile_name="sales",
        config=settings,
        pending=pending,
    )
    assert updated is not None and updated.pending is not None

    decision = orchestrate_sales_lead_request(
        "ok xác nhận",
        profile_name="sales",
        config=settings,
        pending=updated.pending,
    )

    assert decision is not None
    assert decision.clear_pending is True
    assert seen[0][1] == "dự án căn hộ"
    assert seen[1][1] == "dự án căn hộ"
    assert seen[2][1] == "dự án căn hộ"


def test_follow_up_stays_in_leadgen_preflight_when_pending_exists(monkeypatch, tmp_path):
    repo = _copy_fixture_repo(tmp_path)
    pending = LeadGenPendingRequest(
        request=LeadGenSearchRequest("gym", "VN", "dukick_tvc", 1, 1.0, 1),
        estimate_usd=0.02,
        preflight_ok=True,
        command_preview="leadgen-agent search-leads ...",
        created_at=1.0,
    )
    called = []

    def fake_run(cmd, **kwargs):
        called.append(cmd)
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps({"apify": {"ok": True, "estimated_cost_usd": 0.02, "reasons": []}}),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    decision = orchestrate_sales_lead_request(
        "spa đi",
        profile_name="sales",
        config=_settings(repo),
        pending=pending,
    )

    assert decision is not None
    assert decision.pending is not None
    assert decision.pending.request.query == "spa"
    assert called == [
        ["leadgen-agent", "smoke-plan", "--source", "apify"],
        ["leadgen-agent", "preflight", "--source", "apify"],
    ]


def test_follow_up_updates_specific_real_estate_alias(monkeypatch, tmp_path):
    repo = _copy_fixture_repo(tmp_path)
    pending = LeadGenPendingRequest(
        request=LeadGenSearchRequest("spa", "VN", "dukick_tvc", 1, 1.0, 1),
        estimate_usd=0.02,
        preflight_ok=True,
        command_preview="leadgen-agent search-leads ...",
        created_at=1.0,
    )

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps({"apify": {"ok": True, "estimated_cost_usd": 0.02, "reasons": []}}),
            stderr="",
        ),
    )
    decision = orchestrate_sales_lead_request(
        "đổi sang dự án căn hộ",
        profile_name="sales",
        config=_settings(repo),
        pending=pending,
    )

    assert decision is not None
    assert decision.pending is not None
    assert decision.pending.request.query == "dự án căn hộ"


def test_explicit_public_web_search_is_still_allowed_with_pending(tmp_path):
    repo = tmp_path / "lead-gen-agent"
    repo.mkdir()
    pending = LeadGenPendingRequest(
        request=LeadGenSearchRequest("spa", "VN", "dukick_tvc", 1, 1.0, 1),
        estimate_usd=0.02,
        preflight_ok=True,
        command_preview="leadgen-agent search-leads ...",
        created_at=1.0,
    )

    decision = orchestrate_sales_lead_request(
        "không dùng lead-gen-agent, search web thường",
        profile_name="sales",
        config=_settings(repo),
        pending=pending,
    )

    assert decision is None


def test_orchestrator_refuses_email_and_openai_live(tmp_path):
    settings = _settings(_copy_fixture_repo(tmp_path))

    email = orchestrate_sales_lead_request("gửi email luôn", profile_name="sales", config=settings, pending=None)
    openai_live = orchestrate_sales_lead_request("bật OpenAI live", profile_name="sales", config=settings, pending=None)

    assert email is not None and "không gửi email" in email.reply
    assert openai_live is not None and "không bật OpenAI live" in openai_live.reply


def test_conversation_loop_fast_path_skips_model_and_persists(monkeypatch):
    persisted = {}
    session_row = {"model_config": "{}", "model": "x"}
    agent = SimpleNamespace(
        language="vi",
        session_id="sess-1",
        _session_db=SimpleNamespace(
            get_session=lambda _sid: session_row,
            update_session_meta=lambda _sid, meta, model=None: session_row.update({"model_config": meta, "model": model}),
        ),
        _persist_session=lambda messages, conversation_history: persisted.update(
            {"messages": list(messages), "conversation_history": conversation_history}
        ),
    )
    messages = [{"role": "user", "content": "đọc lead mới nhất"}]

    monkeypatch.setattr(
        "hermes_cli.leadgen_agent_orchestrator.orchestrate_sales_lead_request",
        lambda *_args, **_kwargs: SimpleNamespace(reply="Đã đọc kết quả từ lead-gen-agent tại /tmp/repo.", pending=None, clear_pending=True),
    )
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "sales")

    result = _maybe_route_sales_lead_request(
        agent,
        user_message="đọc lead mới nhất",
        messages=messages,
        conversation_history=[],
    )

    assert result == {
        "final_response": "Đã đọc kết quả từ lead-gen-agent tại /tmp/repo.",
        "messages": messages,
        "api_calls": 0,
        "completed": True,
    }
    assert persisted["messages"][-1]["content"] == "Đã đọc kết quả từ lead-gen-agent tại /tmp/repo."


def test_repo_path_expands_tilde():
    settings = LeadGenAgentSettings(repo="~/CongThang/lead-gen-agent")
    assert str(settings.repo_path).endswith("CongThang/lead-gen-agent")


def test_runtime_settings_can_enable_allow_live_from_profile_config(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.leadgen_agent_integration.load_config",
        lambda: {
            "leadgen_agent": {
                "allow_live": True,
                "allow_dry_run": True,
                "max_live_results": 3,
                "default_country": "VN",
                "default_profile": "dukick_tvc",
            }
        },
    )
    monkeypatch.delenv("LEADGEN_AGENT_ALLOW_LIVE", raising=False)

    settings = LeadGenAgentSettings.from_runtime()

    assert settings.allow_live is True
    assert settings.allow_dry_run is True
    assert settings.max_live_results == 3


def test_runtime_env_override_still_wins_over_profile_config(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.leadgen_agent_integration.load_config",
        lambda: {"leadgen_agent": {"allow_live": True}},
    )
    monkeypatch.setenv("LEADGEN_AGENT_ALLOW_LIVE", "false")

    settings = LeadGenAgentSettings.from_runtime()

    assert settings.allow_live is False


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
    leadgen_parser = subparsers.choices["leadgen"]
    assert {"show", "summary", "read", "plan", "run-confirmed"} <= set(leadgen_parser._subparsers._group_actions[0].choices.keys())


def test_cli_plan_command_uses_preflight_runner(monkeypatch, tmp_path, capsys):
    repo = _copy_fixture_repo(tmp_path)
    monkeypatch.setenv("LEADGEN_AGENT_ALLOW_LIVE", "true")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(
                {
                    "request": {"request_id": "20260615-req-4"},
                    "apify": {
                        "ok": True,
                        "estimated_cost_usd": 0.02,
                        "reasons": [],
                        "details": {"planned_queries": ["spa", "showroom nội thất"]},
                    },
                }
            ),
            stderr="",
        ),
    )
    args = argparse.Namespace(
        leadgen_command="plan",
        repo=str(repo),
        language="vi",
        limit=1,
        mode="confirm_live",
        query="spa",
        country="VN",
        max_results=1,
        profile="dukick_tvc",
    )
    cmd_leadgen(args)
    out = capsys.readouterr().out
    assert "Ngành/query: spa" in out
    assert "Estimate: khoảng $0.02" in out
    assert "Dự kiến sẽ thử các query sau" in out
    assert "showroom nội thất" in out
    assert "Mã plan/run dự kiến: 20260615-req-4" in out
    assert "showroom nội thất" in out
    assert "Anh xác nhận `chạy live 1 lead`" in out


def test_preflight_reply_includes_planned_queries_and_confirmation(monkeypatch, tmp_path):
    repo = _copy_fixture_repo(tmp_path)

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(
                {
                    "request": {"request_id": "20260615-req-2"},
                    "apify": {
                        "ok": True,
                        "estimated_cost_usd": 0.28,
                        "reasons": [],
                        "details": {"planned_queries": ["chung cư mở bán", "booking căn hộ"]},
                    },
                }
            ),
            stderr="",
        ),
    )

    text = route_sales_lead_request(
        "Tìm lead chung cư mở bán ở Việt Nam cho Dukick, tối đa 3 lead",
        profile_name="sales",
        config=LeadGenAgentSettings(**{**_settings(repo).__dict__, "allow_live": True}),
    )

    assert text is not None
    assert "Em đã chuẩn bị tìm lead bằng Apify" in text
    assert "Ngành/query: chung cư mở bán" in text
    assert "Dự kiến sẽ thử các query sau" in text
    assert "booking căn hộ" in text
    assert "Mã plan/run dự kiến: 20260615-req-2" in text
    assert "Anh xác nhận `chạy live 3 lead`" in text


def test_post_run_summary_shows_review_lead_evidence_and_run_id(tmp_path):
    repo = tmp_path / "lead-gen-agent"
    (repo / "data" / "output").mkdir(parents=True)
    (repo / "data" / "output" / "hermes_leads.json").write_text(
        json.dumps(
            [
                {
                    "lead_id": "lead-1",
                    "company_name": "Eurowindow Light City",
                    "website": "https://eurowindowlightcity.vn/",
                    "domain": "eurowindowlightcity.vn",
                    "page_url": "https://www.facebook.com/eurowindowlightcity",
                    "source": "apify",
                    "source_id": "456",
                    "source_url": "https://www.facebook.com/ads/library/?q=booking%20c%C4%83n%20h%E1%BB%99",
                    "ad_title": "Eurowindow Light City - Booking căn hộ",
                    "ad_text_snippet": "Đăng ký tư vấn và tham quan nhà mẫu.",
                    "cta_text": "Learn more",
                    "page_category": "Real Estate",
                    "platforms": ["Facebook"],
                    "is_active": True,
                    "query_used": "booking căn hộ",
                    "matched_query": "booking căn hộ",
                    "query_mode": "known_industry",
                    "industry_match_decision": "weak_match",
                    "industry_match_score": 40,
                    "industry_match_reason": "Weak match for requested_query='chung cư mở bán'",
                    "query_match_decision": "match",
                    "query_match_score": 69,
                    "query_match_reason": "strong textual evidence for query 'chung cư mở bán'",
                    "query_match_evidence": ["booking căn hộ", "tư vấn", "dự án"],
                    "market_fit_decision": "local",
                    "market_fit_reason": "Vietnam-local evidence found.",
                    "relevance_score": 60,
                    "relevance_decision": "review",
                    "qualification": "review",
                    "relevance_subreasons": ["weak_industry_score"],
                    "relevance_evidence": ["dự án", "booking", "tư vấn"],
                    "evidence_summary": "company=Eurowindow Light City | landing_page=eurowindowlightcity.vn | category=Real Estate | cta=Learn more",
                    "suggested_video_angle": "Use walkthrough footage and project narration.",
                    "why_dukick_should_care": "Real-estate launch campaigns benefit from video walkthroughs.",
                    "raw_artifact_reference": "data/state/apify_raw/ad-456.json",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (repo / "data" / "output" / "demo_leads.md").write_text("# Dukick Lead Discovery Summary\n", encoding="utf-8")
    (repo / "data" / "output" / "report.json").write_text(
        json.dumps(
            {
                "run_id": "20260615081820-450b4c43",
                "status": "completed",
                "source": "apify",
                "profile_name": "dukick_tvc",
                "target_country": "VN",
                "lead_query": "chung cư mở bán",
                "requested_query": "chung cư mở bán",
                "query_mode": "known_industry",
                "counts": {
                    "raw_total_rows": 8,
                    "rejected_rows": 5,
                    "relevance_keep_rows": 0,
                    "relevance_review_rows": 1,
                    "relevance_reject_rows": 4,
                    "hermes_export_rows": 1,
                },
                "top_reject_reasons": {"irrelevant_icp": 4, "requested_query_mismatch": 1},
                "top_reject_subreasons": {"no_real_estate_evidence": 4, "query_mismatch": 1},
                "example_rejected_candidates": [
                    {
                        "company_name": "Ngọc Anh 12",
                        "reject_reason": "irrelevant_icp",
                        "query_used": "mở bán căn hộ",
                        "matched_query": "mở bán căn hộ",
                        "relevance_subreasons": ["no_real_estate_evidence", "no_commercial_cta"],
                        "relevance_evidence": ["page_category=Beauty Supplies"],
                    }
                ],
                "artifact_paths": {"run_output_dir": "data/output/runs/20260615081820-450b4c43"},
                "search_plan": {
                    "search_queries": [
                        "chung cư mở bán",
                        "mở bán căn hộ",
                        "dự án căn hộ",
                        "đăng ký tư vấn căn hộ",
                        "tham quan nhà mẫu",
                        "nhận bảng giá căn hộ",
                        "booking căn hộ",
                    ]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    text = route_sales_lead_request(
        "đọc lead mới nhất",
        profile_name="sales",
        config=_settings(repo),
    )

    assert text is not None
    assert "Run ID: 20260615081820-450b4c43" in text
    assert "Eurowindow Light City" in text
    assert "Trạng thái: Review" in text
    assert "Query tìm thấy: query_used=booking căn hộ" in text
    assert "matched_query=booking căn hộ" in text
    assert "Tín hiệu phù hợp: dự án, booking, tư vấn" in text
    assert "Lý do chưa xếp Keep" in text
    assert "Có snapshot audit cho run này trong data/output/runs/20260615081820-450b4c43" in text


def test_empty_result_diagnostics_surface_reject_reasons_and_examples(tmp_path):
    repo = tmp_path / "lead-gen-agent"
    (repo / "data" / "output").mkdir(parents=True)
    (repo / "data" / "output" / "hermes_leads.json").write_text("[]", encoding="utf-8")
    (repo / "data" / "output" / "demo_leads.md").write_text("# Dukick Lead Discovery Summary\n", encoding="utf-8")
    (repo / "data" / "output" / "report.json").write_text(
        json.dumps(
            {
                "run_id": "20260615090000-abcdef12",
                "status": "completed",
                "source": "apify",
                "profile_name": "dukick_tvc",
                "target_country": "VN",
                "lead_query": "chung cư mở bán",
                "requested_query": "chung cư mở bán",
                "query_mode": "known_industry",
                "counts": {
                    "raw_total_rows": 8,
                    "rejected_rows": 5,
                    "relevance_keep_rows": 0,
                    "relevance_review_rows": 0,
                    "relevance_reject_rows": 4,
                    "hermes_export_rows": 0,
                },
                "top_reject_reasons": {"irrelevant_icp": 4, "requested_query_mismatch": 1},
                "top_reject_subreasons": {
                    "no_real_estate_evidence": 4,
                    "no_commercial_cta": 3,
                    "query_mismatch": 1,
                },
                "example_rejected_candidates": [
                    {
                        "company_name": "Ngọc Anh 12",
                        "reject_reason": "irrelevant_icp",
                        "query_used": "mở bán căn hộ",
                        "relevance_subreasons": ["no_real_estate_evidence", "no_commercial_cta"],
                        "relevance_evidence": ["page_category=Beauty Supplies"],
                    }
                ],
                "search_plan": {
                    "search_queries": [
                        "chung cư mở bán",
                        "mở bán căn hộ",
                        "dự án căn hộ",
                    ]
                },
                "artifact_paths": {"run_output_dir": "data/output/runs/20260615090000-abcdef12"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    text = route_sales_lead_request(
        "đọc lead mới nhất",
        profile_name="sales",
        config=_settings(repo),
    )

    assert text is not None
    assert "Chưa có lead đủ điều kiện export." in text
    assert "Raw candidates: 8" in text
    assert "Top reject reasons: irrelevant_icp=4, requested_query_mismatch=1" in text
    assert "Top reject subreasons:" in text
    assert "Ngọc Anh 12" in text
    assert "Suggested next queries:" in text


def test_backward_compatibility_missing_new_fields_still_summarizes(tmp_path):
    repo = tmp_path / "lead-gen-agent"
    (repo / "data" / "output").mkdir(parents=True)
    (repo / "data" / "output" / "hermes_leads.json").write_text(
        json.dumps([{"company_name": "Legacy Lead", "qualification": "review"}], ensure_ascii=False),
        encoding="utf-8",
    )
    (repo / "data" / "output" / "demo_leads.md").write_text("# Dukick Lead Discovery Summary\n", encoding="utf-8")

    text = LeadGenAgentAdapter(_settings(repo)).summarize_for_user()

    assert "Legacy Lead" in text
    assert "Trạng thái: Review" in text
