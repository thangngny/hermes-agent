from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from hermes_cli.leadgen_agent_integration import (
    LeadGenAgentAdapter,
    LeadGenAgentSettings,
    looks_like_explicit_public_web_search_request,
    looks_like_live_request,
    refuse_live_execution_message,
)

_SEARCH_INTENT_RE = re.compile(r"(?iu)\b(tìm|khảo\s*sát|search)\b")
_READ_INTENT_RE = re.compile(r"(?iu)\b(đọc|xem|show|read|summary|danh\s*sách)\b")
_LEAD_RE = re.compile(r"(?iu)\b(lead|khách\s*hàng\s*tiềm\s*năng|prospect(?:s)?)\b")
_CONFIRM_RE = re.compile(
    r"(?iu)\b(xác\s*nhận|ok(?:ay)?|đồng\s*ý|chạy(?:\s*live)?|chạy\s*apify|tôi\s*xác\s*nhận)\b"
)
_NEGATED_CONFIRM_RE = re.compile(r"(?iu)\b(chưa\s*xác\s*nhận|không\s*xác\s*nhận|đừng\s*chạy|không\s*chạy)\b")
_NO_CONFIRM_RE = re.compile(r"(?iu)\b(không\s*cần\s*hỏi|tự\s*chạy\s*luôn|khỏi\s*cần\s*xác\s*nhận)\b")
_EMAIL_RE = re.compile(r"(?iu)\b(email|gmail|gửi\s*mail|gửi\s*email)\b")
_OPENAI_LIVE_RE = re.compile(r"(?iu)\b(openai\s*live|bật\s*openai\s*live)\b")
_READ_LATEST_RE = re.compile(
    r"(?iu)\b(đọc\s*lead\s*mới\s*nhất|xem\s*danh\s*sách\s*lead|lead\s*mới\s*cho\s*dukick|đọc\s*kết\s*quả\s*lead-gen-agent)\b"
)
_MAX_RESULTS_RE = re.compile(r"(?iu)\b(\d+)\s*lead\b")
_MAX_RESULTS_CONTROL_RE = re.compile(r"(?iu)(?:,|\s)*tối\s*đa\s*\d+\s*lead\b|(?:,|\s)*\d+\s*lead\b")
_COUNTRY_VN_RE = re.compile(r"(?iu)\b(việt\s*nam|vn|hà\s*nội|hồ\s*chí\s*minh|sài\s*gòn)\b")
_COUNTRY_US_RE = re.compile(r"(?iu)\b(us|usa|mỹ|hoa\s*kỳ|united\s*states)\b")
_QUERY_REFINEMENT_RE = re.compile(r"(?iu)\b(ngành|đổi|chuyển|sang)\b")
_QUERY_STOPWORDS_RE = re.compile(
    r"(?iu)\b("
    r"tìm|khảo\s*sát|search|lead|khách\s*hàng(?:\s*tiềm\s*năng)?|"
    r"cho|dukick(?:\s*studio)?|ở|tại|việt\s*nam|vn|mỹ|us|usa|"
    r"tối\s*đa|giúp\s*tôi|ngành|đổi|chuyển|sang|đi"
    r")\b"
)
_QUERY_TRAILING_PUNCT_RE = re.compile(r"[\s,;:.-]+$")

_QUERY_ALIASES: tuple[tuple[str, str], ...] = (
    ("thẩm mỹ viện", "thẩm mỹ viện"),
    ("thẩm mỹ", "thẩm mỹ"),
    ("nha khoa", "nha khoa"),
    ("mỹ phẩm", "mỹ phẩm"),
    ("skincare", "skincare"),
    ("spa", "spa"),
    ("gym", "gym"),
    ("supplement", "supplement"),
    ("nhà hàng", "nhà hàng"),
    ("bất động sản", "bất động sản"),
    ("bđs", "bđs"),
    ("bds", "bds"),
    ("dự án căn hộ", "dự án căn hộ"),
    ("chung cư mở bán", "chung cư mở bán"),
    ("căn hộ cao cấp", "căn hộ cao cấp"),
    ("dự án chung cư", "dự án chung cư"),
    ("căn hộ", "căn hộ"),
    ("nhà phố", "nhà phố"),
    ("shophouse", "shophouse"),
    ("biệt thự", "biệt thự"),
    ("villa", "villa"),
    ("đất nền", "đất nền"),
    ("khu đô thị", "khu đô thị"),
    ("real estate", "real estate"),
    ("property", "property"),
    ("khóa học", "khóa học"),
)

_PENDING_MODEL_CONFIG_KEY = "_leadgen_pending"
_CLI_BIN = "leadgen-agent"


@dataclass(frozen=True)
class LeadGenSearchRequest:
    query: str
    country: str
    profile: str
    max_results: int
    spend_limit_usd: float
    requested_max_results: int
    capped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LeadGenPreflightResult:
    request: LeadGenSearchRequest
    estimate_usd: float | None
    ok: bool
    request_id: str | None
    planned_queries: tuple[str, ...]
    smoke_plan: dict[str, Any] | None
    preflight: dict[str, Any] | None
    reasons: list[str]
    command_preview: str


@dataclass(frozen=True)
class LeadGenPendingRequest:
    request: LeadGenSearchRequest
    estimate_usd: float | None
    preflight_ok: bool
    command_preview: str
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["request"] = self.request.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LeadGenPendingRequest | None":
        try:
            request_raw = payload.get("request", {})
            request = LeadGenSearchRequest(
                query=str(request_raw.get("query") or "").strip(),
                country=str(request_raw.get("country") or "VN").strip() or "VN",
                profile=str(request_raw.get("profile") or "dukick_tvc").strip() or "dukick_tvc",
                max_results=int(request_raw.get("max_results") or 1),
                spend_limit_usd=float(request_raw.get("spend_limit_usd") or 1.0),
                requested_max_results=int(request_raw.get("requested_max_results") or request_raw.get("max_results") or 1),
                capped=bool(request_raw.get("capped")),
            )
            return cls(
                request=request,
                estimate_usd=float(payload["estimate_usd"]) if payload.get("estimate_usd") not in (None, "") else None,
                preflight_ok=bool(payload.get("preflight_ok")),
                command_preview=str(payload.get("command_preview") or ""),
                created_at=float(payload.get("created_at") or time.time()),
            )
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True)
class LeadGenRouteDecision:
    reply: str
    pending: LeadGenPendingRequest | None = None
    clear_pending: bool = False


class LeadGenAgentRunner:
    def __init__(self, config: LeadGenAgentSettings | None = None):
        self.config = config or LeadGenAgentSettings.from_runtime()

    def preflight_search(self, request: LeadGenSearchRequest) -> LeadGenPreflightResult:
        smoke = self._run_cli(["smoke-plan", "--source", "apify"], request=request, dry_run=True)
        preflight = self._run_cli(["preflight", "--source", "apify"], request=request, dry_run=True)
        smoke_json = _safe_json_loads(smoke.stdout)
        preflight_json = _safe_json_loads(preflight.stdout)
        apify = _extract_apify_section(preflight_json) or _extract_apify_section(smoke_json)
        estimate = _extract_estimate_usd(preflight_json, smoke_json)
        reasons = list(apify.get("reasons", [])) if isinstance(apify, Mapping) else []
        ok = bool(apify.get("ok")) if isinstance(apify, Mapping) else (preflight.returncode == 0)
        if preflight.returncode != 0 and not reasons:
            reasons.append("preflight_failed")
        preview = (
            f"{_CLI_BIN} search-leads --profile {request.profile} --source apify "
            f"--country {request.country} --query {request.query} --max-results {request.max_results} --live"
        )
        return LeadGenPreflightResult(
            request=request,
            estimate_usd=estimate,
            ok=ok,
            request_id=_extract_request_id(preflight_json, smoke_json),
            planned_queries=tuple(_extract_planned_queries(preflight_json, smoke_json)),
            smoke_plan=smoke_json if isinstance(smoke_json, dict) else None,
            preflight=preflight_json if isinstance(preflight_json, dict) else None,
            reasons=reasons,
            command_preview=preview,
        )

    def format_preflight_reply(self, preflight: LeadGenPreflightResult) -> str:
        return _build_preflight_reply(preflight, config=self.config)

    def run_live_search(self, request: LeadGenSearchRequest, *, confirmation: str) -> str:
        if not looks_like_confirmation_message(confirmation):
            raise RuntimeError("missing_confirmation")
        if not self.config.allow_live:
            raise RuntimeError("live_disabled")

        self._run_cli(
            [
                "search-leads",
                "--profile", request.profile,
                "--source", "apify",
                "--country", request.country,
                "--query", request.query,
                "--max-results", str(request.max_results),
                "--live",
            ],
            request=request,
            dry_run=False,
            require_success=True,
        )
        self._run_cli(["demo-summary"], request=request, dry_run=False, require_success=True)
        adapter = LeadGenAgentAdapter(self.config)
        export_result = adapter.load_exports()
        summary = adapter.summarize_for_user(language="vi", limit=request.max_results)
        if export_result.ok and export_result.lead_count > 0:
            prefix = f"Đã chạy xong search mới. Tìm được {export_result.lead_count} lead đủ điều kiện export cho Hermes."
        else:
            prefix = "Đã chạy xong search mới, nhưng chưa có lead nào đủ điều kiện export cho Hermes."
        return f"{prefix}\n\n{summary}"

    def read_latest(self, *, limit: int = 5, language: str = "vi") -> str:
        return LeadGenAgentAdapter(self.config).summarize_for_user(language=language, limit=limit)

    def _run_cli(
        self,
        args: list[str],
        *,
        request: LeadGenSearchRequest,
        dry_run: bool,
        require_success: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(
            {
                "LEADGEN_AGENT_REPO": str(self.config.repo_path),
                "LEADGEN_AGENT_MODE": self.config.mode,
                "DRY_RUN": "true" if dry_run else "false",
                "SEND_MODE": "draft_only",
                "AI_MODE": "mock",
                "DISCOVERY_ONLY": "true",
                "DRAFTS_ENABLED": "false",
                "APIFY_ENABLED": "true",
                "OPENAI_ENABLED": "false",
                "SEND_ENABLED": "false",
                "LEAD_PROFILE": request.profile,
                "TARGET_COUNTRY": request.country,
                "LEAD_QUERY": request.query,
                "LEAD_QUERIES": request.query,
                "APIFY_MAX_RESULTS": str(request.max_results),
                "APIFY_SPEND_LIMIT_USD": str(request.spend_limit_usd),
            }
        )
        completed = subprocess.run(
            [_CLI_BIN, *args],
            cwd=self.config.repo_path,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if require_success and completed.returncode != 0:
            stderr = _replace_secretish(completed.stderr or completed.stdout or "")
            raise RuntimeError(stderr.strip() or "leadgen_command_failed")
        return completed


def parse_lead_search_request(
    user_text: str,
    *,
    config: LeadGenAgentSettings | None = None,
) -> LeadGenSearchRequest | None:
    config = config or LeadGenAgentSettings.from_runtime()
    text = (user_text or "").strip()
    if not text:
        return None

    query = _extract_query(text)
    if not query:
        return None

    requested_max_results = _extract_max_results(text)
    max_results = requested_max_results
    capped = False
    if max_results > config.max_live_results:
        max_results = config.max_live_results
        capped = True

    return LeadGenSearchRequest(
        query=query,
        country=_extract_country(text, default=config.default_country),
        profile=config.default_profile,
        max_results=max_results,
        spend_limit_usd=config.spend_limit_usd,
        requested_max_results=requested_max_results,
        capped=capped,
    )


def looks_like_confirmation_message(text: str) -> bool:
    if _NEGATED_CONFIRM_RE.search(text or ""):
        return False
    return bool(_CONFIRM_RE.search(text or ""))


def looks_like_read_latest_request(text: str) -> bool:
    text = text or ""
    if _READ_LATEST_RE.search(text):
        return True
    return bool(_LEAD_RE.search(text) and _READ_INTENT_RE.search(text) and not _SEARCH_INTENT_RE.search(text))


def looks_like_search_request(text: str) -> bool:
    return bool(_LEAD_RE.search(text or "") and _SEARCH_INTENT_RE.search(text or ""))


def route_sales_lead_request(
    user_text: str,
    *,
    language: str = "vi",
    profile_name: str = "",
    config: LeadGenAgentSettings | None = None,
) -> str | None:
    decision = orchestrate_sales_lead_request(
        user_text,
        language=language,
        profile_name=profile_name,
        config=config,
        pending=None,
    )
    return decision.reply if decision else None


def orchestrate_sales_lead_request(
    user_text: str,
    *,
    language: str = "vi",
    profile_name: str,
    config: LeadGenAgentSettings | None = None,
    pending: LeadGenPendingRequest | None,
) -> LeadGenRouteDecision | None:
    config = config or LeadGenAgentSettings.from_runtime()
    if (profile_name or "").strip().lower() != "sales":
        return None

    text = user_text or ""
    if _EMAIL_RE.search(text):
        return LeadGenRouteDecision("Em không gửi email hoặc kết nối Gmail trong flow leadgen này.", clear_pending=False)
    if _OPENAI_LIVE_RE.search(text):
        return LeadGenRouteDecision("Em không bật OpenAI live trong flow này. Flow hiện tại chỉ preflight/run Apify với gate xác nhận.", clear_pending=False)
    if _NO_CONFIRM_RE.search(text):
        return LeadGenRouteDecision("Em không tự chạy live mà không hỏi trước. Live Apify luôn cần xác nhận rõ ràng.", clear_pending=False)
    if looks_like_live_request(text) and not pending:
        return LeadGenRouteDecision(refuse_live_execution_message(language=language))

    runner = LeadGenAgentRunner(config)

    if looks_like_confirmation_message(text):
        if pending is None:
            return LeadGenRouteDecision("Chưa có yêu cầu lead nào đang chờ xác nhận. Hãy nói rõ query như: `Tìm lead spa ở Việt Nam cho Dukick`.", clear_pending=True)
        if not config.allow_live:
            return LeadGenRouteDecision(
                "Em đã có plan nhưng live Apify hiện đang bị tắt trong Hermes (`LEADGEN_AGENT_ALLOW_LIVE=false`). "
                "Em chỉ có thể preflight và đọc export hiện có.",
                pending=pending,
                clear_pending=False,
            )
        reply = runner.run_live_search(pending.request, confirmation=text)
        return LeadGenRouteDecision(reply, clear_pending=True)

    if looks_like_read_latest_request(text):
        if looks_like_explicit_public_web_search_request(text):
            result = LeadGenAgentAdapter(config).load_exports()
            if result.errors or not result.leads:
                return None
        return LeadGenRouteDecision(runner.read_latest(language=language, limit=5), clear_pending=False)

    if looks_like_explicit_public_web_search_request(text) and _READ_INTENT_RE.search(text) and _LEAD_RE.search(text):
        result = LeadGenAgentAdapter(config).load_exports()
        if result.errors or not result.leads:
            return None
        return LeadGenRouteDecision(runner.read_latest(language=language, limit=5), clear_pending=False)

    if looks_like_explicit_public_web_search_request(text) and _SEARCH_INTENT_RE.search(text) and _LEAD_RE.search(text):
        return None

    if looks_like_search_request(text):
        request = parse_lead_search_request(text, config=config)
        if request is None:
            return LeadGenRouteDecision(
                "Em cần rõ ngành/query trước khi lên plan. Ví dụ: `Tìm lead spa ở Việt Nam`, `Tìm 3 lead gym`, hoặc `Tìm lead nha khoa Hà Nội`.",
                clear_pending=False,
            )

        preflight = runner.preflight_search(request)
        pending_request = LeadGenPendingRequest(
            request=request,
            estimate_usd=preflight.estimate_usd,
            preflight_ok=preflight.ok,
            command_preview=preflight.command_preview,
            created_at=time.time(),
        )
        return LeadGenRouteDecision(
            _build_preflight_reply(preflight, config=config),
            pending=pending_request,
            clear_pending=False,
        )

    if pending is not None and not looks_like_explicit_public_web_search_request(text):
        refined_request = refine_pending_search_request(text, pending=pending, config=config)
        if refined_request is not None:
            preflight = runner.preflight_search(refined_request)
            pending_request = LeadGenPendingRequest(
                request=refined_request,
                estimate_usd=preflight.estimate_usd,
                preflight_ok=preflight.ok,
                command_preview=preflight.command_preview,
                created_at=time.time(),
            )
            return LeadGenRouteDecision(
                _build_refinement_reply(previous=pending.request, preflight=preflight, config=config),
                pending=pending_request,
                clear_pending=False,
            )

    if not looks_like_search_request(text):
        return None
    return None


def load_pending_search(agent) -> LeadGenPendingRequest | None:
    pending = getattr(agent, "_pending_leadgen_search", None)
    if isinstance(pending, LeadGenPendingRequest):
        return pending
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", "")
    if not session_db or not session_id:
        return None
    try:
        row = session_db.get_session(session_id) or {}
        model_config = json.loads(row.get("model_config") or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    restored = LeadGenPendingRequest.from_dict(model_config.get(_PENDING_MODEL_CONFIG_KEY, {}))
    if restored is not None:
        agent._pending_leadgen_search = restored
    return restored


def save_pending_search(agent, pending: LeadGenPendingRequest | None) -> None:
    agent._pending_leadgen_search = pending
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", "")
    if not session_db or not session_id:
        return
    try:
        row = session_db.get_session(session_id) or {}
        model_config = json.loads(row.get("model_config") or "{}")
        if not isinstance(model_config, dict):
            model_config = {}
        if pending is None:
            model_config.pop(_PENDING_MODEL_CONFIG_KEY, None)
        else:
            model_config[_PENDING_MODEL_CONFIG_KEY] = pending.to_dict()
        session_db.update_session_meta(session_id, json.dumps(model_config), model=row.get("model"))
    except Exception:
        return


def _build_preflight_reply(preflight: LeadGenPreflightResult, *, config: LeadGenAgentSettings) -> str:
    req = preflight.request
    estimate = f"${preflight.estimate_usd:.2f}" if preflight.estimate_usd is not None else "chưa rõ"
    lines = [
        "Em đã chuẩn bị tìm lead bằng Apify:",
        "",
        f"* Ngành/query: {req.query}",
        f"* Quốc gia: {req.country}",
        f"* ICP profile: {req.profile}",
        f"* Số lead tối đa: {req.max_results}",
    ]
    if req.capped:
        lines.append(
            f"* Ghi chú cap: anh yêu cầu {req.requested_max_results} lead nhưng Hermes đang cap ở {config.max_live_results} lead cho live run."
        )
    lines.extend(
        [
            f"* Estimate: khoảng {estimate}",
            "* Chế độ: chưa chạy live",
        ]
    )
    if preflight.request_id:
        lines.append(f"* Mã plan/run dự kiến: {preflight.request_id}")
    if preflight.planned_queries:
        lines.append("* Dự kiến sẽ thử các query sau:")
        lines.extend(f"  - {query}" for query in preflight.planned_queries)
    if preflight.reasons:
        lines.append(f"* Preflight: {'; '.join(preflight.reasons)}")
    if not config.allow_live:
        lines.append("* Live Apify hiện đang bị tắt trong Hermes. Nếu muốn bật, cần mở gate `LEADGEN_AGENT_ALLOW_LIVE=true` trước.")
    else:
        lines.append(f"* Lệnh live sau khi xác nhận: {preflight.command_preview}")
        lines.append(f"* Anh xác nhận `chạy live {req.max_results} lead` thì em mới chạy.")
    return "\n".join(lines)


def _extract_query(text: str) -> str:
    lowered = (text or "").lower()
    for alias, canonical in _QUERY_ALIASES:
        if alias in lowered:
            return canonical

    cleaned = _strip_query_control_phrases(lowered)
    return cleaned[:80].strip()


def _extract_country(text: str, *, default: str) -> str:
    if _COUNTRY_US_RE.search(text or ""):
        return "US"
    if _COUNTRY_VN_RE.search(text or ""):
        return "VN"
    return default


def _extract_country_override(text: str) -> str | None:
    if _COUNTRY_US_RE.search(text or ""):
        return "US"
    if _COUNTRY_VN_RE.search(text or ""):
        return "VN"
    return None


def _extract_max_results(text: str) -> int:
    match = _MAX_RESULTS_RE.search(text or "")
    if not match:
        return 1
    try:
        value = int(match.group(1))
    except ValueError:
        return 1
    return max(value, 1)


def refine_pending_search_request(
    text: str,
    *,
    pending: LeadGenPendingRequest,
    config: LeadGenAgentSettings,
) -> LeadGenSearchRequest | None:
    query = _extract_followup_query(text) or pending.request.query
    country = _extract_country_override(text) or pending.request.country
    requested_max_results = pending.request.requested_max_results
    if _MAX_RESULTS_RE.search(text or ""):
        requested_max_results = _extract_max_results(text)

    max_results = requested_max_results
    capped = False
    if max_results > config.max_live_results:
        max_results = config.max_live_results
        capped = True

    updated = LeadGenSearchRequest(
        query=query,
        country=country,
        profile=pending.request.profile,
        max_results=max_results,
        spend_limit_usd=pending.request.spend_limit_usd,
        requested_max_results=requested_max_results,
        capped=capped,
    )
    if updated == pending.request:
        return None
    return updated


def _extract_followup_query(text: str) -> str:
    lowered = (text or "").lower().strip()
    if not lowered:
        return ""
    for alias, canonical in _QUERY_ALIASES:
        if alias in lowered:
            return canonical
    if not _QUERY_REFINEMENT_RE.search(lowered):
        return ""
    cleaned = _strip_query_control_phrases(lowered)
    return cleaned[:80].strip()


def _build_refinement_reply(
    *,
    previous: LeadGenSearchRequest,
    preflight: LeadGenPreflightResult,
    config: LeadGenAgentSettings,
) -> str:
    updates: list[str] = []
    current = preflight.request
    if current.query != previous.query:
        updates.append(f"đổi ngành sang {current.query}")
    if current.country != previous.country:
        updates.append(f"đổi quốc gia sang {current.country}")
    if current.requested_max_results != previous.requested_max_results:
        updates.append(f"đổi số lead tối đa sang {current.requested_max_results}")

    if not updates:
        return _build_preflight_reply(preflight, config=config)

    prefix = "Em đã " + " và ".join(updates) + " và chuẩn bị tìm lead bằng Apify:"
    body = _build_preflight_reply(preflight, config=config).split("\n", 1)
    if len(body) == 2:
        return prefix + "\n\n" + body[1]
    return prefix


def _safe_json_loads(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _extract_apify_section(payload: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    apify = payload.get("apify")
    return apify if isinstance(apify, Mapping) else None


def _extract_estimate_usd(*payloads: Mapping[str, Any] | None) -> float | None:
    for payload in payloads:
        apify = _extract_apify_section(payload)
        if isinstance(apify, Mapping) and apify.get("estimated_cost_usd") not in (None, ""):
            try:
                return float(apify["estimated_cost_usd"])
            except (TypeError, ValueError):
                continue
        if isinstance(payload, Mapping) and payload.get("estimated_cost_usd") not in (None, ""):
            try:
                return float(payload["estimated_cost_usd"])
            except (TypeError, ValueError):
                continue
    return None


def _extract_request_id(*payloads: Mapping[str, Any] | None) -> str | None:
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        for path in (
            ("request", "request_id"),
            ("result", "request", "request_id"),
            ("result", "run_id"),
            ("run_id",),
            ("report", "run_id"),
            ("search_plan", "request_id"),
            ("preflight", "discovery", "request_id"),
            ("apify", "request_id"),
            ("apify", "details", "request_id"),
        ):
            value = _mapping_path(payload, *path)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _extract_planned_queries(*payloads: Mapping[str, Any] | None) -> list[str]:
    collected: list[str] = []
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        for path in (
            ("search_plan", "search_queries"),
            ("search_plan", "queries"),
            ("result", "request", "queries"),
            ("result", "search_plan", "search_queries"),
            ("result", "report", "preflight", "apify_plan", "details", "planned_queries"),
            ("result", "report", "preflight", "apify_plan", "input_preview", "planned_queries"),
            ("result", "report", "preflight", "discovery", "lead_queries"),
            ("apify", "details", "planned_queries"),
            ("apify", "details", "queries"),
            ("apify", "planned_queries"),
            ("preflight", "apify_plan", "details", "planned_queries"),
            ("preflight", "apify_plan", "input_preview", "planned_queries"),
            ("apify_plan", "details", "planned_queries"),
            ("apify_plan", "input_preview", "planned_queries"),
            ("apify_plan", "planned_queries"),
            ("discovery", "lead_queries"),
            ("discovery", "queries"),
            ("request", "queries"),
        ):
            value = _mapping_path(payload, *path)
            if isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        cleaned = item.strip()
                        if cleaned not in collected:
                            collected.append(cleaned)
    return collected


def _mapping_path(payload: Mapping[str, Any], *path: str) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _replace_secretish(text: str) -> str:
    if not text:
        return text
    return re.sub(r"(?i)(api[_-]?key|token|secret|bearer|authorization)[^\s]*", "[REDACTED]", text)


def _strip_query_control_phrases(text: str) -> str:
    cleaned = _MAX_RESULTS_CONTROL_RE.sub(" ", text or "")
    cleaned = re.sub(r"(?iu)(?:,|\s)*(ở|tại)\s*(việt\s*nam|vn|mỹ|us|usa)\b", " ", cleaned)
    cleaned = re.sub(r"(?iu)(?:,|\s)*cho\s*dukick(?:\s*studio)?\b", " ", cleaned)
    cleaned = _QUERY_STOPWORDS_RE.sub(" ", cleaned)
    cleaned = re.sub(r"(?iu)\b\d+\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = _QUERY_TRAILING_PUNCT_RE.sub("", cleaned)
    return cleaned.strip()
