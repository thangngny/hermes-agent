from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

try:
    from hermes_cli.config import load_config
except Exception:  # pragma: no cover - config import should normally work
    load_config = None  # type: ignore[assignment]

try:
    from utils import is_truthy_value
except Exception:  # pragma: no cover - fallback for minimal test import paths
    def is_truthy_value(value: Any, default: bool = False) -> bool:  # type: ignore[no-redef]
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return default


DEFAULT_REPO = "~/CongThang/lead-gen-agent"
DEFAULT_JSON = "data/output/hermes_leads.json"
DEFAULT_CSV = "data/output/hermes_leads.csv"
DEFAULT_DEMO = "data/output/demo_leads.md"
DEFAULT_REPORT = "data/output/report.json"
DEFAULT_MODE = "confirm_live"
DEFAULT_ALLOW_DRY_RUN = True
DEFAULT_ALLOW_LIVE = False
DEFAULT_MAX_LIVE_RESULTS = 3
DEFAULT_DEFAULT_COUNTRY = "VN"
DEFAULT_DEFAULT_PROFILE = "dukick_tvc"
DEFAULT_SPEND_LIMIT_USD = 1.0
STALE_REPORT_AFTER = timedelta(hours=24)

_SECRETISH_RE = re.compile(r"(?i)(api[_-]?key|token|secret|bearer|authorization)")
_LIVE_REQUEST_RE = re.compile(
    r"(?i)\b(run\s*apify|apify\s*live|live\s*apify|\bquét mới\b|\btìm lead mới\b|chạy live|run live|live search)\b"
)
_LEAD_SUBJECT_RE = re.compile(
    r"(?iu)\b(lead(?:-gen-agent)?|khách\s*hàng\s*tiềm\s*năng|prospect(?:s)?)\b"
)
_LEAD_ACTION_RE = re.compile(
    r"(?iu)\b(tìm|khảo\s*sát|đọc|xem|show|read|summary|danh\s*sách|mới|mới\s*nhất|latest)\b"
)
_DUKICK_RE = re.compile(r"(?iu)\bdukick(?:\s*studio)?\b")
_EXPLICIT_PUBLIC_WEB_SEARCH_RE = re.compile(
    r"(?iu)\b(public\s+web\s+search|web\s+search|google\s+search|search\s+google|"
    r"tìm\s+trên\s+google|tra\s+google|web\s+công\s+khai|tìm\s+trên\s+web\s+công\s+khai)\b"
)


@dataclass(frozen=True)
class LeadGenAgentSettings:
    repo: str = DEFAULT_REPO
    hermes_json: str = DEFAULT_JSON
    hermes_csv: str = DEFAULT_CSV
    demo_summary: str = DEFAULT_DEMO
    report: str = DEFAULT_REPORT
    mode: str = DEFAULT_MODE
    allow_dry_run: bool = DEFAULT_ALLOW_DRY_RUN
    allow_live: bool = DEFAULT_ALLOW_LIVE
    max_live_results: int = DEFAULT_MAX_LIVE_RESULTS
    default_country: str = DEFAULT_DEFAULT_COUNTRY
    default_profile: str = DEFAULT_DEFAULT_PROFILE
    spend_limit_usd: float = DEFAULT_SPEND_LIMIT_USD

    @property
    def repo_path(self) -> Path:
        return Path(self.repo).expanduser().resolve()

    def resolve_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (self.repo_path / path).resolve()

    @property
    def hermes_json_path(self) -> Path:
        return self.resolve_path(self.hermes_json)

    @property
    def hermes_csv_path(self) -> Path:
        return self.resolve_path(self.hermes_csv)

    @property
    def demo_summary_path(self) -> Path:
        return self.resolve_path(self.demo_summary)

    @property
    def report_path(self) -> Path:
        return self.resolve_path(self.report)

    @classmethod
    def from_runtime(cls) -> "LeadGenAgentSettings":
        config_section: Mapping[str, Any] = {}
        if load_config is not None:
            try:
                config = load_config() or {}
                maybe_section = config.get("leadgen_agent", {}) if isinstance(config, Mapping) else {}
                if isinstance(maybe_section, Mapping):
                    config_section = maybe_section
            except Exception:
                config_section = {}

        def _pick(name: str, key: str, default: Any) -> Any:
            env_value = os.environ.get(name)
            if env_value not in (None, ""):
                return env_value
            value = config_section.get(key, default)
            return default if value in (None, "") else value

        return cls(
            repo=str(_pick("LEADGEN_AGENT_REPO", "repo", DEFAULT_REPO)),
            hermes_json=str(_pick("LEADGEN_AGENT_HERMES_JSON", "hermes_json", DEFAULT_JSON)),
            hermes_csv=str(_pick("LEADGEN_AGENT_HERMES_CSV", "hermes_csv", DEFAULT_CSV)),
            demo_summary=str(_pick("LEADGEN_AGENT_DEMO_SUMMARY", "demo_summary", DEFAULT_DEMO)),
            report=str(_pick("LEADGEN_AGENT_REPORT", "report", DEFAULT_REPORT)),
            mode=str(_pick("LEADGEN_AGENT_MODE", "mode", DEFAULT_MODE)),
            allow_dry_run=is_truthy_value(_pick("LEADGEN_AGENT_ALLOW_DRY_RUN", "allow_dry_run", DEFAULT_ALLOW_DRY_RUN), default=DEFAULT_ALLOW_DRY_RUN),
            allow_live=is_truthy_value(_pick("LEADGEN_AGENT_ALLOW_LIVE", "allow_live", DEFAULT_ALLOW_LIVE), default=DEFAULT_ALLOW_LIVE),
            max_live_results=int(_pick("LEADGEN_AGENT_MAX_LIVE_RESULTS", "max_live_results", DEFAULT_MAX_LIVE_RESULTS) or DEFAULT_MAX_LIVE_RESULTS),
            default_country=str(_pick("LEADGEN_AGENT_DEFAULT_COUNTRY", "default_country", DEFAULT_DEFAULT_COUNTRY)),
            default_profile=str(_pick("LEADGEN_AGENT_DEFAULT_PROFILE", "default_profile", DEFAULT_DEFAULT_PROFILE)),
            spend_limit_usd=float(_pick("LEADGEN_AGENT_SPEND_LIMIT_USD", "spend_limit_usd", DEFAULT_SPEND_LIMIT_USD) or DEFAULT_SPEND_LIMIT_USD),
        )


@dataclass(frozen=True)
class LeadGenLead:
    lead_id: str
    company_name: str
    website: str = ""
    domain: str = ""
    page_url: str = ""
    source: str = ""
    source_id: str = ""
    source_url: str = ""
    ad_title: str = ""
    ad_text_snippet: str = ""
    cta_text: str = ""
    page_category: str = ""
    platforms: tuple[str, ...] = ()
    is_active: bool | None = None
    icp_profile: str = ""
    icp_fit_score: float | None = None
    creative_opportunity_score: float | None = None
    creative_opportunity_reason: str = ""
    market_fit_score: float | None = None
    market_fit_decision: str = ""
    market_fit_reason: str = ""
    relevance_score: float | None = None
    relevance_decision: str = ""
    qualification: str = ""
    evidence_summary: str = ""
    suggested_video_angle: str = ""
    why_dukick_should_care: str = ""
    raw_artifact_reference: str = ""


@dataclass(frozen=True)
class LeadGenAgentExportResult:
    ok: bool
    source_repo: str
    report_run_id: str | None
    report_started_at: str | None
    report_completed_at: str | None
    report_status: str | None
    source: str | None
    profile_name: str | None
    target_country: str | None
    lead_query: str | None
    lead_count: int
    leads: list[LeadGenLead] = field(default_factory=list)
    demo_summary_text: str = ""
    report: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "leads": [asdict(lead) for lead in self.leads],
        }


@dataclass(frozen=True)
class LeadGenLeadGenReadOnlyReply:
    text: str
    matched: bool


class LeadGenAgentAdapter:
    def __init__(self, config: LeadGenAgentSettings | None = None):
        self.config = config or LeadGenAgentSettings.from_runtime()

    @classmethod
    def from_runtime(cls) -> "LeadGenAgentAdapter":
        return cls(LeadGenAgentSettings.from_runtime())

    def load_exports(self) -> LeadGenAgentExportResult:
        warnings: list[str] = []
        errors: list[str] = []
        repo_path = self.config.repo_path
        source_repo = str(repo_path)

        if not repo_path.exists():
            errors.append(f"Không tìm thấy repo lead-gen-agent tại {source_repo}")
            return LeadGenAgentExportResult(
                ok=False,
                source_repo=source_repo,
                report_run_id=None,
                report_started_at=None,
                report_completed_at=None,
                report_status=None,
                source=None,
                profile_name=None,
                target_country=None,
                lead_query=None,
                lead_count=0,
                leads=[],
                demo_summary_text="",
                report=None,
                warnings=warnings,
                errors=errors,
            )

        leads_path = self.config.hermes_json_path
        demo_path = self.config.demo_summary_path
        report_path = self.config.report_path
        csv_path = self.config.hermes_csv_path

        report: dict[str, Any] | None = None
        report_run_id = None
        report_started_at = None
        report_completed_at = None
        report_status = None
        source = None
        profile_name = None
        target_country = None
        lead_query = None
        lead_entries: list[Mapping[str, Any]] = []

        if leads_path.exists():
            try:
                raw = json.loads(leads_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                errors.append(f"File hermes_leads.json không hợp lệ: {exc.msg}")
                raw = None
            if raw is not None:
                lead_entries = self._extract_lead_entries(raw, warnings)
        else:
            errors.append("Chưa có file lead export. Cần chạy lead-gen-agent trước.")

        if report_path.exists():
            try:
                report_raw = json.loads(report_path.read_text(encoding="utf-8"))
                if isinstance(report_raw, Mapping):
                    report = dict(report_raw)
                    report_run_id = self._first_str(report_raw, "run_id", "report_run_id")
                    report_started_at = self._first_str(report_raw, "started_at", "report_started_at")
                    report_completed_at = self._first_str(report_raw, "completed_at", "report_completed_at")
                    report_status = self._first_str(report_raw, "status", "report_status")
                    source = self._first_str(report_raw, "source")
                    profile_name = self._first_str(report_raw, "profile_name")
                    target_country = self._first_str(report_raw, "target_country")
                    lead_query = self._first_str(report_raw, "lead_query", "query")
                    report.setdefault("source_repo", source_repo)
                else:
                    warnings.append("report.json không phải object JSON hợp lệ.")
            except json.JSONDecodeError as exc:
                warnings.append(f"report.json không hợp lệ: {exc.msg}")
        else:
            warnings.append("Thiếu report.json, nhưng vẫn có thể đọc lead export nếu file hermes_leads.json tồn tại.")

        if csv_path.exists():
            try:
                with csv_path.open(encoding="utf-8", newline="") as handle:
                    reader = csv.reader(handle)
                    csv_rows = sum(1 for _ in reader) - 1
                    if csv_rows < 0:
                        csv_rows = 0
                if report is not None:
                    report.setdefault("hermes_csv_rows", csv_rows)
            except Exception:
                warnings.append("Không thể đọc hermes_leads.csv; bỏ qua file CSV.")

        leads: list[LeadGenLead] = []
        for idx, entry in enumerate(lead_entries):
            lead = self._coerce_lead(entry, idx, leads_path)
            if lead is not None:
                leads.append(lead)

        if not leads and not errors:
            warnings.append("Hiện chưa có lead keep/review trong export.")

        freshness_warning = self._freshness_warning(report_completed_at, report_path)
        if freshness_warning:
            warnings.append(freshness_warning)

        demo_summary_text = ""
        if demo_path.exists():
            try:
                demo_summary_text = demo_path.read_text(encoding="utf-8")
            except Exception:
                warnings.append("Không thể đọc demo_leads.md.")
        else:
            warnings.append("Thiếu demo_leads.md; chỉ dùng JSON export để hiển thị.")

        ok = not errors
        return LeadGenAgentExportResult(
            ok=ok,
            source_repo=source_repo,
            report_run_id=report_run_id,
            report_started_at=report_started_at,
            report_completed_at=report_completed_at,
            report_status=report_status,
            source=source,
            profile_name=profile_name,
            target_country=target_country,
            lead_query=lead_query,
            lead_count=len(leads),
            leads=leads,
            demo_summary_text=demo_summary_text,
            report=report,
            warnings=warnings,
            errors=errors,
        )

    def summarize_for_user(self, language: str = "vi", limit: int = 5) -> str:
        result = self.load_exports()
        if result.errors:
            if any("Chưa có file lead export" in error for error in result.errors):
                return "Chưa có file lead export. Cần chạy lead-gen-agent trước."
            return "\n".join(result.errors)

        usable_leads = [
            lead for lead in result.leads
            if (lead.qualification or lead.relevance_decision or "").lower() in {"keep", "review"}
        ]
        if not usable_leads:
            return "Hiện chưa có lead keep/review trong export."

        lead_lines = []
        selected = usable_leads[: max(limit, 1)]
        for idx, lead in enumerate(selected, start=1):
            lead_lines.extend(self._format_lead_for_vi(lead, idx))
            lead_lines.append("")

        header = [
            f"Đã đọc kết quả từ lead-gen-agent tại {result.source_repo}.",
            f"Hiện có {len(usable_leads)} lead phù hợp để review cho Dukick Studio.",
        ]
        if result.report_run_id:
            header.append(f"Run ID: {result.report_run_id}")
        if result.report_completed_at:
            header.append(f"Hoàn tất lúc: {result.report_completed_at}")
        if result.profile_name:
            header.append(f"ICP profile: {result.profile_name}")
        return "\n".join(header + [""] + lead_lines).strip()

    def respond_to_request(self, user_text: str, language: str = "vi") -> str:
        if looks_like_live_request(user_text):
            return refuse_live_execution_message(language=language)
        return self.summarize_for_user(language=language)

    def _extract_lead_entries(self, raw: Any, warnings: list[str]) -> list[Mapping[str, Any]]:
        if isinstance(raw, list):
            entries = [item for item in raw if isinstance(item, Mapping)]
            if len(entries) != len(raw):
                warnings.append("Bỏ qua một số item không phải object trong hermes_leads.json.")
            return entries
        if isinstance(raw, Mapping):
            for key in ("leads", "items", "results", "data"):
                value = raw.get(key)
                if isinstance(value, list):
                    entries = [item for item in value if isinstance(item, Mapping)]
                    if len(entries) != len(value):
                        warnings.append(f"Bỏ qua item không phải object trong {key}.")
                    return entries
            if any(key in raw for key in ("company_name", "lead_id", "source_id", "website", "page_url")):
                return [raw]
            warnings.append("hermes_leads.json không có mảng leads hợp lệ.")
            return []
        warnings.append("hermes_leads.json không phải JSON object hoặc array hợp lệ.")
        return []

    def _coerce_lead(self, entry: Mapping[str, Any], idx: int, leads_path: Path) -> LeadGenLead | None:
        company_name = self._coerce_text(entry.get("company_name"))
        source_id = self._coerce_text(entry.get("source_id"))
        lead_id = self._coerce_text(entry.get("lead_id"))
        if not lead_id:
            lead_id = source_id or self._stable_lead_id(entry, idx)
        if not company_name:
            company_name = source_id or "Unknown"
        website = self._coerce_text(entry.get("website"))
        domain = self._coerce_text(entry.get("domain"))
        page_url = self._coerce_text(entry.get("page_url"))
        source = self._coerce_text(entry.get("source"))
        source_url = self._coerce_text(entry.get("source_url"))
        ad_title = self._coerce_text(entry.get("ad_title"))
        ad_text_snippet = self._short_text(entry.get("ad_text_snippet") or entry.get("ad_text"))
        cta_text = self._coerce_text(entry.get("cta_text"))
        page_category = self._coerce_text(entry.get("page_category"))
        platforms = tuple(self._coerce_list(entry.get("platforms")))
        is_active = self._coerce_bool(entry.get("is_active"))
        icp_profile = self._coerce_text(entry.get("icp_profile"))
        icp_fit_score = self._coerce_float(entry.get("icp_fit_score"))
        creative_opportunity_score = self._coerce_float(entry.get("creative_opportunity_score"))
        creative_opportunity_reason = self._coerce_text(entry.get("creative_opportunity_reason"))
        market_fit_score = self._coerce_float(entry.get("market_fit_score"))
        market_fit_decision = self._coerce_text(entry.get("market_fit_decision"))
        market_fit_reason = self._coerce_text(entry.get("market_fit_reason"))
        relevance_score = self._coerce_float(entry.get("relevance_score"))
        relevance_decision = self._coerce_text(entry.get("relevance_decision"))
        qualification = self._coerce_text(entry.get("qualification"))
        evidence_summary = self._short_text(entry.get("evidence_summary"))
        suggested_video_angle = self._coerce_text(entry.get("suggested_video_angle"))
        why_dukick_should_care = self._coerce_text(entry.get("why_dukick_should_care"))
        raw_artifact_reference = self._coerce_text(entry.get("raw_artifact_reference")) or str(leads_path)

        return LeadGenLead(
            lead_id=lead_id,
            company_name=company_name,
            website=website,
            domain=domain,
            page_url=page_url,
            source=source,
            source_id=source_id,
            source_url=source_url,
            ad_title=ad_title,
            ad_text_snippet=ad_text_snippet,
            cta_text=cta_text,
            page_category=page_category,
            platforms=platforms,
            is_active=is_active,
            icp_profile=icp_profile,
            icp_fit_score=icp_fit_score,
            creative_opportunity_score=creative_opportunity_score,
            creative_opportunity_reason=creative_opportunity_reason,
            market_fit_score=market_fit_score,
            market_fit_decision=market_fit_decision,
            market_fit_reason=market_fit_reason,
            relevance_score=relevance_score,
            relevance_decision=relevance_decision,
            qualification=qualification,
            evidence_summary=evidence_summary,
            suggested_video_angle=suggested_video_angle,
            why_dukick_should_care=why_dukick_should_care,
            raw_artifact_reference=raw_artifact_reference,
        )

    def _format_lead_for_vi(self, lead: LeadGenLead, idx: int) -> list[str]:
        lines = [f"{idx}. {lead.company_name}"]
        if lead.website:
            lines.append(f"* Website: {lead.website}")
        if lead.domain:
            lines.append(f"* Domain: {lead.domain}")
        if lead.page_category:
            lines.append(f"* Ngành: {lead.page_category}")
        if lead.cta_text or lead.platforms:
            ad_parts = []
            if lead.cta_text:
                ad_parts.append(f"CTA {lead.cta_text}")
            if lead.platforms:
                ad_parts.append(f"đang chạy trên {', '.join(lead.platforms)}")
            lines.append(f"* Tín hiệu ads: {'; '.join(ad_parts)}")
        if lead.creative_opportunity_reason:
            lines.append(f"* Cơ hội TVC: {lead.creative_opportunity_reason}")
        if lead.market_fit_decision:
            market = lead.market_fit_decision
            if lead.market_fit_reason:
                market = f"{market} — {lead.market_fit_reason}"
            lines.append(f"* Market fit: {market}")
        if lead.qualification:
            lines.append(f"* Trạng thái: {lead.qualification}")
        if lead.suggested_video_angle:
            lines.append(f"* Gợi ý cho Dukick: {lead.suggested_video_angle}")
        if lead.why_dukick_should_care:
            lines.append(f"* Vì sao Dukick nên quan tâm: {lead.why_dukick_should_care}")
        if lead.evidence_summary:
            lines.append(f"* Bằng chứng: {lead.evidence_summary}")
        return lines

    def _coerce_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (int, float, bool)):
            return str(value)
        return str(value).strip()

    def _short_text(self, value: Any, max_len: int = 220) -> str:
        text = self._coerce_text(value)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > max_len:
            return text[: max_len - 1].rstrip() + "…"
        return text

    def _coerce_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [part.strip() for part in re.split(r"[;,]", value) if part.strip()]
        if isinstance(value, Iterable):
            result: list[str] = []
            for item in value:
                text = self._coerce_text(item)
                if text:
                    result.append(text)
            return result
        return []

    def _coerce_bool(self, value: Any) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y", "on"}:
                return True
            if lowered in {"false", "0", "no", "n", "off"}:
                return False
        return None

    def _coerce_float(self, value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _stable_lead_id(self, entry: Mapping[str, Any], idx: int) -> str:
        serializable = {k: v for k, v in entry.items() if not _SECRETISH_RE.search(str(k))}
        payload = json.dumps(serializable, sort_keys=True, ensure_ascii=False, default=str)
        digest = sha256(payload.encode("utf-8")).hexdigest()[:12]
        return f"lead-{idx + 1}-{digest}"

    def _first_str(self, mapping: Mapping[str, Any], *keys: str) -> str | None:
        for key in keys:
            value = mapping.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if value not in (None, ""):
                return str(value)
        return None

    def _freshness_warning(self, completed_at: str | None, report_path: Path) -> str | None:
        when = _parse_datetime(completed_at) if completed_at else None
        if when is None:
            try:
                mtime = datetime.fromtimestamp(report_path.stat().st_mtime, tz=UTC)
                when = mtime
            except Exception:
                return None
        if datetime.now(tz=UTC) - when > STALE_REPORT_AFTER:
            return "Cảnh báo: report_export có thể đã cũ; hãy chạy lead-gen-agent lại trước khi demo."
        return None


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _replace_secretish(text: str) -> str:
    if not text:
        return text
    return re.sub(r"(?i)(api[_-]?key|token|secret|bearer)[^\s]*", "[REDACTED]", text)


def looks_like_live_request(text: str) -> bool:
    return bool(_LIVE_REQUEST_RE.search(text or ""))


def looks_like_lead_discovery_request(text: str) -> bool:
    text = text or ""
    if _LEAD_SUBJECT_RE.search(text) and (_LEAD_ACTION_RE.search(text) or _DUKICK_RE.search(text)):
        return True
    return False


def looks_like_explicit_public_web_search_request(text: str) -> bool:
    return bool(_EXPLICIT_PUBLIC_WEB_SEARCH_RE.search(text or ""))


def route_sales_lead_request(
    user_text: str,
    *,
    language: str = "vi",
    profile_name: str = "",
    config: LeadGenAgentSettings | None = None,
) -> str | None:
    from hermes_cli.leadgen_agent_orchestrator import route_sales_lead_request as _route

    return _route(
        user_text,
        language=language,
        profile_name=profile_name,
        config=config,
    )


def refuse_live_execution_message(language: str = "vi") -> str:
    if language.lower().startswith("vi"):
        return (
            "Em chưa tự chạy live trong chế độ hiện tại. Live Apify có thể phát sinh phí nên cần xác nhận thủ công. "
            "Hiện em có thể đọc kết quả lead-gen-agent đã export."
        )
    return (
        "Live execution is disabled in the current mode. Apify can incur cost, so manual confirmation is required. "
        "I can read the lead-gen-agent exports that already exist."
    )


def load_leadgen_agent_exports(config: LeadGenAgentSettings | None = None) -> LeadGenAgentExportResult:
    return LeadGenAgentAdapter(config).load_exports()


def summarize_leadgen_agent_exports(language: str = "vi", limit: int = 5, config: LeadGenAgentSettings | None = None) -> str:
    return LeadGenAgentAdapter(config).summarize_for_user(language=language, limit=limit)


def respond_to_leadgen_request(user_text: str, language: str = "vi", config: LeadGenAgentSettings | None = None) -> str:
    return LeadGenAgentAdapter(config).respond_to_request(user_text, language=language)
