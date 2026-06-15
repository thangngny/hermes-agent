# LeadGen Agent Integration

Hermes uses `lead-gen-agent` as a read-only backend for Dukick Studio lead discovery.

## Repo Path

Default repo path:

```text
~/CongThang/lead-gen-agent
```

You can override it with `LEADGEN_AGENT_REPO` or the `leadgen_agent.repo` config key.

## Read-Only Contract

Hermes only reads exported files from the lead-gen repo. In the default mode it does **not**:

* run subprocesses
* call Apify
* call OpenAI
* send email
* connect Gmail or SMTP
* write runtime artifacts into the lead-gen repo

## Files Read

Hermes reads these files when they exist:

* `data/output/hermes_leads.json`
* `data/output/hermes_leads.csv`
* `data/output/demo_leads.md`
* `data/output/report.json`

The JSON export is the authoritative lead list. The CSV and markdown files are optional supporting outputs.

## User-Facing Output

Hermes formats the export into concise Vietnamese for Dukick Studio:

* Tên lead
* Website
* Ngành
* Tín hiệu quảng cáo hiện tại
* Cơ hội TVC/video ads
* Market fit
* Trạng thái: keep / review / reject
* Gợi ý angle cho Dukick
* Vì sao Dukick nên quan tâm

Raw payloads, API tokens, and long ad text blobs are not shown.

## Safe Manual Workflow

1. Run `lead-gen-agent` in its own repo.
2. Inspect the export files in `~/CongThang/lead-gen-agent/data/output/`.
3. In Hermes, run:

```bash
hermes leadgen show
```

or read the export through the adapter directly from code.

## Sales Profile Routing

Inside the `sales` profile, lead-discovery requests now prefer the read-only `lead-gen-agent` export path before any public web search flow.

Examples:

* `tìm lead`
* `khảo sát thêm lead trên mạng`
* `đọc lead mới nhất`
* `lead mới cho Dukick`
* `xem danh sách lead`

Behavior:

* Hermes reads the latest exported `hermes_leads.json` / `demo_leads.md` / `report.json` first.
* Hermes returns the Vietnamese Dukick summary from that export.
* Hermes does **not** auto-run Apify, Gmail, email, or any live lead refresh.
* If the user explicitly asks for live search, Hermes refuses automatic execution and says manual confirmation is required.
* Hermes only falls back to public web search when the export is missing or empty **and** the user explicitly asks for a public web search instead of the export flow.

## Current Safety Defaults

These settings stay disabled unless you explicitly override them in your local environment:

* `LEADGEN_AGENT_MODE=read_only`
* `LEADGEN_AGENT_ALLOW_DRY_RUN=false`
* `LEADGEN_AGENT_ALLOW_LIVE=false`

## Missing or Empty Export Behavior

* Missing `hermes_leads.json`: Hermes says the export file is missing and asks you to run `lead-gen-agent` first.
* Empty export: Hermes says there are no keep/review leads yet.
* Missing `report.json`: Hermes can still read the leads, but shows a metadata warning.
* Invalid JSON: Hermes returns a safe error instead of crashing.
* Missing export during `sales` lead-search routing: Hermes returns `Chưa có lead export từ lead-gen-agent. Cần chạy lead-gen-agent trước hoặc xác nhận nếu muốn public web search.`

## Future Human-Confirmed Flow

Future manual live execution should remain a human-gated step. Hermes must not do it automatically.

Example future manual command in the lead-gen repo:

```bash
cd ~/CongThang/lead-gen-agent && leadgen-agent run --source apify --dry-run
```

If live execution is ever enabled later, it must stay behind explicit approval and local secrets. That path is intentionally disabled in this Hermes integration sprint.

## Safety Rules

* Never commit `.env`
* Never commit runtime artifacts
* Never print secrets
* Never auto-run Apify or OpenAI from Hermes
* Never send email from Hermes in this integration
