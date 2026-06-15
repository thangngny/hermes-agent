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

## Phase 1: Read-Only

Phase 1 is the stable baseline:

* Hermes reads existing exports from `lead-gen-agent`
* `hermes leadgen show` stays read-only
* `sales` profile requests like `đọc lead mới nhất` return the latest exported summary

## Phase 2: Confirmed Execution

Phase 2 adds controlled Apify orchestration for the `sales` profile:

* Hermes parses natural requests like `Tìm lead spa ở Việt Nam cho Dukick`
* Hermes runs `leadgen-agent smoke-plan --source apify` and `leadgen-agent preflight --source apify`
* Hermes returns the preflight summary in Vietnamese, including query, country, max results, and estimated cost
* Hermes does **not** run live Apify until the user explicitly confirms
* Hermes still does **not** send email, connect Gmail, or enable OpenAI live in this flow

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

Inside the `sales` profile, lead-discovery requests now use `lead-gen-agent` before any public web search flow.

Examples:

* `tìm lead`
* `khảo sát thêm lead trên mạng`
* `đọc lead mới nhất`
* `lead mới cho Dukick`
* `xem danh sách lead`
* `tìm lead spa ở Việt Nam`
* `tìm 3 lead gym`

Behavior:

* `đọc lead mới nhất` / `xem danh sách lead` reads the latest exported `hermes_leads.json` / `demo_leads.md` / `report.json`.
* Search requests like `tìm lead spa` run `smoke-plan` and `preflight` first, then ask for confirmation.
* Hermes returns the Vietnamese Dukick summary from the export after a successful run.
* Hermes does **not** auto-run Apify, Gmail, email, or OpenAI live.
* If the user explicitly asks for live search, Hermes still requires confirmation.
* Hermes only falls back to public web search when the export is missing or empty **and** the user explicitly asks for a public web search instead of the leadgen flow.

Example:

1. User: `Tìm lead spa ở Việt Nam`
2. Hermes: preflight only, for example:

   `Em đã chuẩn bị tìm lead bằng Apify ... Anh xác nhận 'chạy live 1 lead' thì em mới chạy.`

3. User: `xác nhận chạy live 1 lead`
4. Hermes: runs the gated Apify command, regenerates `demo_leads.md`, then summarizes the exported leads in Vietnamese.

## Current Safety Defaults

These settings stay disabled unless you explicitly override them in your local environment:

* `LEADGEN_AGENT_MODE=confirm_live`
* `LEADGEN_AGENT_ALLOW_DRY_RUN=true`
* `LEADGEN_AGENT_ALLOW_LIVE=false`
* `LEADGEN_AGENT_MAX_LIVE_RESULTS=3`
* `LEADGEN_AGENT_DEFAULT_COUNTRY=VN`
* `LEADGEN_AGENT_DEFAULT_PROFILE=dukick_tvc`
* `LEADGEN_AGENT_SPEND_LIMIT_USD=1`

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
* Never run live Apify without explicit user confirmation
* Never exceed the configured live-result cap without the user lowering scope first
* Never send email from Hermes in this integration
