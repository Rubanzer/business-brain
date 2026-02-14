DOtn# Business Brain — Product Requirements Document

**Version:** 3.0
**Date:** 2026-02-13
**Author:** Krishna Jindal + Claude
**Status:** Draft for approval

---

## 1. Executive Summary

Business Brain is a proactive intelligence platform for manufacturing companies. It ingests data from every system in a factory — ERP, SCADA, logistics software, Google Sheets, manual uploads — unifies it into a single source of truth, automatically discovers patterns and anomalies across departments, and delivers actionable alerts before problems escalate.

**The core promise:** No more blame games. No more firefighting. The system knows something is wrong before you do.

**First customer:** Krishna's own steel manufacturing company (~15 systems, ~50 data sheets, multiple departments).

**Business model:** Per-plant SaaS pricing.

---

## 2. Vision

Traditional manufacturing runs on tribal knowledge, scattered spreadsheets, and reactive problem-solving. When a quality rejection happens, production blames raw material. Procurement blames the supplier. Nobody has a unified view.

Business Brain eliminates this by:

1. **Collecting everything** — automatic sync from every source (Tally, VEGA, SCADA, Google Sheets, maintenance apps, manual uploads)
2. **Understanding it** — semantic column classification, cross-source reconciliation, same-data-different-format detection
3. **Finding patterns** — anomaly detection, breakdown prediction from SCADA signatures, cross-department correlations
4. **Acting on it** — natural language alert deployment, Telegram notifications, persistent auto-refreshing reports
5. **Explaining it** — root cause analysis powered by company context (processes, thresholds, org structure)

**Future vision:** Domain-specific AI agents trained on Six Sigma, 5S, TPM, and Lean Manufacturing frameworks that provide world-class operational recommendations.

---

## 3. Target Users & Personas

### Primary: Plant Operations Team (Non-technical)

- **Plant Manager** — wants a morning dashboard: production vs target, breakdowns overnight, power consumption
- **Production Head** — wants shift-wise output, machine utilization, time-wise production
- **Quality Head** — wants rejection rates, grade-wise analysis, supplier quality scores
- **Procurement Head** — wants supplier delivery tracking, rate comparison, material consumption
- **Maintenance Head** — wants breakdown frequency, MTBF/MTTR, predictive alerts from SCADA
- **Finance/Accounts** — wants cost reports from Tally, energy costs, variance analysis
- **Marketing Head** — wants officer movement tracking, spend vs outcomes

### Key trait: None of these people write SQL or code. Everything must be natural language or click-based.

### Secondary: Company Management

- **CEO/Owner** — cross-department unified view, daily KPI digest
- **IT Admin** — manages data source connections, user setup

---

## 4. What's Already Built

### v1: Reactive Q&A Engine (Complete)

| Feature | Description |
|---------|-------------|
| **Column semantic classifier** | Pure Python — detects identifiers, categoricals, currencies, percentages, temporals, booleans. Detects business domain (sales, finance, HR, procurement, marketing, inventory) |
| **Multi-agent analysis pipeline** | Supervisor → SQL Agent → Business Analyst → Python Analyst → CFO Agent. LangGraph orchestration |
| **Insight-first frontend** | Key metrics (hero cards) → Charts → CFO verdict → Findings → Computations → Collapsible details |
| **Drill-down** | Click any finding/metric → full pipeline re-runs focused on that insight |
| **Smart data upload** | DataEngineerAgent: parse, clean, deduplicate, load, auto-generate metadata |
| **Business context** | Free-text or file upload (.txt, .md, .pdf) → chunked, embedded, used for RAG |
| **Session-based chat** | Conversation memory within analysis sessions |
| **Inline data editing** | Double-click cells in data viewer to edit values |
| **Indian locale** | Rs formatting, en-IN number locale throughout |

### v2: Proactive Discovery Engine (Complete)

| Feature | Description |
|---------|-------------|
| **Table profiler** | Auto-classifies all columns across all tables, caches profiles, detects data changes via hashing |
| **Relationship finder** | Cross-table join detection: name matching, value overlap, semantic matching |
| **Anomaly detector** | Null spikes, numeric outliers (2σ), impossible values (negative currency, >100%), constant columns, rare categories, time trend availability |
| **Composite metric discoverer** | 5 templates: Buyer Credit Score, Employee Performance Index, Supplier Risk Score, Product Profitability, Customer Churn Risk |
| **Cross-event correlator** | Finds event↔metric correlations across tables (e.g., "absence correlates with productivity change") |
| **Narrative builder** | Gemini LLM connects related insights into cause-and-effect stories |
| **Feed tab** | Default landing page showing ranked insight cards, color-coded by severity |
| **Deployed reports** | Pin any insight as a persistent report that auto-refreshes when data changes |
| **Smart suggestions** | Auto-generated question pills based on profiled column types |
| **Export** | CSV download + PDF export (html2canvas + jsPDF) |
| **Background discovery** | Auto-triggers after file upload via FastAPI BackgroundTasks |

### Tech Stack

- **Backend:** FastAPI (async), SQLAlchemy 2.0 (async), PostgreSQL + pgvector
- **LLM:** Google Gemini 2.0 Flash (analysis + narratives), Gemini embeddings
- **Agents:** LangGraph multi-agent orchestration
- **Frontend:** Vanilla HTML/CSS/JS, Chart.js
- **Deployment:** Vercel (serverless), Docker Compose (local PostgreSQL)

---

## 5. What We're Building Now (v3)

### The Three Pillars of Manufacturing

Everything revolves around **Material, Energy, People** — the three inputs every factory optimizes.

### v3 Feature Overview

| # | Feature | Impact |
|---|---------|--------|
| 1 | **Google Sheets real-time sync + recurring auto-ingestion** | Gets 50 sheets flowing automatically |
| 2 | **Sanctity engine** — change tracking, impossible values, cross-source conflicts | Trust = adoption. No trust = no product |
| 3 | **Natural language alert deployment + Telegram bot** | The "40 trucks" moment. Killer feature |
| 4 | **Pattern memory** — learns what preceded past events, recognizes recurrence | SCADA breakdown prediction. Sells the product |
| 5 | **Structured onboarding** — guided company context collection | Powers all RCA and recommendations |
| 6 | **Same-data-different-format detection** — auto-reconciles duplicate data sources | Eliminates conflicts, enables cross-source truth |

---

## 6. Feature Specifications

### 6.1 Google Sheets Real-Time Sync + Recurring Auto-Ingestion

#### Problem
50 sheets across the company. Some are SCADA exports (same format every hour). Some are manually maintained logs. Currently, someone has to download and upload each one.

#### Solution

**A. Google Sheets Connector**

- User pastes a Google Sheets URL (or sheet ID)
- System authenticates via a service account (Google Sheets API v4)
- System reads all tabs, loads data into PostgreSQL
- **Polling sync:** Every N minutes (configurable, default 5), system checks for changes using the sheet's `modifiedTime` from Drive API
- When changes detected → re-fetch → diff against existing data → apply updates
- Show sync status: "Last synced 2 min ago · 3 rows changed"

**B. Recurring Sheet Ingestion**

- User designates a "recurring source" when connecting a sheet or uploading a file
- System remembers the format (column names, types, table name)
- When a new file with the same structure is uploaded (or a sheet tab is appended), system auto-appends to the existing table instead of creating a new one
- Format fingerprinting: hash of column names + types → match against known recurring formats

**C. Multi-Source Management UI**

- New "Sources" tab showing all connected data sources
- Each source shows: type (sheet/upload/API), table name, last sync, row count, sync frequency, status (active/paused/error)
- Add Source button → choose type → configure
- Pause/resume/delete sources

#### Data Model

```
DataSource:
  id: UUID
  name: str                     # "SCADA Power Log", "Gate Register"
  source_type: str              # google_sheet / api / recurring_upload / manual_upload
  connection_config: JSON       # {sheet_id, tab_name, api_url, headers, ...}
  table_name: str               # target PostgreSQL table
  format_fingerprint: str       # hash of column structure for recurring detection
  sync_frequency_minutes: int   # 0 = manual only
  last_sync_at: datetime
  last_sync_status: str         # success / error
  last_sync_error: str | null
  rows_total: int
  created_at: datetime
  active: bool

DataChangeLog:
  id: int
  data_source_id: UUID
  change_type: str              # row_added / row_modified / row_deleted
  table_name: str
  row_identifier: str           # PK or row hash
  column_name: str | null       # which column changed (for modifications)
  old_value: str | null
  new_value: str | null
  detected_at: datetime
  acknowledged: bool
```

#### API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/sources` | List all data sources |
| POST | `/sources/google-sheet` | Connect a Google Sheet |
| POST | `/sources/api` | Connect an API endpoint |
| POST | `/sources/{id}/sync` | Manually trigger sync |
| PUT | `/sources/{id}` | Update source config (frequency, name) |
| DELETE | `/sources/{id}` | Disconnect a source |
| GET | `/sources/{id}/changes` | Get recent change log |
| POST | `/sources/{id}/pause` | Pause auto-sync |
| POST | `/sources/{id}/resume` | Resume auto-sync |

#### Google Sheets Auth Flow

- Use a **service account** (no OAuth flow needed for users)
- User shares their Google Sheet with the service account email
- System reads using gspread or Google API client
- Store service account credentials in environment variable (`GOOGLE_SERVICE_ACCOUNT_JSON`)

---

### 6.2 Sanctity Engine

#### Problem
When data comes from 15 systems and 50 sheets, trust is everything. If someone silently changes a number in a Google Sheet, the plant manager needs to know. If SCADA reports power consumption of -500 kWh, the system should flag it. If the gate register and VEGA show different truck counts, someone is wrong.

#### Solution

Three layers of data integrity:

**Layer 1: Change Tracking**

- Every sync from Google Sheets (or any source) diffs against the previous version
- Changes logged to `DataChangeLog` with old_value → new_value
- UI shows a "Changes" feed: "Yesterday at 14:32, cell B7 in 'Production Log' changed from 28.5 to 32.1 (by sync from Google Sheet)"
- Critical changes (values changing by >20% or key metric columns) trigger alerts

**Layer 2: Impossible Value Detection**

- Runs automatically after every data ingestion (sync or upload)
- Uses column classification from the profiler:
  - `numeric_currency` < 0 → flag (unless column allows negatives per context)
  - `numeric_percentage` outside 0-100 → flag
  - Values > 4σ from historical mean → flag as statistical outlier
  - Nulls in columns that historically have 0% nulls → flag
  - String in a numeric column → flag
  - Date in the future (for historical logs) → flag
- Manufacturing-specific rules (configurable via company context):
  - Power consumption: 0 ≤ value ≤ plant_max_kva
  - Temperature: ambient ≤ value ≤ furnace_max
  - Weight/tonnage: 0 ≤ value ≤ truck_max_capacity
- Each flag generates a `SanctityIssue` record

**Layer 3: Cross-Source Conflict Detection**

- When two data sources report the same metric (detected via same-data-different-format detection in 6.6):
  - Compare values for the same entity + time period
  - If values differ by more than a threshold → generate conflict insight
  - Show side-by-side: "Gate register shows 38 trucks at 10:00, VEGA shows 42 trucks at 10:00"
  - Let user mark which source is authoritative

#### Data Model

```
SanctityIssue:
  id: int
  table_name: str
  column_name: str
  row_identifier: str
  issue_type: str          # impossible_value / statistical_outlier / null_spike /
                           # cross_source_conflict / unauthorized_change / future_date
  severity: str            # critical / warning / info
  description: str
  current_value: str
  expected_range: str | null    # "0-100" or "0-500 kWh"
  conflicting_source: str | null
  conflicting_value: str | null
  detected_at: datetime
  resolved: bool
  resolved_by: str | null       # "user:krishna" or "auto:re-sync"
  resolution_note: str | null
```

#### API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/sanctity` | List all open sanctity issues |
| GET | `/sanctity/summary` | Issue counts by type and severity |
| POST | `/sanctity/{id}/resolve` | Mark issue as resolved with note |
| GET | `/changes` | Recent data changes across all sources |

#### UI

- New "Data Health" indicator on every tab: green checkmark (no issues), yellow warning (minor issues), red alert (critical issues)
- Sanctity issues appear in the Feed as high-priority insight cards
- Click to see details + resolve

---

### 6.3 Natural Language Alert Deployment + Telegram Bot

#### Problem
Plant managers think in conditions: "Tell me when trucks exceed 40." "Alert me if power consumption drops below 200 kWh for more than 30 minutes." They don't want to configure monitoring rules in a form — they want to say it in plain language.

#### Solution

**A. Natural Language → Alert Rule**

User types (in a chat-like interface or the existing question input):

> "Alert me on Telegram when gate truck count exceeds 40"

System uses LLM to parse this into a structured alert rule:

```json
{
  "table": "gate_register",
  "column": "truck_count",
  "condition": "greater_than",
  "threshold": 40,
  "check_trigger": "on_data_change",
  "notification_channel": "telegram",
  "message_template": "Gate alert: {{truck_count}} trucks at gate (threshold: 40)"
}
```

System responds:
> "I'll monitor the gate_register table. When truck_count exceeds 40 (checked every time new data arrives), I'll send you a Telegram alert. Want me to deploy this?"

User confirms → alert is live.

**B. Alert Rule Types**

| Type | Example | Detection |
|------|---------|-----------|
| **Threshold** | "Alert when X > N" | Simple value comparison |
| **Trend** | "Alert when production drops 3 days in a row" | Rolling window comparison |
| **Absence** | "Alert if no data arrives from SCADA for 30 minutes" | Missing data detection |
| **Pattern** | "Alert when SCADA readings look like they did before the last breakdown" | Pattern memory match (6.4) |
| **Cross-source** | "Alert when gate count and VEGA count differ by more than 5" | Cross-source comparison |
| **Composite** | "Alert when power-per-ton exceeds 500 kWh" | Computed metric threshold |

**C. Alert Lifecycle**

```
User describes condition (natural language)
  → LLM parses into structured rule
  → System shows parsed rule for confirmation
  → User confirms
  → Rule deployed as AlertRule
  → On every data change (sync/upload), system evaluates all active rules
  → When triggered:
    → Create AlertEvent record
    → Send Telegram message
    → Show in Feed as a triggered alert
  → User can: pause, modify threshold, view history, delete
```

**D. Telegram Bot**

- One-way: bot sends alerts to users
- Users register by chatting with the bot → bot gives them a registration code → they enter it in the web app → linked
- Bot message format:
  ```
  🚨 ALERT: Gate Truck Count

  Current: 43 trucks (threshold: 40)
  Time: 2026-02-13 14:32 IST
  Source: gate_register (VEGA sync)

  View details: [link to web app]
  ```
- Group support: alerts can be sent to a Telegram group (e.g., "Logistics Team" group)

**E. Natural Language Management**

Users can also say:
- "Pause the truck alert for this weekend" → system pauses rule, auto-resumes Monday
- "Change the truck threshold from 40 to 45" → system updates rule
- "Show me the last 10 times the truck alert fired" → system shows history
- "How often does the power drop alert trigger?" → system shows frequency analysis

#### Data Model

```
AlertRule:
  id: UUID
  name: str                     # "Gate Truck Count Alert"
  description: str              # Original natural language input
  rule_config: JSON             # {table, column, condition, threshold, ...}
  rule_type: str                # threshold / trend / absence / pattern / cross_source / composite
  check_trigger: str            # on_data_change / scheduled
  schedule_cron: str | null     # for scheduled checks
  notification_channel: str     # telegram / feed
  notification_config: JSON     # {chat_id, group_id, ...}
  message_template: str
  active: bool
  paused_until: datetime | null
  created_at: datetime
  last_triggered_at: datetime | null
  trigger_count: int
  session_id: str | null

AlertEvent:
  id: int
  alert_rule_id: UUID
  triggered_at: datetime
  trigger_value: str            # "43"
  threshold_value: str          # "40"
  context: JSON                 # snapshot of relevant data at trigger time
  notification_sent: bool
  notification_error: str | null
```

#### API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/alerts/deploy` | Deploy alert from natural language |
| GET | `/alerts` | List all alert rules |
| GET | `/alerts/{id}` | Get alert rule details + history |
| PUT | `/alerts/{id}` | Update alert rule |
| POST | `/alerts/{id}/pause` | Pause alert (optional: until datetime) |
| POST | `/alerts/{id}/resume` | Resume alert |
| DELETE | `/alerts/{id}` | Delete alert rule |
| GET | `/alerts/{id}/events` | Get trigger history |
| POST | `/alerts/evaluate` | Manually trigger evaluation of all rules |
| POST | `/telegram/register` | Register Telegram chat for alerts |
| GET | `/telegram/status` | Check bot connection status |

---

### 6.4 Pattern Memory

#### Problem
A SCADA system shows electrical readings every few seconds. Before the last 3 furnace breakdowns, there was a specific signature: KVA dropped while unit consumption stayed high (wasted energy). The maintenance team knows this pattern from experience, but the system doesn't.

#### Solution

**A. Pattern Learning**

Two modes:

1. **User-labeled patterns:** User tells the system "this period in the data was a breakdown" → system captures the data signature (all columns, time window) as a named pattern
2. **Auto-detected patterns:** When the anomaly detector or cross-event correlator finds something significant, and the user confirms it's meaningful ("yes, this was a real breakdown"), system stores the pattern

**B. Pattern Structure**

A pattern is:
- A set of conditions across one or more columns
- Over a time window (e.g., "15 minutes", "3 data points")
- With a similarity threshold (how close does new data need to be to trigger)

Example pattern "Pre-Breakdown SCADA Signature":
```json
{
  "name": "Pre-Breakdown SCADA Signature",
  "source_table": "scada_readings",
  "conditions": [
    {"column": "kva", "behavior": "decreasing", "magnitude": ">10%"},
    {"column": "unit_consumption", "behavior": "stable_or_increasing"},
    {"column": "power_factor", "behavior": "decreasing", "magnitude": ">5%"}
  ],
  "time_window": "15_minutes",
  "historical_occurrences": [
    {"start": "2026-01-15T10:30:00", "end": "2026-01-15T10:45:00", "outcome": "furnace_2_breakdown"},
    {"start": "2026-02-01T14:00:00", "end": "2026-02-01T14:15:00", "outcome": "furnace_1_breakdown"}
  ],
  "confidence": 0.85
}
```

**C. Pattern Matching**

- Every time new data lands for a table that has patterns registered:
  1. Get the latest N data points (based on pattern's time_window)
  2. Compare against each registered pattern using similarity scoring
  3. If similarity > threshold → trigger alert: "SCADA readings match the pattern seen before the last 3 breakdowns (85% match)"
- Pattern matching uses:
  - Trend direction matching (increasing/decreasing/stable)
  - Magnitude comparison (within ±tolerance)
  - Multi-column correlation (all conditions must match simultaneously)

**D. Pattern Improvement**

- When a pattern triggers and the predicted event happens → increase confidence
- When a pattern triggers but nothing happens → decrease confidence, ask user if this is a false positive
- When an event happens without the pattern triggering → ask user to help identify what the precursor pattern was → learn new pattern

#### Data Model

```
Pattern:
  id: UUID
  name: str
  description: str
  source_tables: JSON           # ["scada_readings"]
  conditions: JSON              # [{column, behavior, magnitude, ...}]
  time_window_minutes: int
  similarity_threshold: float   # 0.0-1.0
  historical_occurrences: JSON  # [{start, end, outcome}]
  confidence: float
  created_by: str               # "user" / "auto_detected"
  created_at: datetime
  last_matched_at: datetime | null
  match_count: int
  false_positive_count: int
  active: bool

PatternMatch:
  id: int
  pattern_id: UUID
  matched_at: datetime
  similarity_score: float
  data_snapshot: JSON           # the actual values that matched
  outcome: str | null           # filled in later: "confirmed_breakdown" / "false_positive" / null
  alert_sent: bool
```

---

### 6.5 Structured Onboarding

#### Problem
The business context file is critical for RCA and recommendations, but right now it's free-text. The system needs structured knowledge about the company to give useful recommendations: What does the company make? What are acceptable ranges for key metrics? What's the process flow? What are the departments?

#### Solution

**A. Guided Onboarding Flow**

When a new company starts (or when user clicks "Setup Company Profile"), a step-by-step wizard:

```
Step 1: Company Basics
  - Company name
  - Industry (Steel / Cement / Textile / Pharma / Auto / Food / Other)
  - Products made (free text: "TMT bars, billets, wire rods")
  - Number of plants

Step 2: Departments & Functions
  - Select active departments: [Production, Quality, Procurement, Logistics,
    Maintenance, HR, Finance, Marketing, Safety]
  - For each: who heads it? (name, optional contact)

Step 3: Key Metrics & Thresholds
  - System shows detected metrics from profiled data
  - User sets acceptable ranges:
    - Power consumption: normal range 300-450 kWh/ton
    - Furnace temperature: normal range 1500-1650°C
    - Production target: 500 tons/day
    - Quality rejection: acceptable < 2%
  - These become the baseline for anomaly detection and alerts

Step 4: Process Flow (optional)
  - "Describe your production process in a few lines"
  - Or structured: Input materials → Process stages → Output products
  - Example: "Scrap/sponge iron → Induction furnace → Continuous casting → Rolling mill → TMT bars"

Step 5: Known Relationships
  - "Which departments' data is connected?"
  - "When procurement quality is low, which production metrics are affected?"
  - This seeds the cross-event correlator

Step 6: Data Sources
  - List existing systems: Tally, VEGA, SCADA, etc.
  - For each: what data does it contain? Which department uses it?
```

**B. Context Storage**

All onboarding data is stored as structured JSON AND converted to natural language context (for the existing RAG pipeline). This way:
- The structured data powers thresholds, anomaly rules, and alert defaults
- The natural language version powers LLM-based analysis and RCA

**C. Progressive Enhancement**

- The system can work with zero onboarding (pure data discovery)
- But every piece of context makes it smarter
- After the system discovers something, it can ask: "I noticed power consumption ranges from 280-520 kWh/ton. What's the acceptable range for your plant?"
- These micro-interactions progressively build the company profile

#### Data Model

```
CompanyProfile:
  id: UUID
  name: str
  industry: str
  products: JSON                # ["TMT bars", "billets"]
  departments: JSON             # [{name, head, contact}]
  process_flow: str | null
  created_at: datetime
  updated_at: datetime

MetricThreshold:
  id: int
  company_id: UUID
  metric_name: str              # "power_consumption_per_ton"
  table_name: str | null        # which table this metric comes from
  column_name: str | null
  unit: str                     # "kWh/ton"
  normal_min: float
  normal_max: float
  warning_min: float | null
  warning_max: float | null
  critical_min: float | null
  critical_max: float | null
  created_at: datetime
```

#### API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/company` | Get company profile |
| PUT | `/company` | Update company profile |
| POST | `/company/onboard` | Submit full onboarding |
| GET | `/thresholds` | List all metric thresholds |
| POST | `/thresholds` | Add a threshold |
| PUT | `/thresholds/{id}` | Update a threshold |
| DELETE | `/thresholds/{id}` | Remove a threshold |

#### UI

- New "Setup" tab (or first-run wizard that appears before Feed)
- Progress indicator: "Company profile: 60% complete"
- "Add more context" prompts throughout the app when the system could benefit from more information

---

### 6.6 Same-Data-Different-Format Detection

#### Problem
In a steel company, production data might exist in:
- A SCADA-generated CSV with columns: `TIMESTAMP, HEAT_NO, GRADE, WT_TONS`
- A manually maintained Google Sheet with columns: `Date, Heat Number, Steel Grade, Weight (MT)`
- An ERP export with columns: `production_date, heat_id, material_grade, output_tonnage`

All three describe the same thing. The system needs to recognize this.

#### Solution

**A. Semantic Column Matching**

When a new data source is connected:
1. Profile its columns using the existing column classifier
2. Compare the semantic profile against all existing table profiles
3. Scoring:
   - Column name similarity (fuzzy match): "HEAT_NO" ↔ "Heat Number" ↔ "heat_id"
   - Semantic type match: all three are identifiers with similar cardinality
   - Value overlap: if the actual values match (same heat numbers appear in both)
   - Domain match: both classified as "production" domain
4. If similarity score > threshold → flag as potential duplicate source

**B. Reconciliation**

When duplicates are detected:
1. Show user: "These two sources appear to contain the same data"
2. Side-by-side comparison with column mapping
3. User confirms or rejects the match
4. If confirmed:
   - System creates a column mapping (HEAT_NO → Heat Number → heat_id)
   - Designates one source as authoritative (or uses latest value)
   - Cross-source conflicts become detectable (sanctity layer 3)
5. Future uploads of either format are auto-recognized

**C. Format Fingerprinting**

For recurring uploads (SCADA exports, daily reports):
- Hash of: sorted column names (normalized) + column types + column count
- When a new upload matches a known fingerprint → auto-route to the correct table
- Even if column names vary slightly ("Wt" vs "WT" vs "Weight") — fuzzy match within the fingerprint

#### Data Model

```
FormatFingerprint:
  id: int
  fingerprint_hash: str         # normalized column structure hash
  table_name: str               # target table
  column_mapping: JSON          # {"source_col": "target_col", ...}
  source_variations: JSON       # [list of column name sets that matched this fingerprint]
  match_count: int              # how many times this format was seen
  created_at: datetime

SourceMapping:
  id: int
  source_a_table: str
  source_b_table: str
  column_mappings: JSON         # [{"a": "HEAT_NO", "b": "Heat Number", "canonical": "heat_number"}]
  entity_type: str              # "production_data", "power_readings", etc.
  authoritative_source: str     # which table is the source of truth
  confirmed_by_user: bool
  created_at: datetime
```

---

## 7. Technical Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────┐
│                     FRONTEND (SPA)                       │
│  Feed │ Analyze │ Reports │ Alerts │ Sources │ Setup     │
└──────────────────────┬──────────────────────────────────┘
                       │ REST API
┌──────────────────────┴──────────────────────────────────┐
│                   FastAPI (Async)                         │
│                                                          │
│  ┌─────────┐ ┌──────────┐ ┌─────────┐ ┌──────────────┐ │
│  │ Analyze │ │ Discovery│ │ Sanctity│ │ Alert Engine │ │
│  │ Pipeline│ │ Engine   │ │ Engine  │ │              │ │
│  └─────────┘ └──────────┘ └─────────┘ └──────────────┘ │
│                                                          │
│  ┌─────────────────┐ ┌───────────────┐ ┌─────────────┐ │
│  │ Ingestion Layer │ │ Pattern Memory│ │ Telegram Bot│ │
│  │ (Sheets, API,   │ │               │ │             │ │
│  │  Upload, Sync)  │ │               │ │             │ │
│  └─────────────────┘ └───────────────┘ └─────────────┘ │
└──────────────────────┬──────────────────────────────────┘
                       │
           ┌───────────┴───────────┐
           │  PostgreSQL + pgvector │
           └───────────────────────┘
```

### Data Flow

```
Google Sheet ──polling──→ Sync Engine ──→ ┐
API endpoint ──polling──→ Sync Engine ──→ ├→ Sanctity Check ──→ PostgreSQL
Manual upload ──────────→ DataEngineer →  ├→ Change Log
Recurring upload ───────→ Format Match →  ┘     │
                                                 ▼
                                          Discovery Engine
                                          (profile, relationships, anomalies,
                                           composites, cross-events, patterns)
                                                 │
                                                 ▼
                                     ┌───────────┴───────────┐
                                     │                       │
                                     ▼                       ▼
                                Feed Insights         Alert Evaluation
                                     │                       │
                                     ▼                       ▼
                                  Web UI              Telegram Bot
```

---

## 8. Frontend Changes

### New Tabs

| Tab | Purpose | Default? |
|-----|---------|----------|
| **Feed** | Discovered insights + triggered alerts | Yes (landing page) |
| **Analyze** | Q&A with smart suggestions | No |
| **Reports** | Deployed persistent reports | No |
| **Alerts** | Manage deployed alert rules | New |
| **Sources** | Manage data source connections | New |
| **Setup** | Company profile + onboarding | New (first-run wizard) |
| **Context** | Free-text business context | Existing |
| **Upload** | Manual file upload | Existing |
| **Schema** | View table schemas + data | Existing |

### Feed Tab Enhancements

Current feed shows discovered insights. Enhanced to also show:
- **Triggered alerts** (with timestamp and current value)
- **Sanctity issues** (data quality problems requiring attention)
- **Pattern matches** ("SCADA readings match pre-breakdown pattern — 85% similarity")
- Filters: All | Alerts | Insights | Data Issues
- Priority: Triggered alerts > Critical sanctity issues > High-impact insights > Info

### Alerts Tab

- List of deployed alert rules with status (active/paused/triggered)
- Natural language input at top: "Describe an alert condition..."
- Each rule shows: name, condition summary, last triggered, trigger count
- Click to expand: trigger history, edit threshold, pause/resume, delete

### Sources Tab

- Grid of connected data sources
- Each source card: name, type icon (Sheets/API/Upload), table, last sync, status
- "Add Source" button → type selector → configuration form
- Sync now / Pause / Edit / Disconnect actions

### Setup Tab

- Step-by-step onboarding wizard (6.5)
- Company profile editor
- Metric threshold manager
- Process flow editor

---

## 9. New File Summary

### New Files (estimated)

| File | Lines | Purpose |
|------|-------|---------|
| `src/business_brain/ingestion/sheets_sync.py` | ~200 | Google Sheets connector + polling sync |
| `src/business_brain/ingestion/api_sync.py` | ~120 | API endpoint polling sync |
| `src/business_brain/ingestion/format_matcher.py` | ~150 | Recurring format detection + fingerprinting |
| `src/business_brain/ingestion/sync_engine.py` | ~180 | Orchestrates all sync sources, manages schedules |
| `src/business_brain/db/v3_models.py` | ~150 | DataSource, DataChangeLog, SanctityIssue, AlertRule, AlertEvent, Pattern, PatternMatch, CompanyProfile, MetricThreshold, FormatFingerprint, SourceMapping |
| `src/business_brain/discovery/sanctity_engine.py` | ~200 | Change tracking + impossible value detection + cross-source conflict |
| `src/business_brain/discovery/pattern_memory.py` | ~250 | Pattern learning, storage, matching, confidence adjustment |
| `src/business_brain/discovery/format_detector.py` | ~180 | Same-data-different-format detection + semantic column matching |
| `src/business_brain/action/alert_engine.py` | ~200 | Alert rule evaluation + trigger detection |
| `src/business_brain/action/alert_parser.py` | ~120 | Natural language → structured alert rule (LLM) |
| `src/business_brain/action/telegram_bot.py` | ~150 | Telegram bot: send alerts, registration |
| `src/business_brain/action/onboarding.py` | ~100 | Structured onboarding flow + context generation |

### Modified Files

| File | Changes |
|------|---------|
| `src/business_brain/action/api.py` | ~30 new endpoints |
| `src/business_brain/db/models.py` | Import v3_models |
| `src/business_brain/discovery/engine.py` | Add sanctity + pattern matching to discovery pipeline |
| `src/business_brain/discovery/anomaly_detector.py` | Use metric thresholds from onboarding |
| `public/index.html` | Alerts tab, Sources tab, Setup tab, enhanced Feed |
| `vercel.json` | New routes |

---

## 10. Implementation Order

| Phase | Features | Duration |
|-------|----------|----------|
| **Phase 1** | v3 DB models + Google Sheets sync + recurring ingestion + Sources tab | Build first |
| **Phase 2** | Sanctity engine (all 3 layers) + Data Health UI | Build second |
| **Phase 3** | Alert rule model + natural language parser + alert evaluation engine | Build third |
| **Phase 4** | Telegram bot integration + alert delivery | Build fourth |
| **Phase 5** | Pattern memory (learning + matching) | Build fifth |
| **Phase 6** | Structured onboarding + company profile + metric thresholds | Build sixth |
| **Phase 7** | Same-data-different-format detection + reconciliation | Build seventh |

---

## 11. Future Roadmap (Not in v3)

| Feature | When |
|---------|------|
| **Access control** — role-based (admin, dept_head, viewer) with department-level data isolation | v4 |
| **Multi-tenancy** — per-plant instances with unified management | v4 |
| **Six Sigma agent** — trained on DMAIC methodology, suggests improvements | v5 |
| **5S agent** — workplace organization recommendations | v5 |
| **TPM agent** — total productive maintenance recommendations | v5 |
| **WhatsApp Business API** — when Telegram proves the model | v4 |
| **Mobile app** — React Native, primarily for alerts and quick views | v5 |
| **Offline/on-premise** — Docker deployment for factories with unreliable internet | v5 |
| **Billing & subscription** — per-plant pricing, usage metering | v4 |
| **Custom dashboards** — drag-and-drop dashboard builder | v5 |
| **Anomaly auto-labeling** — system asks "was this really a breakdown?" to build training data | v4 |

---

## 12. Success Metrics

### For the steel company pilot:

| Metric | Target |
|--------|--------|
| Data sources connected | 10+ (of 15 systems) |
| Sheets auto-syncing | 30+ (of 50 sheets) |
| Active alert rules | 15+ across departments |
| Daily active users | 5+ (department heads) |
| Time to detect anomaly | < 5 minutes (from data change to alert) |
| False positive rate on alerts | < 20% |
| "Blame game" incidents | Measurable reduction (qualitative) |
| Pattern predictions confirmed | 3+ correct predictions in first month |

### For commercial launch:

| Metric | Target |
|--------|--------|
| Time to onboard new plant | < 1 day |
| Time to first useful alert | < 1 hour after data connected |
| NPS from plant managers | > 50 |
| Monthly churn | < 5% |

---

## 13. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Google Sheets API rate limits | Batch reads, smart polling (check modifiedTime before full read), exponential backoff |
| SCADA data volume (readings every few seconds) | Aggregate at ingestion: store per-minute averages, keep raw only for pattern analysis windows |
| False positive alerts overwhelming users | Start conservative (high thresholds), let users tune, track false positive rate, auto-adjust |
| LLM hallucination in alert parsing | Always show parsed rule for human confirmation before deployment |
| Telegram bot reliability | Queue-based sending with retry, store unsent alerts for later delivery |
| Pattern matching too slow on large datasets | Pre-compute rolling statistics, match against summaries not raw data |
| User adoption (non-technical users) | Guided onboarding, natural language everything, zero-config defaults that work |

---

*End of PRD*
