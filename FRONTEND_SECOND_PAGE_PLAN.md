# Strategy Lab — Delegation Plan (Second Frontend Page)

This is a task-oriented implementation plan you can hand to multiple contributors in parallel.

---

## 1) Goal and Definition of Done

### Goal
Ship a second frontend page (`/strategy`) that exposes **model/research lifecycle health** (training quality, optimization recency, robustness, and explainability-lite), while keeping your current `/` page focused on live execution monitoring.

### Product Direction Update (Requested)
All **paper-trading views and controls** are moved from `/` to `/strategy` so execution monitoring and research/simulation workflows are separated.

- `/` (Trading Terminal): spot/CDE prices, market chart, live trades/signals, wallet summary.
- `/strategy` (Strategy Lab): model health, optimization history, explainability, **paper positions/equity/performance/fills**.

### Definition of Done (DoD)
- Route exists for `/strategy` and page renders in production build.
- Paper-trading panels are removed from `/` and available on `/strategy`.
- Page shows:
  - global model health KPIs,
  - per-coin strategy scoreboard,
  - run timeline,
  - feature-importance/signal-distribution panel,
  - paper-trading panels (positions/equity/performance/fills).
- Backend exposes stable API endpoints for all widgets.
- Empty/loading/error states handled for every panel.
- Baseline tests pass (API + frontend).
- README/frontend docs updated with new route and endpoint summary.

---

## 2) Task 1 Started — Product & UX Spec (Delegation-Ready)

## A. Product & UX (Owner: Product/UI)
**Purpose:** Lock scope and interaction details before coding starts.

### A1. Information Architecture (Approved)

### Navigation
- Add a top-level page switch:
  - `Trading Terminal` → `/`
  - `Strategy Lab` → `/strategy`

### `/` page scope (after move)
- Keep:
  - Price cards
  - Market chart
  - Signals table
  - Trades table
  - Wallet summary
- Remove:
  - Paper Positions
  - Paper Equity
  - Paper Performance
  - Paper fills-focused widgets

### `/strategy` page scope
- Add/Keep:
  - Model Health Snapshot (KPI cards)
  - Coin Strategy Scoreboard
  - Experiment Timeline
  - Explainability Lite
  - Paper Trading section:
    - Open positions
    - Equity curve
    - Performance panel
    - Recent fills

### A2. Wireframe (Desktop-first)

### `/strategy` layout (12-column grid)
1. **Row 1 (cols 1–12):** page header + last updated + refresh indicator.
2. **Row 2 (cols 1–12):** Model Health KPI cards (4–6 cards).
3. **Row 3:**
   - cols 1–8: Coin Strategy Scoreboard
   - cols 9–12: Explainability Lite (selected coin)
4. **Row 4 (cols 1–12):** Experiment Timeline.
5. **Row 5 (cols 1–12):** Paper Trading tab group:
   - Positions
   - Equity
   - Performance
   - Fills

### Mobile behavior
- Stack sections in this order: KPIs → Scoreboard → Explainability → Timeline → Paper tabs.
- Paper tabs become horizontally scrollable pill controls.

### A3. KPI Glossary (for UI labels/tooltips)

- **Holdout AUC**: ROC AUC on out-of-sample holdout data. Higher is better.
- **PR-AUC**: Precision-Recall AUC on holdout; more informative under class imbalance.
- **Precision@Threshold**: Fraction of positive predictions that were correct at current signal threshold.
- **Win Rate (Realized)**: % of closed trades with positive net PnL in selected period.
- **Acted Signal Rate**: % of generated signals that were acted on/executed.
- **Drift Delta**: Difference between realized live win rate and expected backtest/holdout win rate.
- **Robustness Gate**: Boolean policy check (e.g., min AUC, minimum sample size, stability constraints).
- **Optimization Freshness**: Time since last successful optimize run for a coin.

### A4. Health Badge Rules (traffic-light)

For each coin row in scorecard:
- **Green (Healthy)**
  - Holdout AUC >= configured min AUC + buffer
  - Drift Delta within tolerance band
  - Last optimization within freshness window
  - Robustness Gate = pass
- **Yellow (Watch)**
  - AUC near threshold OR mild drift OR stale optimization
- **Red (At Risk)**
  - Robustness gate fail OR severe drift OR missing critical artifacts

Initial defaults (configurable):
- `auc_buffer = 0.02`
- `drift_warn = -0.05`
- `drift_fail = -0.10`
- `optimize_stale_days = 14`

### A5. State Copy Spec

#### Loading
- “Loading strategy telemetry…”
- “Fetching paper trading state…”

#### Empty
- “No research runs found yet. Run training/optimization to populate this view.”
- “No paper positions are currently open.”

#### Error
- “Could not load strategy data. Retrying automatically.”
- “Paper data unavailable right now. Check API health and credentials.”

#### Stale-data banner
- Show yellow banner if any panel age > 2x polling interval:
  - “Some data may be stale (last update: {timestamp}).”

### A6. Refresh Cadence Spec
- Strategy KPIs / scoreboard / explainability: every 30–60s.
- Timeline: every 60s.
- Paper section: every 15–30s.
- Manual refresh button in header triggers all panel queries.

### Deliverables (Task 1)
- Wireframe + IA (sections above)
- KPI glossary and health-state rules
- Loading/empty/error copy
- Refresh-cadence policy

### Estimate
- 0.5–1.0 day (completed in plan form; ready for design sign-off)

### Dependencies
- None

---

## 3) Remaining Workstream Breakdown (Parallelizable)

## B. Research API Contract (Owner: Backend Lead)
**Purpose:** Create a contract the frontend can build against.

### Tasks
1. Add endpoint schemas (Pydantic models) for:
   - `GET /research/summary`
   - `GET /research/coins/{coin}`
   - `GET /research/runs?limit=...`
   - `GET /research/features/{coin}`
2. Add/confirm paper endpoint payload compatibility for Strategy Lab consumption:
   - `GET /paper/positions`
   - `GET /paper/equity`
   - `GET /paper/fills`
   - performance derivation contract from equity/fills
3. Define field semantics and units (UTC timestamps, percentages vs decimals, currency).
4. Add example payloads and edge-case responses (no models yet, stale runs, missing features).
5. Add OpenAPI tags + endpoint docs.

### Deliverables
- `backend/api/endpoints/research.py` (new router)
- Response model definitions and docs
- Contract examples in docs

### Estimate
- 1–1.5 days

### Dependencies
- A (field names + UX expectations)

---

## C. Trader Metadata Extraction (Owner: ML/Trader Eng)
**Purpose:** Produce robust inputs for research endpoints.

### Tasks
1. Identify and standardize where run metadata is emitted:
   - train/optimize/validate start/end/status/duration
   - selected params by coin
   - holdout metrics and robustness flags
2. Add minimal persistence layer (table or artifact JSON) for run history.
3. Expose feature-importance snapshot per coin model.
4. Add fallback behavior when artifacts are absent.

### Deliverables
- Normalized run metadata producer(s)
- Stored artifacts readable by API
- Data dictionary for research metrics

### Estimate
- 1.5–2.5 days

### Dependencies
- B (contract expectations), can start in parallel with B after rough schema alignment

---

## D. Backend Integration & Query Layer (Owner: Backend Eng)
**Purpose:** Wire API endpoints to actual data sources.

### Tasks
1. Implement query/services used by `/research/*` endpoints.
2. Add lightweight caching where expensive (e.g., 15–60s for summaries).
3. Implement graceful degradation for missing data.
4. Add endpoint-level tests for success + edge/error paths.

### Deliverables
- Functional `/research/*` endpoints
- Test coverage for contract correctness

### Estimate
- 1–2 days

### Dependencies
- B + C

---

## E. Frontend Foundation (Owner: Frontend Eng #1)
**Purpose:** Add route/page shell and data layer.

### Tasks
1. Add router + top-nav entry (`/strategy`).
2. Move paper tab/panels from `/` to `/strategy`.
3. Create API client module (`frontend/src/api/researchApi.ts`).
4. Define TS types in `frontend/src/types.ts` or split `researchTypes.ts`.
5. Add page-level polling and stale-data indicator.

### Deliverables
- Route + empty page scaffold
- Typed API hooks/helpers
- `/` page cleaned of paper widgets

### Estimate
- 0.5–1.0 day

### Dependencies
- B (contract)

---

## F. Frontend Widgets (Owner: Frontend Eng #2)
**Purpose:** Build visual components and panel logic.

### Tasks
1. **Model Health KPI cards**
2. **Coin Scoreboard table** with health badges
3. **Experiment Timeline** list/table
4. **Explainability panel** (top features + confidence histogram)
5. **Paper section** integration (positions/equity/performance/fills) on `/strategy`
6. Add loading/skeleton/empty/error states for each panel.

### Deliverables
- Reusable components under `frontend/src/components/strategy/`
- Responsive layout matching existing dashboard style

### Estimate
- 1.5–2.5 days

### Dependencies
- E (route/data layer), B contract, D endpoint readiness

---

## G. QA & Hardening (Owner: QA + Dev)
**Purpose:** Prevent regressions and production surprises.

### Tasks
1. API schema checks and response validation.
2. Frontend behavior checks for network failures/timeouts.
3. Verify polling doesn’t overload backend.
4. Validate date/time and numeric formatting consistency.
5. Add regression checklist for both `/` and `/strategy`.
6. Verify paper widgets no longer render on `/` and do render on `/strategy`.

### Deliverables
- Test report + defect list
- Signed release checklist

### Estimate
- 1 day

### Dependencies
- D + F

---

## H. Release & Observability (Owner: Infra/Lead)
**Purpose:** Safe rollout with visibility.

### Tasks
1. Add logging for `/research/*` latency + failure counts.
2. Add simple dashboard/alerts for 5xx spikes.
3. Roll out behind feature flag if desired.
4. Update README/docs with route and endpoint usage.

### Deliverables
- Release notes
- Monitoring hooks
- Updated documentation

### Estimate
- 0.5–1 day

### Dependencies
- D + F + G

---

## 4) Suggested Assignment Matrix

- **Backend Lead:** B + D
- **ML/Trader Engineer:** C
- **Frontend Engineer #1:** E
- **Frontend Engineer #2:** F
- **Product/UI:** A
- **QA:** G
- **Infra/Tech Lead:** H

---

## 5) Sequence Plan (What Can Run in Parallel)

### Day 1
- A starts and finishes scope/spec.
- B starts contract draft.
- C starts metadata normalization with provisional schema.

### Day 2
- B finalizes contract.
- D starts endpoint implementation.
- E starts route + API client based on contract.

### Day 3
- C final integration with D.
- F builds widgets against live/stubbed endpoints.
- D adds tests and caching.

### Day 4
- G executes validation + bug bash.
- H prepares observability and release docs.

### Day 5 (buffer)
- Fixes, polish, and release.

---

## 6) Ticket Backlog (Copy/Paste Ready)

### Epic: Strategy Lab Page

1. **STRAT-01** — Define Strategy Lab UX spec and KPI glossary (A) ✅ started in this plan
2. **STRAT-02** — Create `/research/*` OpenAPI contract and Pydantic models (B)
3. **STRAT-03** — Emit normalized train/optimize/validate metadata artifacts (C)
4. **STRAT-04** — Implement `/research/summary` endpoint + tests (D)
5. **STRAT-05** — Implement `/research/coins/{coin}` endpoint + tests (D)
6. **STRAT-06** — Implement `/research/runs` endpoint + tests (D)
7. **STRAT-07** — Implement `/research/features/{coin}` endpoint + tests (D)
8. **STRAT-08** — Add frontend route/navigation for `/strategy` (E)
9. **STRAT-09** — Add research API client + TS types (E)
10. **STRAT-10** — Build Model Health KPI panel (F)
11. **STRAT-11** — Build Coin Scoreboard panel (F)
12. **STRAT-12** — Build Experiment Timeline panel (F)
13. **STRAT-13** — Build Explainability panel (F)
14. **STRAT-14** — Move/build paper panels on `/strategy` + stale timestamp UI (F)
15. **STRAT-15** — End-to-end QA checklist and performance validation (G)
16. **STRAT-16** — Docs/README update + rollout monitoring hooks (H)

---

## 7) Risks and Mitigations

1. **Risk:** Metrics are inconsistent across scripts/artifacts.
   - **Mitigation:** C defines a strict data dictionary before D wiring.

2. **Risk:** Feature importance unavailable for some models.
   - **Mitigation:** Endpoint returns `status: unavailable` + reason; UI fallback text.

3. **Risk:** Polling creates avoidable backend load.
   - **Mitigation:** 30–60s polling, endpoint caching, and conditional refresh.

4. **Risk:** Timeline data too sparse at launch.
   - **Mitigation:** Backfill last N runs from available logs where possible.

5. **Risk:** UX confusion during page transition.
   - **Mitigation:** Add “Paper Trading moved to Strategy Lab” helper notice on `/` for one release.

---

## 8) Acceptance Checklist (Release Gate)

- [ ] `/strategy` route accessible from navigation.
- [ ] Paper panels removed from `/` and visible on `/strategy`.
- [ ] All research panels render with real data.
- [ ] API endpoints return documented fields and status codes.
- [ ] Frontend handles loading/empty/error states cleanly.
- [ ] Tests pass in CI/local.
- [ ] Docs updated.
- [ ] Basic observability in place.

---

## 9) Nice-to-Have Follow-ups (Post-MVP)

- Deployment gate recommendations (“safe to deploy” automation)
- Model-to-model diff view with threshold impact simulation
- Alerting on drift threshold breaches
- Experiment comparison export (CSV/JSON)
