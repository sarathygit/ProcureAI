# ProcureAI — Autonomous Requirement-to-Quotation Procurement System

> **Beyond the Lowest Price. Towards the Right Purchase.**

ProcureAI turns a purchase requirement and a pile of supplier quotations (PDF / Excel) into a
compliant, risk-checked, human-approved purchase order — with a full audit trail.

**Core principle:** *rules decide, humans approve.* Compliance checks, landed-cost arithmetic,
rankings, risk and policy are deterministic Python. Nothing is ever assumed: missing or uncertain
specifications trigger clarification or human review, never a default pass.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python generate_samples.py
.venv/bin/streamlit run app.py
```

Click **📂 Load Sample Quotations** to run the demo. The app runs fully offline.

## Features

| Area | What it does |
|---|---|
| 📊 Evaluation | Hard compliance gates (RAM, storage, warranty, delivery, budget), landed cost (base + GST + shipping), strict cost ranking, weighted composite score, trade-off / sensitivity insights |
| ✉️ Clarification loop | Missing specs → `NEEDS_CLARIFICATION`, drafted supplier e-mail, simulated reply that re-ranks live |
| 🔍 Extraction with confidence | Every field gets a confidence score and the evidence sentence; values below 80% are withheld until a human confirms them; understands messy wording ("half a terabyte", "a fortnight") |
| 🛡️ Risk & fraud | Fine-print hidden costs, below/above-market pricing, GSTIN checksum validation, bid-collusion (copied quotation) detection, history-based supplier risk and risk-adjusted cost |
| 🤝 Negotiation agent | Multi-round negotiation using competing quotes as leverage; simulated supplier personas with hidden limits; a human decides whether to apply the terms |
| 🧮 Decision lab | Total Cost of Ownership, what-if simulator, capacity-aware split-order optimizer |
| 📜 Policy & approval | Policy rules (min quotes, GSTIN, risk, collusion, MSME preference), value-based multi-level approval chain, GST-compliant purchase order PDF (CGST+SGST / IGST), 1-page audit memo PDF |
| 📈 Analytics | Spend, savings, cycle time and supplier share dashboards |
| 💬 Request chat | WhatsApp-style intake that understands everyday phrasing ("two dozen laptops, 32 gigs, by end of next week, 85k each") |

## Architecture

```
Quotation files ─► extractor.py / ai_extractor.py ─► values + confidence + evidence
                                                       │  low confidence → human review
                                                       ▼
Requirement ─────► engine.py (compliance, landed cost, ranking, trade-offs)
                                                       ▼
                   intelligence.py (fraud, risk, TCO, what-if, split order, policy)
                   negotiator.py   (negotiation strategy + simulated supplier)
                                                       ▼
                   workflow.py (approval chain, purchase order, chat intake, analytics)
                                                       ▼
                   app.py (Streamlit UI)
```

| File | Purpose |
|---|---|
| `engine.py` | Pydantic schemas, compliance engine, landed cost, scoring, ranking, sensitivity analysis, requirement text parsing |
| `extractor.py` | PDF (PyMuPDF) and Excel (openpyxl) ingestion with regex / keyword rules |
| `ai_extractor.py` | Confidence-scored extraction, heuristics for messy wording, fine-print detection |
| `intelligence.py` | Fraud & hidden-cost detection, supplier risk, TCO, what-if, split-order optimizer, policy rules |
| `negotiator.py` | Negotiation strategy, supplier simulation, guardrails |
| `workflow.py` | Chat intake, approval chain, purchase order PDF, analytics data |
| `llm.py` | Optional Claude integration (off unless `ANTHROPIC_API_KEY` is set) |
| `generate_samples.py` | Deterministic demo data: 5 quotations, supplier history, procurement history |
| `app.py` | Streamlit application |

## Demo data

`generate_samples.py` creates five quotations designed to exercise every path:

- **Apex Solutions** — fully compliant, strong track record (limited capacity: 12 units)
- **Beta Tech Enterprises** — cheapest, but fails RAM / storage / warranty and has an invalid GSTIN
- **Gamma Systems** (Excel) — warranty and delivery left blank → clarification loop; MSME vendor
- **Delta Infotech** — messy wording, hidden costs in the fine print, suspiciously low price
- **Epsilon Traders** — near-copy of Apex's quotation → bid-collusion flag

Supplier history, market reference prices and the 12-month procurement history are **simulated**
for demonstration; GSTIN validation checks format and checksum only. They are designed to be replaced
by real data sources.

## Optional: Claude

Set `ANTHROPIC_API_KEY` to let Claude read very messy or scanned documents and draft negotiation /
chat messages. Claude never makes compliance, cost or ranking decisions, and every feature has a
deterministic offline fallback.

## Deployment

- **App (Render):** `render.yaml` is a Render Blueprint. In Render choose *New → Blueprint*, pick this repo, and deploy.
  The free tier sleeps when idle; open the link a minute before presenting.
- **Landing page (Vercel):** `landing/` is a static site. In Vercel choose *Add New → Project*, import this repo,
  set **Root Directory** to `landing` and **Framework Preset** to *Other*, then deploy.
  The live app is at https://procureai-4az0.onrender.com; if it changes, update `APP_URL` near the bottom of `landing/index.html`.
