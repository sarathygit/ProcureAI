"""
app.py — ProcureAI: Autonomous Requirement-to-Quotation Procurement System
==========================================================================
"Beyond the Lowest Price. Towards the Right Purchase."

Run:
    python generate_samples.py
    streamlit run app.py

Runs fully offline on deterministic logic. Claude is an optional add-on (ANTHROPIC_API_KEY) for
reading very messy documents and drafting negotiation / chat messages.
"""

from __future__ import annotations

import io
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from xml.sax.saxutils import escape

import altair as alt
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import llm
from ai_extractor import ExtractionReport, extract_with_confidence, field_label
from engine import (
    FIELD_LABELS,
    STATUS_ELIGIBLE,
    STATUS_NEEDS_CLARIFICATION,
    STATUS_NOT_ELIGIBLE,
    EvaluationResult,
    ProcureAIDecisionEngine,
    PurchaseRequirement,
    SupplierQuotation,
    TradeOffInsight,
    format_inr,
    parse_natural_language_requirement,
)
from intelligence import (
    WHAT_IF_PARAMS,
    PolicyConfig,
    TCOAssumptions,
    analyse_fraud_and_hidden_costs,
    approval_blocked,
    check_policies,
    compute_tco,
    load_supplier_history,
    optimise_split,
    score_risk,
    sweep_values,
    validate_gstin,
    what_if_sweep,
)
from negotiator import MAX_ROUNDS, PERSONAS, default_persona, negotiable, run_round, start_session
from workflow import (
    SLOT_LABELS,
    SLOTS,
    act_on_chain,
    build_purchase_order_pdf,
    intake_turn,
    load_procurement_history,
    new_chain,
    po_email,
    required_roles,
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(APP_DIR, "samples")

STATUS_BADGES = {
    STATUS_ELIGIBLE: "🟢 ELIGIBLE",
    STATUS_NOT_ELIGIBLE: "🔴 NOT_ELIGIBLE",
    STATUS_NEEDS_CLARIFICATION: "🟡 NEEDS_CLARIFICATION",
}
SEVERITY_ICON = {"CRITICAL": "⛔", "HIGH": "🔺", "MEDIUM": "⚠️", "LOW": "ℹ️"}
POLICY_ICON = {"PASS": "✅", "WARN": "⚠️", "FAIL": "⛔", "INFO": "ℹ️"}

# Reference chart palette (dataviz skill defaults): categorical slots in fixed order + text ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK_SECONDARY = "#52514e"

RANGES = {
    "req_ram": (4, 128),
    "req_storage": (128, 8192),
    "req_warranty": (0, 60),
    "req_delivery": (1, 120),
    "req_qty": (1, 100000),
    "req_budget": (1000.0, 10_000_000.0),
}

DEFAULT_STATE: Dict[str, Any] = {
    "nl_text": "We need 20 laptops with 16GB RAM and 512GB SSD within 10 days under 60000",
    "req_ram": 16,
    "req_storage": 512,
    "req_warranty": 24,
    "req_delivery": 10,
    "req_qty": 20,
    "use_budget": True,
    "req_budget": 60000.0,
    "w_cost": 0.50,
    "w_speed": 0.25,
    "w_warranty": 0.25,
    "use_ai": True,
    "approver_name": "",
    "approval_comment": "",
    "award_mode": "Single vendor (Rank #1)",
    "pol_min_quotes": 3,
    "pol_max_risk": 60.0,
    "pol_msme": True,
    "pol_gstin": True,
    "pol_collusion": True,
    "tco_years": 4,
    "tco_watts": 45.0,
    "tco_rate": 8.0,
    "tco_repair": 8.0,
    "tco_resale": 12.0,
    "split_exclude_flagged": True,
}

st.set_page_config(page_title="ProcureAI — Right Purchase Engine", page_icon="🛒", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 3rem;}
      .pa-hero {padding: 1.1rem 1.4rem; border-radius: 14px;
                background: linear-gradient(120deg, #0f3d5e 0%, #1f6f8b 60%, #2a9d8f 100%); color: #fff;
                margin-bottom: 1rem;}
      .pa-hero h1 {margin: 0; font-size: 1.9rem; color: #fff;}
      .pa-hero p {margin: .25rem 0 0 0; opacity: .92;}
      .pa-card {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: .9rem 1.1rem;
                background: rgba(128,128,128,.05); height: 100%;}
      .pa-rec {border-left: 6px solid #2e7d32; border-radius: 10px; padding: 1rem 1.2rem;
               background: rgba(46,125,50,.10);}
      .pa-rec h3 {margin: 0 0 .3rem 0;}
      .pa-muted {opacity: .7; font-size: .88rem;}
      div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); border-radius: 12px;
                                   padding: .6rem .9rem; background: rgba(128,128,128,.05);}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def init_state() -> None:
    for key, value in DEFAULT_STATE.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("reports", {})           # supplier -> ExtractionReport
    st.session_state.setdefault("confirmed", {})         # supplier -> {field: value} (human / supplier / negotiation)
    st.session_state.setdefault("source_files", {})      # supplier -> filename
    st.session_state.setdefault("audit_log", [])
    st.session_state.setdefault("chain", None)           # ApprovalChain
    st.session_state.setdefault("negotiations", {})      # supplier -> NegotiationSession
    st.session_state.setdefault("nl_feedback", None)
    st.session_state.setdefault("clar_feedback", {})
    st.session_state.setdefault("ingest_errors", [])
    st.session_state.setdefault("ai_notes", [])
    st.session_state.setdefault("chat_messages", [
        {"role": "assistant", "content": "Hi! 👋 Tell me what you need to buy — e.g. *\"20 laptops, 16GB RAM by next week\"*."}
    ])
    st.session_state.setdefault("chat_slots", {})
    st.session_state.setdefault("chat_pending", None)


def log_event(event: str, detail: str) -> None:
    st.session_state["audit_log"].append(
        {"Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Event": event, "Detail": detail}
    )


init_state()
AI_ON = llm.llm_configured() and bool(st.session_state["use_ai"])

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def _clamp(key: str, value: float) -> float:
    lo, hi = RANGES[key]
    return max(lo, min(hi, value))


REQ_KEY_MAP = {
    "min_ram_gb": ("req_ram", int),
    "min_storage_gb": ("req_storage", int),
    "min_warranty_months": ("req_warranty", int),
    "max_delivery_days": ("req_delivery", int),
    "requested_quantity": ("req_qty", int),
    "max_budget_per_unit": ("req_budget", float),
}


def _apply_requirement_values(values: Dict[str, Any], source: str) -> List[str]:
    applied: List[str] = []
    for field, (key, cast) in REQ_KEY_MAP.items():
        if values.get(field) is not None:
            st.session_state[key] = cast(_clamp(key, values[field]))
            applied.append(f"{field} = {st.session_state[key]}")
    if values.get("max_budget_per_unit") is not None:
        st.session_state["use_budget"] = True
    if applied:
        log_event(f"Requirement set from {source}", "; ".join(applied))
    return applied


def cb_autofill_from_text() -> None:
    parsed = parse_natural_language_requirement(st.session_state.get("nl_text", ""))
    applied = _apply_requirement_values(parsed, "text")
    st.session_state["nl_feedback"] = (("success", "Auto-filled: " + ", ".join(applied)) if applied
                                       else ("warning", "No recognisable requirements found in the text."))


def cb_apply_chat_requirement() -> None:
    applied = _apply_requirement_values(st.session_state["chat_slots"], "chat intake")
    st.session_state["chat_messages"].append(
        {"role": "assistant", "content": "✅ Applied to the purchase requirement: " + ", ".join(applied)})


def cb_reset_chat() -> None:
    st.session_state["chat_messages"] = st.session_state["chat_messages"][:1]
    st.session_state["chat_slots"] = {}
    st.session_state["chat_pending"] = None


def cb_reset_weights() -> None:
    st.session_state["w_cost"], st.session_state["w_speed"], st.session_state["w_warranty"] = 0.50, 0.25, 0.25


def _unique_name(name: str, existing: Dict[str, Any]) -> str:
    if name not in existing:
        return name
    i = 2
    while f"{name} ({i})" in existing:
        i += 1
    return f"{name} ({i})"


def ingest_files(files: List[tuple], use_ai: bool) -> None:
    reports: Dict[str, ExtractionReport] = {}
    sources: Dict[str, str] = {}
    errors: List[str] = []
    notes: List[str] = []
    for filename, data in files:
        try:
            report = extract_with_confidence(data, filename, use_ai=use_ai)
        except Exception as exc:  # noqa: BLE001 — surface any parse failure to the user
            errors.append(f"{filename}: {exc}")
            continue
        name = _unique_name(report.supplier_name, reports)
        if name != report.supplier_name:
            report.fields["supplier_name"].value = name
        reports[name] = report
        sources[name] = filename
        if report.ai_error:
            notes.append(f"{filename}: {report.ai_error}")
        log_event("Quotation ingested", f"{filename} → {name} ({report.method}, "
                                        f"{len(report.pending_review())} field(s) need review)")
    st.session_state.update(reports=reports, source_files=sources, ingest_errors=errors, ai_notes=notes,
                            confirmed={}, chain=None, negotiations={}, clar_feedback={})


def load_bundled_samples(use_ai: bool) -> None:
    from generate_samples import QUOTATION_FILES, generate_all

    needed = QUOTATION_FILES + ["supplier_history.json", "procurement_history.json"]
    if not all(os.path.exists(os.path.join(SAMPLES_DIR, f)) for f in needed):
        generate_all(SAMPLES_DIR)
        log_event("Samples generated", "samples/ directory was incomplete and was regenerated")
    files = []
    for f in QUOTATION_FILES:
        with open(os.path.join(SAMPLES_DIR, f), "rb") as fh:
            files.append((f, fh.read()))
    ingest_files(files, use_ai)


def cb_confirm_extraction(supplier: str, fields: List[str]) -> None:
    values = {}
    for f in fields:
        v = st.session_state.get(f"rev_{supplier}_{f}")
        if v is None:
            continue
        values[f] = int(v) if f not in {"unit_price", "gst_percent", "shipping_cost_total"} else float(v)
    if values:
        st.session_state["confirmed"].setdefault(supplier, {}).update(values)
        log_event("Extraction confirmed by human",
                  f"{supplier}: " + ", ".join(f"{field_label(k)} = {v}" for k, v in values.items()))


def cb_simulate_reply(supplier: str, fields: List[str]) -> None:
    values: Dict[str, Any] = {}
    for field in fields:
        v = st.session_state.get(f"clar_{supplier}_{field}")
        if v is None:
            st.session_state["clar_feedback"][supplier] = (
                "error", f"Please enter a value for {FIELD_LABELS[field]} — missing values are never assumed.")
            return
        values[field] = float(v) if field == "unit_price" else int(v)
    st.session_state["confirmed"].setdefault(supplier, {}).update(values)
    detail = ", ".join(f"{FIELD_LABELS[k]} = {v}" for k, v in values.items())
    st.session_state["clar_feedback"][supplier] = ("success", f"Inbound reply applied ({detail}). Re-evaluated.")
    log_event("Supplier clarification received", f"{supplier}: {detail}")


def cb_log_email_sent(supplier: str) -> None:
    log_event("Clarification e-mail dispatched", f"Inquiry sent to {supplier} (simulated)")
    st.session_state["clar_feedback"][supplier] = ("info", "Clarification e-mail logged as sent (simulated).")


# ---------------------------------------------------------------------------
# Audit memo PDF
# ---------------------------------------------------------------------------


def _pdf_safe(text: str) -> str:
    for k, v in {"₹": "INR ", "≥": ">=", "≤": "<=", "≠": "!=", "→": "->", "“": '"', "”": '"'}.items():
        text = text.replace(k, v)
    text = text.encode("cp1252", errors="ignore").decode("cp1252")
    return escape(text)


def _inr_pdf(amount: Optional[float]) -> str:
    return format_inr(amount).replace("₹", "INR ") if amount is not None else "-"


def build_audit_memo_pdf(requirement: PurchaseRequirement, results: List[EvaluationResult],
                         insights: List[TradeOffInsight], policy: List[Any], risk: Dict[str, Any],
                         chain: Optional[Any], award_desc: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm,
                            topMargin=11 * mm, bottomMargin=10 * mm,
                            title="ProcureAI Decision Audit Memo", author="ProcureAI")
    styles = getSampleStyleSheet()
    navy = colors.HexColor("#0f3d5e")
    title = ParagraphStyle("T", parent=styles["Title"], fontSize=15, textColor=navy, spaceAfter=1, leading=18)
    sub = ParagraphStyle("S", parent=styles["Normal"], fontSize=7.5, textColor=colors.grey, alignment=1)
    h = ParagraphStyle("H", parent=styles["Heading4"], fontSize=9.5, textColor=navy, spaceBefore=4, spaceAfter=2)
    body = ParagraphStyle("B", parent=styles["Normal"], fontSize=7.6, leading=9.6)
    cell = ParagraphStyle("C", parent=styles["Normal"], fontSize=7, leading=8.5)

    leader = next((r for r in results if r.rank == 1), None)
    story: List[Any] = [
        Paragraph("ProcureAI — Purchase Decision Audit Memo", title),
        Paragraph(f"Generated {datetime.now():%d %b %Y, %H:%M} | Deterministic evaluation engine | "
                  f"Compliance first, cost second", sub),
        Paragraph("1. Purchase Requirement", h),
    ]
    w_total = (requirement.cost_weight + requirement.speed_weight + requirement.warranty_weight) or 1.0
    req_rows = [
        ["Quantity", f"{requirement.requested_quantity} units", "Min RAM / Storage",
         f"{requirement.min_ram_gb} GB / {requirement.min_storage_gb} GB"],
        ["Min Warranty", f"{requirement.min_warranty_months} months", "Max Delivery", f"{requirement.max_delivery_days} days"],
        ["Max Unit Budget", _inr_pdf(requirement.max_budget_per_unit) if requirement.max_budget_per_unit else "Not enforced",
         "Score Weights", f"Cost {requirement.cost_weight / w_total:.0%} / Speed {requirement.speed_weight / w_total:.0%} / "
                          f"Warranty {requirement.warranty_weight / w_total:.0%}"],
    ]
    t = Table(req_rows, colWidths=[28 * mm, 55 * mm, 32 * mm, 67 * mm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 7.4), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"), ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke), ("BACKGROUND", (2, 0), (2, -1), colors.whitesmoke),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story += [t, Paragraph("2. Vendor Evaluation &amp; Ranking", h)]

    header = ["Rank", "Supplier", "Status", "RAM/SSD", "Warr.", "Deliv.", "Landed Cost", "Score", "Risk", "Key Finding"]
    rows: List[List[Any]] = [header]
    status_colors = {STATUS_ELIGIBLE: colors.HexColor("#2e7d32"), STATUS_NOT_ELIGIBLE: colors.HexColor("#c62828"),
                     STATUS_NEEDS_CLARIFICATION: colors.HexColor("#b58900")}
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), navy), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 6.9),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]
    for i, r in enumerate(results, start=1):
        s = r.raw_specs
        finding = r.reasons[0] if r.reasons else ""
        finding = finding[:95] + "..." if len(finding) > 98 else finding
        rp = risk.get(r.supplier_name)
        rows.append([
            str(r.rank) if r.rank else "-",
            Paragraph(_pdf_safe(r.supplier_name), cell),
            {STATUS_NEEDS_CLARIFICATION: "CLARIFY"}.get(r.status, r.status.replace("_", " ")),
            f"{s.get('ram_gb') or '?'}/{s.get('storage_gb') or '?'}",
            f"{s['warranty_months']}m" if s.get("warranty_months") is not None else "?",
            f"{s['delivery_days']}d" if s.get("delivery_days") is not None else "?",
            _inr_pdf(r.cost_breakdown.total_landed_cost) if r.cost_breakdown else "-",
            f"{r.composite_score:.0f}" if r.composite_score is not None else "-",
            f"{rp.risk_score:.0f} {rp.band[0]}" if rp else "-",
            Paragraph(_pdf_safe(finding), cell),
        ])
        style_cmds.append(("TEXTCOLOR", (2, i), (2, i), status_colors.get(r.status, colors.black)))
        style_cmds.append(("FONTNAME", (2, i), (2, i), "Helvetica-Bold"))
        if r.rank == 1:
            style_cmds.append(("BACKGROUND", (0, i), (1, i), colors.HexColor("#e8f5e9")))
    vt = Table(rows, colWidths=[8 * mm, 25 * mm, 20 * mm, 15 * mm, 9 * mm, 9 * mm, 21 * mm, 9 * mm, 11 * mm, 55 * mm],
               repeatRows=1)
    vt.setStyle(TableStyle(style_cmds))
    story += [vt, Paragraph("3. Recommendation &amp; Award", h)]
    if leader and leader.cost_breakdown:
        cb = leader.cost_breakdown
        story.append(Paragraph(_pdf_safe(
            f"Rank #1: {leader.supplier_name} — landed {format_inr(cb.total_landed_cost)} (base {format_inr(cb.base_subtotal)} "
            f"+ GST {format_inr(cb.gst_amount)} + shipping {format_inr(cb.shipping_total)}). Award: {award_desc}"), body))
    else:
        story.append(Paragraph("No supplier meets all mandatory requirements; no award can be made.", body))

    story.append(Paragraph("4. Policy Compliance", h))
    for f in policy[:7]:
        story.append(Paragraph(_pdf_safe(f"[{f.status}] {f.rule}: {f.detail}"), body))
    story.append(Paragraph("5. Trade-off &amp; Risk Notes", h))
    notes = [i.message for i in insights[:3]] or ["No material trade-offs detected."]
    for n in notes:
        story.append(Paragraph(_pdf_safe("• " + n), body))

    story.append(Paragraph("6. Human Authorization (approval chain)", h))
    sig_rows = [["Role", "Status", "Approver", "Date / time", "Comment"]]
    if chain:
        for s in chain.steps:
            sig_rows.append([s.role, s.status, s.approver or "", s.time or "", Paragraph(_pdf_safe(s.comment or ""), cell)])
    else:
        sig_rows.append(["-", "NOT STARTED", "", "", ""])
    sg = Table(sig_rows, colWidths=[35 * mm, 22 * mm, 38 * mm, 32 * mm, 55 * mm])
    sg.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7), ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke)]))
    story += [sg, Spacer(1, 12)]
    sig = Table([["_____________________________", "", "_____________________________"],
                 ["Procurement Authority Signature", "", "Finance / Budget Controller"]],
                colWidths=[70 * mm, 42 * mm, 70 * mm])
    sig.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7.5), ("TEXTCOLOR", (0, 1), (-1, 1), colors.grey)]))
    story += [sig, Spacer(1, 4),
              Paragraph("Compliance, landed-cost, risk and policy decisions were computed by deterministic rules; AI was used "
                        "only to read documents and draft messages. No default values were assumed for missing specifications.",
                        sub)]
    doc.build(story)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🛒 ProcureAI")
    st.caption("Beyond the Lowest Price. Towards the Right Purchase.")
    ok, line = llm.status_line()
    if ok:
        st.success("🤖 " + line)
        st.toggle("Use Claude AI where available", key="use_ai")
    else:
        st.success("⚙️ " + line)
        st.caption("Rules decide · humans approve · no data leaves this machine")

    st.markdown("### 1 · Describe the need")
    st.text_area("Natural-language requirement", key="nl_text", height=95,
                 help="e.g. 'We need 20 laptops with 16GB RAM, 512GB SSD, 2 years warranty within 10 days under 60000'")
    st.button("✨ Auto-Fill From Text", on_click=cb_autofill_from_text, width="stretch")
    if st.session_state["nl_feedback"]:
        level, msg = st.session_state["nl_feedback"]
        getattr(st, level)(msg)

    st.markdown("### 2 · Mandatory constraints")
    st.slider("Min RAM (GB)", *RANGES["req_ram"], step=1, key="req_ram")
    st.number_input("Min Storage (GB)", min_value=RANGES["req_storage"][0], max_value=RANGES["req_storage"][1],
                    step=128, key="req_storage")
    st.slider("Min Warranty (Months)", *RANGES["req_warranty"], step=1, key="req_warranty")
    st.slider("Max Delivery (Days)", *RANGES["req_delivery"], step=1, key="req_delivery")
    st.number_input("Quantity", min_value=RANGES["req_qty"][0], max_value=RANGES["req_qty"][1], step=1, key="req_qty")
    st.checkbox("Enforce per-unit budget", key="use_budget")
    st.number_input("Max Budget per Unit (₹)", min_value=RANGES["req_budget"][0], max_value=RANGES["req_budget"][1],
                    step=1000.0, key="req_budget", disabled=not st.session_state["use_budget"])

    st.markdown("### 3 · Multi-criteria weights")
    st.slider("Cost weight", 0.0, 1.0, step=0.05, key="w_cost")
    st.slider("Speed weight", 0.0, 1.0, step=0.05, key="w_speed")
    st.slider("Warranty weight", 0.0, 1.0, step=0.05, key="w_warranty")
    w_sum = st.session_state["w_cost"] + st.session_state["w_speed"] + st.session_state["w_warranty"]
    if w_sum > 0:
        st.caption(f"Normalised → Cost {st.session_state['w_cost'] / w_sum:.0%} · "
                   f"Speed {st.session_state['w_speed'] / w_sum:.0%} · "
                   f"Warranty {st.session_state['w_warranty'] / w_sum:.0%}")
    else:
        st.caption("All weights are zero — default 50/25/25 will be used.")
    st.button("Reset weights", on_click=cb_reset_weights)
    st.caption("Weights affect the composite score only. Ranking is strictly by lowest landed cost among "
               "compliant vendors.")

requirement = PurchaseRequirement(
    min_ram_gb=int(st.session_state["req_ram"]),
    min_storage_gb=int(st.session_state["req_storage"]),
    min_warranty_months=int(st.session_state["req_warranty"]),
    max_delivery_days=int(st.session_state["req_delivery"]),
    requested_quantity=int(st.session_state["req_qty"]),
    max_budget_per_unit=float(st.session_state["req_budget"]) if st.session_state["use_budget"] else None,
    cost_weight=float(st.session_state["w_cost"]),
    speed_weight=float(st.session_state["w_speed"]),
    warranty_weight=float(st.session_state["w_warranty"]),
)

# ---------------------------------------------------------------------------
# Header & ingestion
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="pa-hero">
      <h1>ProcureAI · Requirement-to-Quotation Engine</h1>
      <p>Beyond the Lowest Price. Towards the Right Purchase. — AI reads and talks, deterministic rules decide,
         humans approve.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

up_col, act_col = st.columns([3, 1.2])
with up_col:
    uploads = st.file_uploader("Upload supplier quotations (.pdf, .xlsx)", type=["pdf", "xlsx"],
                               accept_multiple_files=True)
with act_col:
    st.write("")
    evaluate_clicked = st.button("🚀 Evaluate Quotations", type="primary", width="stretch")
    samples_clicked = st.button("📂 Load Sample Quotations", width="stretch",
                                help="Loads the 5 demo quotations from ./samples (generated if missing)")
    if st.session_state["reports"] and st.button("🧹 Clear", width="stretch"):
        st.session_state.update(reports={}, source_files={}, confirmed={}, chain=None, negotiations={})
        log_event("Session cleared", "All quotations removed")
        st.rerun()

if evaluate_clicked:
    if uploads:
        with st.spinner("Reading quotations" + (" with Claude…" if AI_ON else "…")):
            ingest_files([(f.name, f.getvalue()) for f in uploads], AI_ON)
        log_event("Evaluation run", f"{len(st.session_state['reports'])} quotation(s) evaluated")
    else:
        st.warning("Please upload at least one quotation, or use **Load Sample Quotations**.")
if samples_clicked:
    with st.spinner("Reading sample quotations" + (" with Claude…" if AI_ON else "…")):
        load_bundled_samples(AI_ON)
    log_event("Evaluation run", f"{len(st.session_state['reports'])} sample quotation(s) evaluated")

for err in st.session_state["ingest_errors"]:
    st.error(f"Could not parse {err}")
for note in st.session_state["ai_notes"]:
    st.caption(f"🤖 {note}")

# ---------------------------------------------------------------------------
# Core evaluation (re-runs on every interaction)
# ---------------------------------------------------------------------------

reports: Dict[str, ExtractionReport] = st.session_state["reports"]
confirmed: Dict[str, Dict[str, Any]] = st.session_state["confirmed"]
quotations: List[SupplierQuotation] = [rep.to_quotation(confirmed.get(name, {})) for name, rep in reports.items()]
screening_quotes = [rep.best_guess_quotation().model_copy(update=confirmed.get(name, {})) for name, rep in reports.items()]
raw_texts = {name: rep.raw_text for name, rep in reports.items()}
history = load_supplier_history()

engine = ProcureAIDecisionEngine(requirement)
results = engine.evaluate(quotations) if quotations else []
insights = engine.sensitivity_analysis(quotations, results) if quotations else []
leader = next((r for r in results if r.rank == 1), None)
intel = analyse_fraud_and_hidden_costs(screening_quotes, raw_texts, requirement) if quotations else {}
risk = score_risk(quotations, results, history, intel) if quotations else {}
policy_cfg = PolicyConfig(
    min_compliant_quotes=int(st.session_state["pol_min_quotes"]), max_risk_score=float(st.session_state["pol_max_risk"]),
    msme_preference=bool(st.session_state["pol_msme"]), require_valid_gstin=bool(st.session_state["pol_gstin"]),
    block_collusion_flags=bool(st.session_state["pol_collusion"]),
)
policy = check_policies(results, quotations, intel, risk, policy_cfg) if quotations else []
blockers = approval_blocked(policy)
by_quote = {q.supplier_name: q for q in quotations}

excluded = set()
if st.session_state["split_exclude_flagged"]:
    excluded = {n for n, si in intel.items() if si.critical} | {n for n, rp in risk.items() if rp.band == "HIGH"}
split_plan = optimise_split(quotations, results, requirement, history, excluded=excluded) if quotations else None

tabs = st.tabs(["📊 Evaluation", "🔍 AI Extraction", "🛡️ Risk & Fraud", "🤝 Negotiation", "🧮 Decision Lab",
                "🖊️ Policy & Approval", "📈 Analytics", "💬 Request Chat"])


def _no_quotes_hint() -> None:
    st.info("Upload quotations and click **Evaluate Quotations** — or click **Load Sample Quotations** for the demo.")


# ============================== TAB 1: Evaluation ===========================

with tabs[0]:
    if not quotations:
        _no_quotes_hint()
        c1, c2, c3, c4 = st.columns(4)
        for col, (head, text) in zip((c1, c2, c3, c4), [
            ("1 · Compliance first", "RAM, storage, warranty, delivery and budget are hard gates — never traded for price."),
            ("2 · No silent defaults", "Missing or low-confidence specs trigger NEEDS_CLARIFICATION or human review."),
            ("3 · Agentic", "Drafts clarifications, negotiates counter-offers, detects fraud and hidden costs."),
            ("4 · Human-approved", "Value-based approval chain, GST-compliant PO and a PDF audit memo."),
        ]):
            col.markdown(f'<div class="pa-card"><b>{head}</b><br/><span class="pa-muted">{text}</span></div>',
                         unsafe_allow_html=True)
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Evaluated", len(results))
        m2.metric("🟢 Eligible", sum(r.status == STATUS_ELIGIBLE for r in results))
        m3.metric("🔴 Disqualified", sum(r.status == STATUS_NOT_ELIGIBLE for r in results))
        m4.metric("🟡 Clarifications Needed", sum(r.status == STATUS_NEEDS_CLARIFICATION for r in results))
        st.write("")

        if leader and leader.cost_breakdown:
            cb = leader.cost_breakdown
            runner_up = next((r for r in results if r.rank == 2), None)
            extra = ""
            if runner_up and runner_up.cost_breakdown:
                diff = runner_up.cost_breakdown.total_landed_cost - cb.total_landed_cost
                extra = (f"<li>Saves <b>{format_inr(diff)}</b> versus the next compliant option "
                         f"({escape(runner_up.supplier_name)}).</li>")
            rp = risk.get(leader.supplier_name)
            if rp:
                extra += (f"<li>Supplier risk <b>{rp.risk_score:.0f}/100 ({rp.band})</b> — "
                          f"risk-adjusted cost {format_inr(rp.risk_adjusted_cost)}.</li>")
            reasons_html = "".join(f"<li>{escape(x)}</li>" for x in leader.reasons)
            st.markdown(
                f"""
                <div class="pa-rec">
                  <h3>✅ Recommended: {escape(leader.supplier_name)} — Rank #1</h3>
                  <div style="font-size:1.6rem;font-weight:700;">{format_inr(cb.total_landed_cost)}
                    <span class="pa-muted" style="font-size:.9rem;font-weight:400;"> total landed cost for
                    {requirement.requested_quantity} units · composite score {leader.composite_score:.1f}/100</span></div>
                  <div class="pa-muted">Base {format_inr(cb.base_subtotal)} + GST {format_inr(cb.gst_amount)} +
                    Shipping {format_inr(cb.shipping_total)}</div>
                  <ul style="margin:.5rem 0 0 0;">{reasons_html}{extra}</ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if split_plan and not split_plan.single_vendor_feasible:
                st.warning(f"📦 Capacity: {split_plan.message} See **Decision Lab → Split-order Optimizer**.")
            if blockers:
                st.error("⛔ Award currently blocked by policy: " + " ".join(b.detail for b in blockers))
            if leader.anomalies:
                st.warning("Anomalies on recommended vendor: " + " ".join(leader.anomalies))
        else:
            st.error("⛔ No supplier currently satisfies every mandatory requirement. Resolve clarifications below or "
                     "review the trade-off insights before relaxing any constraint.")

        st.subheader("📊 Side-by-Side Comparison")

        def _v(value: Any, suffix: str = "") -> str:
            return "—" if value is None else f"{value}{suffix}"

        table_rows = []
        for r in results:
            s = r.raw_specs
            rp = risk.get(r.supplier_name)
            si = intel.get(r.supplier_name)
            table_rows.append({
                "Rank": f"#{r.rank}" if r.rank else "—",
                "Supplier": r.supplier_name,
                "Status": STATUS_BADGES.get(r.status, r.status),
                "RAM": _v(s.get("ram_gb"), " GB"),
                "Storage": _v(s.get("storage_gb"), " GB"),
                "Warranty": _v(s.get("warranty_months"), " mo"),
                "Delivery": _v(s.get("delivery_days"), " days"),
                "Unit Price": format_inr(s.get("unit_price")),
                "Landed Cost": format_inr(r.cost_breakdown.total_landed_cost) if r.cost_breakdown else "—",
                "Score": f"{r.composite_score:.1f}" if r.composite_score is not None else "—",
                "Risk": f"{rp.risk_score:.0f} {rp.band}" if rp else "—",
                "Flags": len(si.flags) if si else 0,
            })
        st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch")

        chart_col, detail_col = st.columns([1, 1.3])
        with chart_col:
            cd = pd.DataFrame([{"Supplier": r.supplier_name, "Landed cost (₹)": r.cost_breakdown.total_landed_cost,
                                "Status": r.status} for r in results if r.cost_breakdown])
            if not cd.empty:
                st.caption("Total landed cost by supplier — compliance decides eligibility, not price")
                chart = alt.Chart(cd).mark_bar(cornerRadiusEnd=4, color=SERIES[0]).encode(
                    x=alt.X("Landed cost (₹):Q", axis=alt.Axis(format="~s", gridOpacity=0.3)),
                    y=alt.Y("Supplier:N", sort="x", title=None),
                    tooltip=["Supplier", alt.Tooltip("Landed cost (₹):Q", format=",.0f"), "Status"],
                ).properties(height=alt.Step(34))
                st.altair_chart(chart, width="stretch")
        with detail_col:
            st.caption("Per-vendor reasoning & cost breakdown")
            for r in results:
                with st.expander(f"{STATUS_BADGES.get(r.status)} · {r.supplier_name}"):
                    src = st.session_state["source_files"].get(r.supplier_name)
                    if src:
                        st.caption(f"Source file: {src}")
                    for reason in r.reasons:
                        st.markdown(f"- {reason}")
                    for a in r.anomalies:
                        st.warning(a)
                    if r.cost_breakdown:
                        cb = r.cost_breakdown
                        st.markdown(
                            f"| Component | Amount |\n|---|---:|\n"
                            f"| Unit price × {requirement.requested_quantity} | {format_inr(cb.base_subtotal)} |\n"
                            f"| GST @ {r.raw_specs.get('gst_percent')}% | {format_inr(cb.gst_amount)} |\n"
                            f"| Shipping | {format_inr(cb.shipping_total)} |\n"
                            f"| **Total landed cost** | **{format_inr(cb.total_landed_cost)}** |"
                        )

        pending = [r for r in results if r.status == STATUS_NEEDS_CLARIFICATION]
        st.subheader("✉️ Agentic Clarification Loop")
        if not pending:
            st.success("No open clarifications — every quotation has complete, confirmed mandatory data.")
        else:
            st.caption("Missing or unconfirmed mandatory specs are never assumed. Send the drafted inquiry, then simulate "
                       "the supplier's reply to inject the confirmed values and re-rank live.")
        for r in pending:
            rep = reports.get(r.supplier_name)
            review = [f for f in r.missing_fields if rep and rep.fields.get(f) and rep.fields[f].needs_review
                      and f not in confirmed.get(r.supplier_name, {})]
            with st.expander(f"🟡 {r.supplier_name} — missing: {', '.join(FIELD_LABELS[f] for f in r.missing_fields)}",
                             expanded=True):
                if review:
                    st.info("🔍 " + ", ".join(f"{FIELD_LABELS[f]} was read as **{rep.fields[f].value}** with only "
                                              f"{rep.fields[f].confidence:.0%} confidence" for f in review)
                            + " — confirm it in the **AI Extraction** tab, or ask the supplier below.")
                email_col, reply_col = st.columns([1.4, 1])
                with email_col:
                    st.markdown("**Drafted clarification e-mail**")
                    st.text_area("E-mail draft", r.suggested_clarification_email or "", height=290,
                                 key=f"email_{r.supplier_name}", label_visibility="collapsed")
                    st.button("📤 Log E-mail as Sent", key=f"sent_{r.supplier_name}", on_click=cb_log_email_sent,
                              args=(r.supplier_name,))
                with reply_col:
                    st.markdown("**Simulated inbound supplier reply**")
                    for field in r.missing_fields:
                        if field == "unit_price":
                            st.number_input(FIELD_LABELS[field], min_value=0.0, step=500.0, value=None,
                                            key=f"clar_{r.supplier_name}_{field}", placeholder="Enter quoted value")
                        else:
                            st.number_input(FIELD_LABELS[field], min_value=0, max_value=10000, step=1, value=None,
                                            key=f"clar_{r.supplier_name}_{field}", placeholder="Enter quoted value")
                    st.button("🔄 Simulate Inbound Supplier Reply & Re-Evaluate", key=f"reply_{r.supplier_name}",
                              type="primary", on_click=cb_simulate_reply, args=(r.supplier_name, list(r.missing_fields)),
                              width="stretch")
                fb = st.session_state["clar_feedback"].get(r.supplier_name)
                if fb:
                    getattr(st, fb[0])(fb[1])

        for supplier, (level, msg) in list(st.session_state["clar_feedback"].items()):
            if level == "success" and supplier not in {p.supplier_name for p in pending}:
                status = next((x.status for x in results if x.supplier_name == supplier), None)
                st.success(f"{supplier}: {msg} New status → {STATUS_BADGES.get(status, status)}")
                del st.session_state["clar_feedback"][supplier]

        st.subheader("⚖️ Trade-off & Sensitivity Insights")
        if not insights:
            st.info("No trade-offs detected — no disqualified or pending supplier offers a cheaper path.")
        for ins in insights:
            icon = {"SINGLE_CONSTRAINT": "💡", "MULTI_CONSTRAINT": "🚫", "PENDING_CLARIFICATION": "⏳",
                    "WEIGHTED_LEADER": "🏁"}.get(ins.insight_type, "•")
            text = f"{icon} **{ins.insight_type.replace('_', ' ').title()}** — {ins.message}"
            rp = risk.get(ins.supplier_name)
            if rp and rp.band == "HIGH" and (ins.potential_savings or 0) > 0:
                text += f" ⚠️ But note: {ins.supplier_name} is a HIGH-risk supplier ({rp.risk_score:.0f}/100)."
            if ins.insight_type == "SINGLE_CONSTRAINT" and (ins.potential_savings or 0) > 0:
                st.warning(text)
            elif ins.insight_type == "MULTI_CONSTRAINT":
                st.error(text)
            else:
                st.info(text)

# ============================== TAB 2: AI Extraction =========================

with tabs[1]:
    st.subheader("🔍 Hybrid AI Extraction with Confidence Scores")
    st.caption(f"Every field is read by {'deterministic rules and Claude' if AI_ON else 'deterministic rules plus a casual-wording interpreter'}; "
               f"values below 80% confidence are withheld from the decision engine until a human confirms them.")
    if not reports:
        _no_quotes_hint()
    for name, rep in reports.items():
        pending_fields = [f for f in rep.pending_review() if f not in confirmed.get(name, {})]
        icon = "🟡" if pending_fields else "🟢"
        with st.expander(f"{icon} {name} — {rep.filename} · method: {rep.method}"
                         + (f" · {len(pending_fields)} to review" if pending_fields else ""), expanded=bool(pending_fields)):
            rows = []
            for f, fr in rep.fields.items():
                conf_value = confirmed.get(name, {}).get(f)
                rows.append({
                    "Field": field_label(f),
                    "Value": "—" if fr.value is None else str(fr.value),
                    "Confidence": round(fr.confidence * 100),
                    "Source": fr.source,
                    "Status": ("✅ confirmed → " + str(conf_value)) if conf_value is not None else
                              ("🔍 needs review" if fr.needs_review else ("—" if fr.value is None else "✔ auto-accepted")),
                    "Evidence": fr.evidence or "",
                    "Note": fr.note or "",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                         column_config={"Confidence": st.column_config.ProgressColumn(
                             "Confidence", min_value=0, max_value=100, format="%d%%")})
            if pending_fields:
                st.markdown("**Human review** — confirm or correct the low-confidence values:")
                cols = st.columns(min(4, len(pending_fields)))
                for i, f in enumerate(pending_fields):
                    fr = rep.fields[f]
                    with cols[i % len(cols)]:
                        st.caption(f"“{(fr.evidence or '')[:90]}”")
                        is_float = f in {"unit_price", "gst_percent", "shipping_cost_total"}
                        st.number_input(field_label(f), value=float(fr.value) if is_float else int(fr.value),
                                        step=1.0 if is_float else 1, key=f"rev_{name}_{f}")
                st.button("✅ Confirm reviewed values", key=f"confirm_{name}", type="primary",
                          on_click=cb_confirm_extraction, args=(name, pending_fields))
            if rep.fine_print:
                st.markdown("**Fine print detected:**")
                for fp in rep.fine_print:
                    st.markdown(f"- {fp}")
            with st.popover("Raw extracted text"):
                st.text(rep.raw_text[:5000])

# ============================== TAB 3: Risk & Fraud ==========================

with tabs[2]:
    st.subheader("🛡️ Hidden-Cost, Fraud & Supplier-Risk Screening")
    if not quotations:
        _no_quotes_hint()
    else:
        risk_rows = []
        for q in quotations:
            rp, si = risk.get(q.supplier_name), intel.get(q.supplier_name)
            res = next((r for r in results if r.supplier_name == q.supplier_name), None)
            risk_rows.append({
                "Supplier": q.supplier_name,
                "Risk score": rp.risk_score if rp else None,
                "Band": rp.band if rp else "—",
                "GSTIN": si.gstin_status if si else "—",
                "Price vs market": f"{si.price_vs_market:.0%}" if si and si.price_vs_market else "—",
                "Hidden ₹/unit": format_inr(si.hidden_cost_per_unit) if si and si.hidden_cost_per_unit else "—",
                "Landed cost": format_inr(res.cost_breakdown.total_landed_cost) if res and res.cost_breakdown else "—",
                "Risk-adjusted": format_inr(rp.risk_adjusted_cost) if rp and rp.risk_adjusted_cost else "—",
                "Critical/High flags": (f"{sum(f.severity == 'CRITICAL' for f in si.flags)}/"
                                        f"{sum(f.severity == 'HIGH' for f in si.flags)}") if si else "—",
            })
        st.dataframe(pd.DataFrame(risk_rows), hide_index=True, width="stretch",
                     column_config={"Risk score": st.column_config.ProgressColumn("Risk score", min_value=0,
                                                                                  max_value=100, format="%.0f")})
        st.caption("Risk-adjusted cost = landed cost × (1 + expected cost of late delivery, defects and disputes, "
                   "from the supplier's history).")
        for q in quotations:
            rp, si = risk.get(q.supplier_name), intel.get(q.supplier_name)
            n_flags = len(si.flags) if si else 0
            with st.expander(f"{q.supplier_name} — risk {rp.risk_score:.0f}/100 ({rp.band}) · {n_flags} flag(s)"):
                c1, c2 = st.columns([1, 1.4])
                with c1:
                    st.markdown("**Risk drivers**")
                    for d in rp.drivers:
                        st.markdown(f"- {d}")
                    if rp.history:
                        h = rp.history
                        st.caption(f"History: {h.orders_completed} orders · {h.on_time_rate:.0%} on time · "
                                   f"{h.defect_rate:.1%} defects · {h.disputes} disputes · capacity "
                                   f"{h.capacity_units or '—'} units per lead time")
                    st.markdown(f"**GSTIN:** {validate_gstin(q.gstin)[1]}")
                    if si and si.market_price:
                        st.markdown(f"**Market reference:** {format_inr(si.market_price)}")
                with c2:
                    st.markdown("**Flags**")
                    if not si or not si.flags:
                        st.success("No fraud or hidden-cost flags.")
                    else:
                        for f in si.flags:
                            st.markdown(f"{SEVERITY_ICON.get(f.severity, '•')} **{f.severity}** · {f.category} — {f.message}")
                    if si and si.hidden_cost_notes:
                        st.markdown("**Estimated hidden costs**")
                        for n in si.hidden_cost_notes:
                            st.markdown(f"- {n}")

# ============================== TAB 4: Negotiation ===========================

with tabs[3]:
    st.subheader("🤝 Agentic Negotiation")
    st.caption("Strategy is computed from competing quotes; " + ("Claude drafts our e-mails and role-plays the supplier"
               if AI_ON else "a simulated supplier responds within its persona's hidden concession limits")
               + ". The agent never accepts a deal — you decide whether to apply the negotiated terms.")
    cands = [q for q in quotations if negotiable(q)]
    if not quotations:
        _no_quotes_hint()
    elif not cands:
        st.info("No quotation has the price, delivery and warranty needed to negotiate.")
    else:
        status_of = {r.supplier_name: r.status for r in results}
        names = [q.supplier_name for q in cands]
        nc1, nc2, nc3 = st.columns([1.4, 1, 1])
        target_name = nc1.selectbox("Supplier to negotiate with", names,
                                    format_func=lambda n: f"{n} ({STATUS_BADGES.get(status_of.get(n), '')})",
                                    key="neg_target")
        session = st.session_state["negotiations"].get(target_name)
        persona = nc2.selectbox("Simulated supplier persona", list(PERSONAS),
                                index=list(PERSONAS).index(session.persona if session else default_persona(target_name)),
                                key=f"neg_persona_{target_name}", disabled=session is not None)
        target_q = by_quote[target_name]

        with nc3:
            st.write("")
            if session is None:
                if st.button("▶️ Start negotiation", type="primary", width="stretch"):
                    st.session_state["negotiations"][target_name] = start_session(target_q, quotations, results,
                                                                                   requirement, persona)
                    log_event("Negotiation started", f"{target_name} ({persona})")
                    st.rerun()
            elif st.button("🔁 Reset negotiation", width="stretch"):
                del st.session_state["negotiations"][target_name]
                st.rerun()

        if session:
            ask = session.ask
            st.markdown("**Negotiation strategy (deterministic):** " + " · ".join(filter(None, [
                f"target price {format_inr(ask.target_unit_price)}" if ask.target_unit_price else "",
                f"delivery ≤ {ask.target_delivery_days} days" if ask.target_delivery_days else "",
                f"warranty ≥ {ask.target_warranty_months} months" if ask.target_warranty_months else "",
                "waive freight" if ask.waive_shipping else ""])))
            for lev in ask.leverage:
                st.caption(f"Leverage: {lev}")

            k1, k2, k3, k4 = st.columns(4)
            best = session.best_offer
            k1.metric("Unit price", format_inr(best.unit_price),
                      delta=format_inr(best.unit_price - session.original.unit_price) if session.rounds else None,
                      delta_color="inverse")
            k2.metric("Delivery", f"{best.delivery_days} days",
                      delta=f"{best.delivery_days - session.original.delivery_days} d" if session.rounds else None,
                      delta_color="inverse")
            k3.metric("Warranty", f"{best.warranty_months} mo",
                      delta=f"{best.warranty_months - session.original.warranty_months} mo" if session.rounds else None)
            k4.metric("Negotiated savings", format_inr(session.savings))

            for rnd in session.rounds:
                with st.chat_message("user", avatar="🧑‍💼"):
                    st.markdown(f"**{rnd.buyer_subject}**")
                    st.text(rnd.buyer_message)
                with st.chat_message("assistant", avatar="🏭"):
                    st.text(rnd.supplier_message)
                    st.caption(f"Offer: {format_inr(rnd.offer.unit_price)}/unit · {rnd.offer.delivery_days} days · "
                               f"{rnd.offer.warranty_months} mo warranty · freight {format_inr(rnd.offer.shipping_cost_total)}"
                               f" → landed {format_inr(rnd.landed_cost)} · {STATUS_BADGES.get(rnd.status_after)}")

            b1, b2 = st.columns(2)
            if not session.closed:
                if b1.button(f"📨 Send round {len(session.rounds) + 1} of {MAX_ROUNDS}", type="primary", width="stretch"):
                    with st.spinner("Negotiating…"):
                        run_round(session, target_q, requirement, AI_ON)
                    log_event("Negotiation round", f"{target_name} round {len(session.rounds)}: "
                                                   f"{format_inr(session.best_offer.unit_price)}/unit, "
                                                   f"{session.best_offer.delivery_days} days ({session.mode})")
                    st.rerun()
            else:
                b1.info("Negotiation closed — the supplier has given its final offer.")
            if session.rounds and not session.applied:
                if b2.button("✅ Apply negotiated terms to evaluation (human decision)", width="stretch"):
                    st.session_state["confirmed"].setdefault(target_name, {}).update(session.best_offer.model_dump())
                    session.applied = True
                    log_event("Negotiated terms applied", f"{target_name}: {session.best_offer.model_dump()} "
                                                          f"(saving {format_inr(session.savings)})")
                    st.rerun()
            elif session.applied:
                b2.success("Negotiated terms applied — see the Evaluation tab for the new ranking.")

# ============================== TAB 5: Decision Lab ==========================

with tabs[4]:
    if not quotations:
        _no_quotes_hint()
    else:
        st.subheader("🧮 Total Cost of Ownership")
        t1, t2, t3, t4, t5 = st.columns(5)
        t1.number_input("Lifecycle (years)", 1, 10, key="tco_years")
        t2.number_input("Power draw (W)", 5.0, 300.0, step=5.0, key="tco_watts")
        t3.number_input("Electricity (₹/kWh)", 1.0, 30.0, step=0.5, key="tco_rate")
        t4.number_input("Out-of-warranty repairs (%/yr)", 0.0, 40.0, step=1.0, key="tco_repair")
        t5.number_input("Resale value (%)", 0.0, 60.0, step=1.0, key="tco_resale")
        assumptions = TCOAssumptions(lifecycle_years=int(st.session_state["tco_years"]),
                                     power_watts=float(st.session_state["tco_watts"]),
                                     electricity_rate=float(st.session_state["tco_rate"]),
                                     annual_repair_rate_out_of_warranty=float(st.session_state["tco_repair"]) / 100,
                                     resale_value_pct=float(st.session_state["tco_resale"]) / 100)
        tco = compute_tco(quotations, results, requirement, assumptions, intel)
        if tco:
            status_of = {r.supplier_name: r.status for r in results}
            tco_df = pd.DataFrame([{
                "Supplier": t.supplier_name, "Status": STATUS_BADGES.get(status_of.get(t.supplier_name)),
                "Acquisition": t.acquisition, "Hidden extras": t.hidden_extras, "Energy": t.energy,
                "Post-warranty repairs": t.post_warranty_repairs, "Resale credit": -t.resale_credit,
                "TCO": t.tco_total, "TCO / unit": t.tco_per_unit, "Months uncovered": t.months_out_of_warranty,
            } for t in tco])
            money_cols = ["Acquisition", "Hidden extras", "Energy", "Post-warranty repairs", "Resale credit", "TCO",
                          "TCO / unit"]
            st.dataframe(tco_df.style.format({c: "₹{:,.0f}" for c in money_cols}), hide_index=True, width="stretch")
            comp = ["Acquisition", "Hidden extras", "Energy", "Post-warranty repairs"]
            long = tco_df.melt(id_vars=["Supplier"], value_vars=comp, var_name="Component", value_name="Amount")
            long["order"] = long["Component"].map({c: i for i, c in enumerate(comp)})
            chart = alt.Chart(long).mark_bar(stroke="white", strokeWidth=2).encode(
                x=alt.X("sum(Amount):Q", title="Lifecycle cost before resale (₹)",
                        axis=alt.Axis(format="~s", gridOpacity=0.3)),
                y=alt.Y("Supplier:N", sort=alt.EncodingSortField(field="Amount", op="sum"), title=None),
                color=alt.Color("Component:N", scale=alt.Scale(domain=comp, range=SERIES[:4]),
                                legend=alt.Legend(orient="top", title=None)),
                order=alt.Order("order:Q"),
                tooltip=["Supplier", "Component", alt.Tooltip("Amount:Q", format=",.0f", title="₹")],
            ).properties(height=alt.Step(34))
            st.altair_chart(chart, width="stretch")
            cheapest_landed = min(tco, key=lambda t: t.acquisition)
            cheapest_tco = tco[0]
            if cheapest_tco.supplier_name != cheapest_landed.supplier_name:
                gap = cheapest_landed.tco_total - cheapest_tco.tco_total
                st.info(f"💡 Lowest landed cost is **{cheapest_landed.supplier_name}**, but lowest TCO over "
                        f"{assumptions.lifecycle_years} years is **{cheapest_tco.supplier_name}** (saves {format_inr(gap)}).")
            st.caption("TCO is shown for every priced quote for insight only; awards still require full compliance.")

        st.divider()
        st.subheader("🎛️ What-if Simulator")
        wc1, wc2 = st.columns([1, 3])
        param = wc1.selectbox("Constraint to vary", list(WHAT_IF_PARAMS), format_func=lambda p: WHAT_IF_PARAMS[p][0],
                              key="whatif_param")
        if param == "max_budget_per_unit" and requirement.max_budget_per_unit is None:
            wc2.info("Enable the per-unit budget in the sidebar to simulate budget changes.")
        else:
            sweep = what_if_sweep(quotations, requirement, param, sweep_values(requirement, param))
            sdf = pd.DataFrame(sweep)
            sdf["label"] = sdf["value"].map(lambda v: f"{v:,.0f}")
            with wc2:
                cur = float(getattr(requirement, param))
                base = alt.Chart(sdf).encode(x=alt.X("value:Q", title=WHAT_IF_PARAMS[param][0]))
                line = base.mark_line(strokeWidth=2, color=SERIES[0],
                                      point=alt.OverlayMarkDef(size=64, filled=True, color=SERIES[0])).encode(
                    y=alt.Y("best_landed_cost:Q", title="Best compliant landed cost (₹)",
                            axis=alt.Axis(format="~s", gridOpacity=0.3), scale=alt.Scale(zero=False)),
                    tooltip=[alt.Tooltip("label:N", title="Value"), "best_supplier:N",
                             alt.Tooltip("best_landed_cost:Q", format=",.0f", title="Best landed (₹)"),
                             alt.Tooltip("savings_vs_current:Q", format=",.0f", title="Savings vs now (₹)"),
                             alt.Tooltip("eligible:Q", title="Eligible suppliers")],
                )
                rule = alt.Chart(pd.DataFrame({"value": [cur]})).mark_rule(strokeDash=[4, 4], color=INK_SECONDARY).encode(
                    x="value:Q")
                st.altair_chart((line + rule).properties(
                    height=230, title="Best compliant price as the constraint moves (dashed = current setting)"),
                    width="stretch")
                bars = alt.Chart(sdf).mark_bar(color=SERIES[2], cornerRadiusEnd=4).encode(
                    x=alt.X("label:O", title=WHAT_IF_PARAMS[param][0], sort=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("eligible:Q", title="Eligible suppliers", axis=alt.Axis(tickMinStep=1, gridOpacity=0.3)),
                    tooltip=[alt.Tooltip("label:N", title="Value"), "eligible:Q", "best_supplier:N"],
                ).properties(height=150)
                st.altair_chart(bars, width="stretch")
            best_row = max((r for r in sweep if r["savings_vs_current"] is not None),
                           key=lambda r: r["savings_vs_current"], default=None)
            if best_row and best_row["savings_vs_current"] and best_row["savings_vs_current"] > 0:
                st.success(f"Setting **{WHAT_IF_PARAMS[param][0]} = {best_row['value']:,.0f}** makes "
                           f"**{best_row['best_supplier']}** the best compliant option, saving "
                           f"**{format_inr(best_row['savings_vs_current'])}** versus today.")
                rp_best = risk.get(best_row["best_supplier"])
                if rp_best and rp_best.band == "HIGH":
                    st.warning(f"…but {best_row['best_supplier']} is HIGH risk ({rp_best.risk_score:.0f}/100: "
                               f"{'; '.join(rp_best.drivers[:2])}). Cheapest is not the right purchase.")

        st.divider()
        st.subheader("📦 Split-order Optimizer")
        st.checkbox("Exclude suppliers with critical fraud flags or HIGH risk", key="split_exclude_flagged")
        if excluded:
            st.caption("Excluded: " + ", ".join(sorted(excluded)))
        if split_plan:
            if split_plan.feasible:
                sp_df = pd.DataFrame([{"Supplier": a.supplier_name, "Quantity": a.quantity,
                                       "Unit price": format_inr(a.unit_price), "Freight": format_inr(a.shipping),
                                       "Line landed cost": format_inr(a.line_cost)} for a in split_plan.allocations])
                st.dataframe(sp_df, hide_index=True, width="stretch")
                s1, s2 = st.columns(2)
                s1.metric("Optimised plan total", format_inr(split_plan.total_cost))
                if split_plan.single_vendor_cost:
                    s2.metric(f"Single vendor ({split_plan.single_vendor_name})", format_inr(split_plan.single_vendor_cost),
                              delta="exceeds capacity" if not split_plan.single_vendor_feasible else "within capacity",
                              delta_color="off")
                (st.warning if not split_plan.single_vendor_feasible else st.info)(split_plan.message)
            else:
                st.error(split_plan.message)

# ============================== TAB 6: Policy & Approval ======================

with tabs[5]:
    if not quotations:
        _no_quotes_hint()
    else:
        st.subheader("📜 Procurement Policy Rules")
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.number_input("Min compliant quotes", 1, 10, key="pol_min_quotes")
        p2.number_input("Max supplier risk", 0.0, 100.0, step=5.0, key="pol_max_risk")
        p3.toggle("MSME preference", key="pol_msme")
        p4.toggle("Require valid GSTIN", key="pol_gstin")
        p5.toggle("Block collusion", key="pol_collusion")
        for f in policy:
            st.markdown(f"{POLICY_ICON.get(f.status, '•')} **{f.rule}** — {f.detail}"
                        + ("  `BLOCKING`" if f.blocking and f.status == "FAIL" else ""))

        st.divider()
        st.subheader("🖊️ Award & Multi-level Approval")
        award_options = ["Single vendor (Rank #1)"]
        if split_plan and split_plan.feasible and len(split_plan.allocations) > 1:
            award_options.append("Optimised split order")
        if st.session_state["award_mode"] not in award_options:
            st.session_state["award_mode"] = award_options[0]
        st.radio("Award mode", award_options, key="award_mode", horizontal=True)

        allocations: List[tuple] = []   # (supplier, qty)
        award_amount: Optional[float] = None
        if leader and leader.cost_breakdown:
            if st.session_state["award_mode"] == "Optimised split order" and split_plan and split_plan.feasible:
                allocations = [(a.supplier_name, a.quantity) for a in split_plan.allocations]
                award_amount = split_plan.total_cost
            else:
                allocations = [(leader.supplier_name, requirement.requested_quantity)]
                award_amount = leader.cost_breakdown.total_landed_cost
        award_desc = (" + ".join(f"{n} × {q}" for n, q in allocations) + f" = {format_inr(award_amount)}") \
            if allocations else "—"
        award_suppliers = {n for n, _ in allocations}
        blocker_msgs = [b.detail for b in blockers] + [
            f"{n} has critical fraud flags." for n in award_suppliers
            if n != (leader.supplier_name if leader else None) and intel.get(n) and intel[n].critical
        ]

        signature = json.dumps({"alloc": allocations, "amount": award_amount, "req": requirement.model_dump(),
                                "quotes": {n: by_quote[n].model_dump() for n in award_suppliers}},
                               sort_keys=True, default=str)
        chain = st.session_state["chain"]
        if chain and chain.signature != signature:
            st.warning("⚠️ The award, quotes or requirements changed after the approval chain started — the chain was "
                       "reset and must be re-approved.")
            log_event("Approval chain invalidated", f"Decision changed after chain for {chain.supplier_name} started")
            st.session_state["chain"] = None
            chain = None

        if not allocations:
            st.error("No compliant supplier — nothing to approve.")
        elif blocker_msgs:
            st.error("⛔ Approval blocked: " + " ".join(blocker_msgs))
        else:
            roles = required_roles(award_amount)
            st.markdown(f"**Award:** {award_desc}  \n**Approval matrix:** {format_inr(award_amount)} requires "
                        + " → ".join(roles))
            if chain is None:
                if st.button("🚦 Submit for approval", type="primary"):
                    st.session_state["chain"] = new_chain(" + ".join(n for n, _ in allocations), award_amount, signature)
                    log_event("Approval chain started", f"{award_desc} → {', '.join(roles)}")
                    st.rerun()
            else:
                cols = st.columns(len(chain.steps))
                for col, step in zip(cols, chain.steps):
                    icon = {"APPROVED": "✅", "REJECTED": "❌", "PENDING": "⏳"}[step.status]
                    col.markdown(f'<div class="pa-card"><b>{icon} {step.role}</b><br/><span class="pa-muted">'
                                 f'{escape(step.approver or "awaiting")}<br/>{escape(step.time or "")}<br/>'
                                 f'{escape(step.comment or "")}</span></div>', unsafe_allow_html=True)
                idx = chain.current_index
                if idx is not None:
                    st.write("")
                    current = chain.steps[idx]
                    needs_justification = any(f.status == "WARN" for f in policy) and idx == 0
                    a1, a2 = st.columns([1, 2])
                    a1.text_input(f"{current.role} name", key="approver_name", placeholder=current.role)
                    a2.text_input("Comment" + (" (required: justify policy warnings)" if needs_justification else ""),
                                  key="approval_comment")
                    bb1, bb2, _ = st.columns([1, 1, 2])
                    approve = bb1.button(f"✅ Approve as {current.role}", type="primary")
                    reject = bb2.button("❌ Reject")
                    if approve or reject:
                        if approve and needs_justification and not st.session_state["approval_comment"].strip():
                            st.error("Policy warnings are present — the first approver must record a justification.")
                        else:
                            act_on_chain(chain, approve, st.session_state["approver_name"].strip() or current.role,
                                         st.session_state["approval_comment"].strip())
                            log_event("Approval " + ("granted" if approve else "rejected"),
                                      f"{current.role}: {chain.steps[idx].approver} — {award_desc}")
                            st.rerun()
                if chain.status == "REJECTED":
                    st.error("Award rejected. Reset to resubmit.")
                    if st.button("Reset approval chain"):
                        st.session_state["chain"] = None
                        st.rerun()
                elif chain.status == "APPROVED":
                    st.success(f"🎉 Purchase AUTHORIZED — {award_desc}. PO {chain.po_number}")
                    st.markdown("**Purchase orders**")
                    for i, (n, qty) in enumerate(allocations, start=1):
                        q = by_quote[n]
                        po_no = chain.po_number + (f"-{i}" if len(allocations) > 1 else "")
                        pc1, pc2 = st.columns([1, 2])
                        pc1.download_button(f"📄 PO {po_no} — {n} ({qty})",
                                            data=build_purchase_order_pdf(po_no, q, qty, requirement, chain),
                                            file_name=f"{po_no.replace('/', '_')}.pdf", mime="application/pdf",
                                            key=f"po_{i}", width="stretch")
                        with pc2.expander(f"Draft e-mail to {n}"):
                            st.text(po_email(po_no, q, qty))

        st.divider()
        memo = build_audit_memo_pdf(requirement, results, insights, policy, risk, chain, award_desc)
        st.download_button("📑 Download Audit Decision Memo (PDF)", data=memo,
                           file_name=f"ProcureAI_Audit_Memo_{datetime.now():%Y%m%d_%H%M}.pdf", mime="application/pdf")
        with st.expander("🧾 Audit trail"):
            if st.session_state["audit_log"]:
                st.dataframe(pd.DataFrame(st.session_state["audit_log"][::-1]), hide_index=True, width="stretch")
            else:
                st.caption("No events yet.")

# ============================== TAB 7: Analytics ==============================

with tabs[6]:
    st.subheader("📈 Procurement Analytics")
    events = load_procurement_history()
    if not events:
        st.info("No history found — run `python generate_samples.py` (simulated 12-month history).")
    else:
        hdf = pd.DataFrame(events)
        hdf["date"] = pd.to_datetime(hdf["date"])
        hdf["month"] = hdf["date"].dt.to_period("M").dt.to_timestamp()
        hdf["savings_vs_highest"] = hdf["highest_quote_value"] - hdf["awarded_value"] + hdf["negotiated_savings"]
        st.caption("Simulated 12-month history for demonstration · current session shown separately below.")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total spend (12 mo)", format_inr(hdf["awarded_value"].sum()))
        k2.metric("Savings vs highest bid", format_inr(hdf["savings_vs_highest"].sum()))
        k3.metric("Avg cycle time", f"{hdf['cycle_days'].mean():.1f} days",
                  delta=f"-{hdf['cycle_days_manual_baseline'].mean() - hdf['cycle_days'].mean():.1f} vs manual",
                  delta_color="inverse")
        k4.metric("On-time deliveries", f"{hdf['on_time'].mean():.0%}")

        monthly = hdf.groupby("month", as_index=False).agg(spend=("awarded_value", "sum"),
                                                          savings=("savings_vs_highest", "sum"))
        c1, c2 = st.columns(2)
        with c1:
            st.altair_chart(alt.Chart(monthly).mark_bar(color=SERIES[0], cornerRadiusEnd=4, size=18).encode(
                x=alt.X("month:T", title=None, axis=alt.Axis(format="%b %y")),
                y=alt.Y("spend:Q", title="Spend (₹)", axis=alt.Axis(format="~s", gridOpacity=0.3)),
                tooltip=[alt.Tooltip("month:T", format="%b %Y"), alt.Tooltip("spend:Q", format=",.0f")],
            ).properties(height=240, title="Monthly spend"), width="stretch")
        with c2:
            st.altair_chart(alt.Chart(monthly).mark_line(
                strokeWidth=2, color=SERIES[2], point=alt.OverlayMarkDef(size=64, filled=True, color=SERIES[2])).encode(
                x=alt.X("month:T", title=None, axis=alt.Axis(format="%b %y")),
                y=alt.Y("savings:Q", title="Savings (₹)", axis=alt.Axis(format="~s", gridOpacity=0.3)),
                tooltip=[alt.Tooltip("month:T", format="%b %Y"), alt.Tooltip("savings:Q", format=",.0f")],
            ).properties(height=240, title="Monthly savings vs highest bid"), width="stretch")
        c3, c4 = st.columns(2)
        share = hdf.groupby("winner", as_index=False)["awarded_value"].sum()
        with c3:
            st.altair_chart(alt.Chart(share).mark_bar(color=SERIES[0], cornerRadiusEnd=4, size=18).encode(
                x=alt.X("awarded_value:Q", title="Awarded value (₹)", axis=alt.Axis(format="~s", gridOpacity=0.3)),
                y=alt.Y("winner:N", sort="-x", title=None),
                tooltip=["winner", alt.Tooltip("awarded_value:Q", format=",.0f")],
            ).properties(height=240, title="Spend by supplier"), width="stretch")
        cat = hdf.groupby("category", as_index=False)["awarded_value"].sum()
        with c4:
            st.altair_chart(alt.Chart(cat).mark_bar(color=SERIES[0], cornerRadiusEnd=4, size=18).encode(
                x=alt.X("awarded_value:Q", title="Awarded value (₹)", axis=alt.Axis(format="~s", gridOpacity=0.3)),
                y=alt.Y("category:N", sort="-x", title=None),
                tooltip=["category", alt.Tooltip("awarded_value:Q", format=",.0f")],
            ).properties(height=240, title="Spend by category"), width="stretch")
        with st.expander("Table view"):
            st.dataframe(hdf.drop(columns=["month"]).sort_values("date", ascending=False), hide_index=True,
                         width="stretch")

    st.subheader("This session")
    if quotations:
        priced = [r.cost_breakdown.total_landed_cost for r in results if r.cost_breakdown]
        neg_savings = sum(s.savings for s in st.session_state["negotiations"].values() if s.applied)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Quotes processed", len(results))
        s2.metric("First-pass compliant", f"{sum(r.status == STATUS_ELIGIBLE for r in results) / len(results):.0%}")
        s3.metric("Spread (max − recommended)",
                  format_inr(max(priced) - leader.cost_breakdown.total_landed_cost) if leader and priced else "—")
        s4.metric("Negotiated savings applied", format_inr(neg_savings))
    else:
        st.caption("Load quotations to see live session metrics.")

# ============================== TAB 8: Request Chat ===========================

with tabs[7]:
    st.subheader("💬 Raise a Purchase Request by Chat")
    st.caption("WhatsApp-style intake: the assistant collects every mandatory detail, then fills the requirement. "
               + ("Claude understands free-form replies." if AI_ON else
                  "Understands everyday phrasing — try “two dozen laptops, 32 gigs, a terabyte, three years cover, "
                  "by end of next week, 85k each”."))
    chat_col, slot_col = st.columns([2, 1])
    with chat_col:
        box = st.container(height=420)
        prompt = st.chat_input("Type your request…", key="chat_input")
        if prompt:
            st.session_state["chat_messages"].append({"role": "user", "content": prompt})
            slots, reply, pending_slot, _mode = intake_turn(prompt, st.session_state["chat_slots"],
                                                            st.session_state["chat_pending"], AI_ON)
            st.session_state["chat_slots"], st.session_state["chat_pending"] = slots, pending_slot
            st.session_state["chat_messages"].append({"role": "assistant", "content": reply})
        with box:
            for m in st.session_state["chat_messages"]:
                with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🤖"):
                    st.markdown(m["content"])
    with slot_col:
        st.markdown("**Captured requirement**")
        slots = st.session_state["chat_slots"]
        for k, _ in SLOTS:
            v = slots.get(k)
            st.markdown(f"{'✅' if v is not None else '⬜'} {SLOT_LABELS[k]}: **{v if v is not None else '—'}**")
        st.button("📥 Apply to requirement", type="primary", disabled=not any(v is not None for v in slots.values()),
                  on_click=cb_apply_chat_requirement, width="stretch")
        st.button("Start over", on_click=cb_reset_chat, width="stretch")
        if not all(slots.get(k) is not None for k, _ in SLOTS):
            st.caption("You can apply a partial request; unspecified constraints keep their current values.")
