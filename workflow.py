"""
workflow.py — Enterprise workflow for ProcureAI
===============================================

  #9  Conversational request intake   — WhatsApp-style slot-filling chat (Claude optional)
  #10 Multi-level approval chain      — value-based approval matrix, sequential sign-off
  #11 Purchase order generation       — GST-compliant PO PDF (CGST+SGST vs IGST) + supplier e-mail
  #13 Spend analytics                 — historical procurement data for the dashboard
"""

from __future__ import annotations

import io
import json
import zlib
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from xml.sax.saxutils import escape

from pydantic import BaseModel, Field
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import llm
from engine import (PurchaseRequirement, SupplierQuotation, format_inr, normalise_casual_text,
                    parse_natural_language_requirement)
from intelligence import STATE_CODES, gstin_checksum

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")

BUYER = {
    "name": "Acme Digital Services Pvt. Ltd.",
    "address": "4th Floor, Prestige Tech Park, Outer Ring Road, Bengaluru 560103",
    "gstin": "29AAACA1234M1Z" + gstin_checksum("29AAACA1234M1Z"),
    "state_code": "29",
}

# ---------------------------------------------------------------------------
# #9 Conversational intake
# ---------------------------------------------------------------------------

SLOTS = [
    ("requested_quantity", "How many laptops do you need?"),
    ("min_ram_gb", "How much RAM is required (e.g. 16 GB)?"),
    ("min_storage_gb", "What SSD storage is required (e.g. 512 GB)?"),
    ("min_warranty_months", "What minimum warranty do you need (e.g. 2 years)?"),
    ("max_delivery_days", "By when do you need delivery (e.g. within 10 days)?"),
    ("max_budget_per_unit", "What is the maximum budget per laptop (e.g. under 60,000)?"),
]
SLOT_LABELS = {
    "requested_quantity": "Quantity", "min_ram_gb": "Min RAM (GB)", "min_storage_gb": "Min Storage (GB)",
    "min_warranty_months": "Min Warranty (months)", "max_delivery_days": "Max Delivery (days)",
    "max_budget_per_unit": "Budget / unit (₹)",
}


class IntakeExtraction(BaseModel):
    requested_quantity: Optional[int] = None
    min_ram_gb: Optional[int] = None
    min_storage_gb: Optional[int] = None
    min_warranty_months: Optional[int] = None
    max_delivery_days: Optional[int] = None
    max_budget_per_unit: Optional[float] = None
    reply: str = Field(description="Short, friendly reply that confirms what was understood and asks for the next missing detail")


def _bare_number_for(slot: str, text: str) -> Optional[float]:
    """Answer to a direct question, e.g. 'how many?' -> '25'."""
    import re

    t = normalise_casual_text(text).replace(",", "")
    t = re.sub(r"^(?:about|around|approx\.?|roughly|maybe|say|max(?:imum)?|at\s+most|up\s+to|under|within|in|by)\s+", "", t)
    t = re.sub(r"\s*(?:each|per\s+\w+|max(?:imum)?|only|please|or\s+so|ish)\s*[.!]?$", "", t)
    m = re.fullmatch(r"\s*(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(gb|tb|days?|weeks?|months?|years?|yrs?|k|lakhs?|laptops?|units?|pcs)?\s*", t)
    if not m:
        return None
    n, unit = float(m.group(1)), (m.group(2) or "")
    if slot == "min_storage_gb" and unit == "tb":
        return n * 1024
    if slot == "min_warranty_months" and unit.startswith(("y", "yr")):
        return n * 12
    if slot == "max_delivery_days" and unit.startswith("week"):
        return n * 7
    if slot == "max_budget_per_unit":
        return n * 1000 if unit == "k" else (n * 100000 if unit.startswith("lakh") else n)
    return n


def intake_turn(message: str, slots: Dict[str, Any], pending_slot: Optional[str], use_ai: bool) -> Tuple[Dict[str, Any], str, Optional[str], str]:
    """Returns (updated_slots, bot_reply, next_pending_slot, mode)."""
    slots = dict(slots)
    mode = "offline"
    ai_reply = None
    if use_ai:
        system = ("You help employees raise laptop purchase requests in a chat. Extract only values the user actually "
                  "stated in this message (convert years to months, weeks to days, TB to GB, '60k' to 60000). "
                  f"Already captured: {json.dumps(slots)}. The bot last asked about: {pending_slot or 'nothing'}.")
        res = llm.structured_call(system, message, IntakeExtraction, effort="low", max_tokens=2000)
        if res:
            mode = "ai"
            for k, _ in SLOTS:
                v = getattr(res, k)
                if v is not None:
                    slots[k] = v
            ai_reply = res.reply
    if mode == "offline":
        parsed = parse_natural_language_requirement(message)
        if not parsed and pending_slot:
            v = _bare_number_for(pending_slot, message)
            if v is not None:
                parsed = {pending_slot: int(v) if pending_slot != "max_budget_per_unit" else float(v)}
        slots.update(parsed)

    missing = [(k, q) for k, q in SLOTS if slots.get(k) is None]
    if missing:
        next_slot, question = missing[0]
        captured = ", ".join(f"{SLOT_LABELS[k]}: {slots[k]:g}" if isinstance(slots[k], float) else f"{SLOT_LABELS[k]}: {slots[k]}"
                             for k, _ in SLOTS if slots.get(k) is not None)
        reply = ai_reply or ((f"Got it — {captured}. " if captured else "Hi! I'll help you raise a purchase request. ") + question)
        return slots, reply, next_slot, mode
    reply = ai_reply or "Perfect, I have everything I need. Review the summary and tap **Apply to requirement**."
    return slots, reply, None, mode


# ---------------------------------------------------------------------------
# #10 Approval chain
# ---------------------------------------------------------------------------

APPROVAL_MATRIX = [
    # (upper limit INR inclusive, roles required)
    (500_000, ["Procurement Manager"]),
    (2_000_000, ["Procurement Manager", "Finance Controller"]),
    (float("inf"), ["Procurement Manager", "Finance Controller", "Director"]),
]


class ApprovalStep(BaseModel):
    role: str
    status: str = "PENDING"       # PENDING | APPROVED | REJECTED
    approver: Optional[str] = None
    time: Optional[str] = None
    comment: Optional[str] = None


class ApprovalChain(BaseModel):
    supplier_name: str
    amount: float
    signature: str
    steps: List[ApprovalStep]
    po_number: Optional[str] = None

    @property
    def status(self) -> str:
        if any(s.status == "REJECTED" for s in self.steps):
            return "REJECTED"
        if all(s.status == "APPROVED" for s in self.steps):
            return "APPROVED"
        return "IN_PROGRESS"

    @property
    def current_index(self) -> Optional[int]:
        for i, s in enumerate(self.steps):
            if s.status == "PENDING":
                return i
            if s.status == "REJECTED":
                return None
        return None


def required_roles(amount: float) -> List[str]:
    for limit, roles in APPROVAL_MATRIX:
        if amount <= limit:
            return roles
    return APPROVAL_MATRIX[-1][1]


def new_chain(supplier: str, amount: float, signature: str) -> ApprovalChain:
    return ApprovalChain(supplier_name=supplier, amount=amount, signature=signature,
                         steps=[ApprovalStep(role=r) for r in required_roles(amount)])


def act_on_chain(chain: ApprovalChain, approve: bool, approver: str, comment: str) -> ApprovalChain:
    idx = chain.current_index
    if idx is None:
        return chain
    step = chain.steps[idx]
    step.status = "APPROVED" if approve else "REJECTED"
    step.approver = approver or step.role
    step.time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    step.comment = comment or None
    if chain.status == "APPROVED" and not chain.po_number:
        chain.po_number = f"PO/{datetime.now():%Y}/{datetime.now():%m%d}-{zlib.crc32(chain.signature.encode()) % 10000:04d}"
    return chain


# ---------------------------------------------------------------------------
# #11 Purchase order
# ---------------------------------------------------------------------------


def _pdf(text: str) -> str:
    for k, v in {"₹": "INR ", "≥": ">=", "≤": "<=", "—": "-"}.items():
        text = text.replace(k, v)
    return escape(text)


def _amt(x: float) -> str:
    return format_inr(x, 2).replace("₹", "INR ")


def tax_split(supplier_gstin: Optional[str], gst_percent: float) -> Tuple[str, float, float]:
    """Intra-state supply -> CGST+SGST (half each); inter-state -> IGST."""
    if supplier_gstin and supplier_gstin[:2] == BUYER["state_code"]:
        return "CGST+SGST", gst_percent / 2, gst_percent / 2
    return "IGST", gst_percent, 0.0


def build_purchase_order_pdf(po_number: str, quotation: SupplierQuotation, quantity: int,
                             requirement: PurchaseRequirement, chain: Optional[ApprovalChain]) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=12 * mm,
                            bottomMargin=12 * mm, title=f"Purchase Order {po_number}")
    ss = getSampleStyleSheet()
    navy = colors.HexColor("#0f3d5e")
    h = ParagraphStyle("h", parent=ss["Title"], fontSize=17, textColor=navy, spaceAfter=2)
    small = ParagraphStyle("s", parent=ss["Normal"], fontSize=8.2, leading=10.5)
    sec = ParagraphStyle("sec", parent=ss["Heading4"], fontSize=9.5, textColor=navy, spaceBefore=6, spaceAfter=3)

    today = datetime.now()
    delivery_by = today + timedelta(days=quotation.delivery_days or requirement.max_delivery_days)
    supplier_state = STATE_CODES.get((quotation.gstin or "")[:2], "—")
    story: List[Any] = [
        Paragraph("PURCHASE ORDER", h),
        Paragraph(f"<b>PO No:</b> {_pdf(po_number)} &nbsp;&nbsp; <b>Date:</b> {today:%d %b %Y} &nbsp;&nbsp; "
                  f"<b>Deliver by:</b> {delivery_by:%d %b %Y}", small),
        Spacer(1, 6),
    ]
    parties = Table([
        [Paragraph("<b>Buyer</b>", small), Paragraph("<b>Supplier</b>", small)],
        [Paragraph(_pdf(f"{BUYER['name']}<br/>{BUYER['address']}<br/>GSTIN: {BUYER['gstin']} "
                        f"(State {BUYER['state_code']} - {STATE_CODES[BUYER['state_code']]})").replace("&lt;br/&gt;", "<br/>"), small),
         Paragraph(_pdf(f"{quotation.supplier_name}<br/>GSTIN: {quotation.gstin or 'NOT PROVIDED'}<br/>"
                        f"Place of supply: {supplier_state}" + ("<br/>MSME / Udyam registered" if quotation.is_msme else ""))
                   .replace("&lt;br/&gt;", "<br/>"), small)],
    ], colWidths=[90 * mm, 90 * mm])
    parties.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
                                 ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story += [parties, Paragraph("Order Lines", sec)]

    taxable = quotation.unit_price * quantity
    mode, r1, r2 = tax_split(quotation.gstin, quotation.gst_percent)
    shipping = quotation.shipping_cost_total
    rows = [["#", "Description", "HSN", "Qty", "Unit Rate", "Taxable Value"],
            ["1", Paragraph(_pdf(f"Business laptop — {quotation.ram_gb} GB RAM, {quotation.storage_gb} GB SSD, "
                                 f"{quotation.warranty_months} months on-site warranty"), small),
             "84713010", str(quantity), _amt(quotation.unit_price), _amt(taxable)]]
    if shipping:
        rows.append(["2", "Freight & delivery (consignment)", "996511", "1", _amt(shipping), _amt(shipping)])
    base = taxable + shipping
    if mode == "IGST":
        tax_rows = [["", "", "", "", f"IGST @ {r1:g}%", _amt(base * r1 / 100)]]
        tax_total = base * r1 / 100
    else:
        tax_rows = [["", "", "", "", f"CGST @ {r1:g}%", _amt(base * r1 / 100)],
                    ["", "", "", "", f"SGST @ {r2:g}%", _amt(base * r2 / 100)]]
        tax_total = base * (r1 + r2) / 100
    rows += [["", "", "", "", "Sub-total", _amt(base)]] + tax_rows + [["", "", "", "", "GRAND TOTAL", _amt(base + tax_total)]]
    t = Table(rows, colWidths=[8 * mm, 72 * mm, 18 * mm, 12 * mm, 32 * mm, 38 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), navy), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, 1 + (1 if shipping else 0)), 0.4, colors.lightgrey),
        ("ALIGN", (3, 0), (-1, -1), "RIGHT"), ("FONTNAME", (4, -1), (-1, -1), "Helvetica-Bold"),
        ("LINEABOVE", (4, -1), (-1, -1), 0.8, navy), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story += [t, Paragraph(f"Tax treatment: {mode} ({'intra-state' if mode != 'IGST' else 'inter-state'} supply). "
                           f"Amounts in INR.", small), Paragraph("Terms &amp; Conditions", sec)]
    terms = [
        f"Delivery: within {quotation.delivery_days} days of PO to the buyer address above; late delivery attracts "
        f"liquidated damages of 0.5% of order value per week, capped at 5%.",
        f"Warranty: {quotation.warranty_months} months comprehensive on-site from date of installation.",
        f"Payment: {quotation.payment_terms_days or 30} days from receipt of goods and valid GST tax invoice quoting this PO number.",
        "Specifications must exactly match the accepted quotation; substitutions require written approval.",
        "Invoices must carry supplier GSTIN, HSN codes and this PO number for input-tax-credit eligibility.",
    ]
    for i, term in enumerate(terms, 1):
        story.append(Paragraph(_pdf(f"{i}. {term}"), small))
    story.append(Paragraph("Authorisation", sec))
    sig_rows = [["Role", "Approved by", "Date / time", "Comment"]]
    if chain:
        for s in chain.steps:
            sig_rows.append([s.role, s.approver or "-", s.time or "-", Paragraph(_pdf(s.comment or "-"), small)])
    st = Table(sig_rows, colWidths=[40 * mm, 45 * mm, 40 * mm, 55 * mm])
    st.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey), ("FONTSIZE", (0, 0), (-1, -1), 8),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke)]))
    story += [st, Spacer(1, 18),
              Paragraph("This is a system-generated purchase order issued after multi-level human approval via ProcureAI.", small)]
    doc.build(story)
    return buffer.getvalue()


def po_email(po_number: str, quotation: SupplierQuotation, quantity: int) -> str:
    total = quotation.unit_price * quantity * (1 + quotation.gst_percent / 100) + quotation.shipping_cost_total * (
        1 + quotation.gst_percent / 100)
    return (f"Subject: Purchase Order {po_number} — {quantity} laptops\n\n"
            f"Dear {quotation.supplier_name} Team,\n\n"
            f"We are pleased to place Purchase Order {po_number} for {quantity} laptops "
            f"({quotation.ram_gb} GB RAM / {quotation.storage_gb} GB SSD / {quotation.warranty_months} months warranty) "
            f"at {format_inr(quotation.unit_price)} per unit, total {format_inr(total)} including GST.\n\n"
            f"Delivery is required within {quotation.delivery_days} days. Please acknowledge this order within 24 hours "
            f"and share the dispatch schedule. The signed PO is attached.\n\n"
            f"Regards,\nProcurement Desk, {BUYER['name']}")


# ---------------------------------------------------------------------------
# #13 Analytics data
# ---------------------------------------------------------------------------


def load_procurement_history(path: Optional[str] = None) -> List[Dict[str, Any]]:
    path = path or os.path.join(DATA_DIR, "procurement_history.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
