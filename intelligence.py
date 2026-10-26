"""
intelligence.py — Decision intelligence for ProcureAI
=====================================================

Deterministic analytics layered on top of the core decision engine:

  #4  Hidden-cost & fraud detection  — fine-print costs, below/above-market pricing,
                                       GSTIN checksum validation, bid-collusion similarity
  #5  Supplier risk score            — history-based risk and risk-adjusted landed cost
  #6  Total Cost of Ownership        — energy, post-warranty repairs, hidden extras, resale
  #7  What-if simulator              — sweep any constraint and watch eligibility/cost move
  #8  Split-order optimizer          — capacity-aware optimal allocation across vendors
  #12 Policy rules engine            — procurement policy checks with PASS / WARN / FAIL
"""

from __future__ import annotations

import itertools
import json
import os
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from engine import (
    STATUS_ELIGIBLE,
    EvaluationResult,
    ProcureAIDecisionEngine,
    PurchaseRequirement,
    SupplierQuotation,
    format_inr,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")

# ---------------------------------------------------------------------------
# #4 Hidden-cost & fraud detection
# ---------------------------------------------------------------------------

# Indicative street prices (INR, ex-GST) for business laptops by (RAM GB, storage GB).
MARKET_REFERENCE: Dict[Tuple[int, int], float] = {
    (8, 256): 38000, (8, 512): 42000, (16, 256): 46000, (16, 512): 50000,
    (16, 1024): 58000, (32, 512): 68000, (32, 1024): 76000, (64, 1024): 110000,
}

GSTIN_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GSTIN_FORMAT = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")
STATE_CODES = {
    "01": "Jammu & Kashmir", "02": "Himachal Pradesh", "03": "Punjab", "04": "Chandigarh", "05": "Uttarakhand",
    "06": "Haryana", "07": "Delhi", "08": "Rajasthan", "09": "Uttar Pradesh", "10": "Bihar", "19": "West Bengal",
    "20": "Jharkhand", "21": "Odisha", "22": "Chhattisgarh", "23": "Madhya Pradesh", "24": "Gujarat",
    "27": "Maharashtra", "29": "Karnataka", "30": "Goa", "32": "Kerala", "33": "Tamil Nadu", "36": "Telangana",
    "37": "Andhra Pradesh",
}


def gstin_checksum(first14: str) -> str:
    total = 0
    for i, ch in enumerate(first14):
        value = GSTIN_CHARS.index(ch) * (1 if i % 2 == 0 else 2)
        total += value // 36 + value % 36
    return GSTIN_CHARS[(36 - total % 36) % 36]


def validate_gstin(gstin: Optional[str]) -> Tuple[str, str]:
    """Returns (status, message) with status in VALID | INVALID | MISSING."""
    if not gstin:
        return "MISSING", "No GSTIN stated on the quotation — input tax credit cannot be claimed."
    g = gstin.strip().upper()
    if not GSTIN_FORMAT.match(g):
        return "INVALID", f"GSTIN {g} does not match the 15-character GSTIN format."
    if g[:2] not in STATE_CODES:
        return "INVALID", f"GSTIN {g} has an unknown state code {g[:2]}."
    if gstin_checksum(g[:14]) != g[14]:
        return "INVALID", f"GSTIN {g} fails the checksum test — possibly mistyped or fabricated."
    return "VALID", f"GSTIN {g} is well-formed ({STATE_CODES[g[:2]]}), checksum verified."


def market_price(ram: Optional[int], storage: Optional[int]) -> Optional[float]:
    if ram is None or storage is None:
        return None
    if (ram, storage) in MARKET_REFERENCE:
        return MARKET_REFERENCE[(ram, storage)]
    key = min(MARKET_REFERENCE, key=lambda k: abs(k[0] - ram) * 3000 + abs(k[1] - storage) * 20)
    return MARKET_REFERENCE[key]


HIDDEN_COST_AMOUNT = re.compile(r"(?:rs\.?|inr|₹)\s*([0-9,]+)\s*(?:/-)?\s*(?:per|/|each|a)\s*(?:unit|laptop|device|machine)", re.I)


class Flag(BaseModel):
    severity: str            # CRITICAL | HIGH | MEDIUM | LOW
    category: str            # Pricing | Hidden cost | Identity | Collusion | Terms
    message: str


class SupplierIntel(BaseModel):
    supplier_name: str
    flags: List[Flag] = Field(default_factory=list)
    gstin_status: str = "MISSING"
    market_price: Optional[float] = None
    price_vs_market: Optional[float] = None
    hidden_cost_per_unit: float = 0.0
    hidden_cost_notes: List[str] = Field(default_factory=list)

    @property
    def critical(self) -> bool:
        return any(f.severity == "CRITICAL" for f in self.flags)


def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[0-9]", "#", text)
    return re.sub(r"\s+", " ", text)


def text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, _normalise(a)[:6000], _normalise(b)[:6000]).ratio()


def analyse_fraud_and_hidden_costs(quotations: List[SupplierQuotation], raw_texts: Dict[str, str],
                                   requirement: PurchaseRequirement) -> Dict[str, SupplierIntel]:
    intel: Dict[str, SupplierIntel] = {}
    for q in quotations:
        si = SupplierIntel(supplier_name=q.supplier_name)
        status, msg = validate_gstin(q.gstin)
        si.gstin_status = status
        if status == "INVALID":
            si.flags.append(Flag(severity="CRITICAL", category="Identity", message=msg))
        elif status == "MISSING":
            si.flags.append(Flag(severity="MEDIUM", category="Identity", message=msg))

        mp = market_price(q.ram_gb, q.storage_gb)
        si.market_price = mp
        if mp and q.unit_price:
            ratio = q.unit_price / mp
            si.price_vs_market = round(ratio, 3)
            if ratio < 0.75:
                si.flags.append(Flag(severity="HIGH", category="Pricing",
                                     message=f"Unit price {format_inr(q.unit_price)} is {1 - ratio:.0%} below the market "
                                             f"reference ({format_inr(mp)}) — too good to be true? Check for refurbished, "
                                             f"grey-market or spec substitution."))
            elif ratio > 1.4:
                si.flags.append(Flag(severity="MEDIUM", category="Pricing",
                                     message=f"Unit price is {ratio - 1:.0%} above the market reference ({format_inr(mp)})."))

        for clause in q.fine_print:
            low = clause.lower()
            sev = "MEDIUM"
            if any(k in low for k in ("refurbished", "renewed", "open box", "advance")):
                sev = "HIGH"
            si.flags.append(Flag(severity=sev, category="Hidden cost" if any(
                k in low for k in ("extra", "freight", "installation", "charged", "ex-works")) else "Terms", message=clause))
            m = HIDDEN_COST_AMOUNT.search(clause)
            if m and any(k in low for k in ("extra", "charged", "additional", "chargeable")):
                amt = float(m.group(1).replace(",", ""))
                si.hidden_cost_per_unit += amt
                si.hidden_cost_notes.append(f"{format_inr(amt)}/unit — {clause[:80]}")
        if any("freight not included" in c.lower() for c in q.fine_print) and q.shipping_cost_total == 0:
            est = 150.0  # typical road freight per laptop when quoted "at actuals"
            si.hidden_cost_per_unit += est
            si.hidden_cost_notes.append(f"{format_inr(est)}/unit — estimated freight (quoted 'at actuals')")

        if q.quote_validity_days is not None and q.quote_validity_days < 15:
            si.flags.append(Flag(severity="MEDIUM", category="Terms",
                                 message=f"Quote valid for only {q.quote_validity_days} days — price may change before PO."))
        intel[q.supplier_name] = si

    # Collusion / template reuse
    names = [q.supplier_name for q in quotations]
    for a, b in itertools.combinations(names, 2):
        sim = text_similarity(raw_texts.get(a, ""), raw_texts.get(b, ""))
        if sim >= 0.80:
            for x, y in ((a, b), (b, a)):
                intel[x].flags.append(Flag(severity="CRITICAL" if sim >= 0.9 else "HIGH", category="Collusion",
                                           message=f"Quotation text is {sim:.0%} identical to {y}'s — possible bid "
                                                   f"rigging or shell-company bids. Verify independence."))
    prices: Dict[float, List[str]] = {}
    for q in quotations:
        if q.unit_price:
            prices.setdefault(q.unit_price, []).append(q.supplier_name)
    for price, group in prices.items():
        if len(group) > 1:
            for n in group:
                intel[n].flags.append(Flag(severity="HIGH", category="Collusion",
                                           message=f"Identical unit price {format_inr(price)} quoted by {', '.join(group)}."))
    order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    for si in intel.values():
        si.flags.sort(key=lambda f: order.get(f.severity, 9))
    return intel


# ---------------------------------------------------------------------------
# #5 Supplier risk score
# ---------------------------------------------------------------------------


class SupplierHistory(BaseModel):
    orders_completed: int = 0
    on_time_rate: float = 0.0
    defect_rate: float = 0.0
    disputes: int = 0
    years_active: float = 0.0
    avg_delay_days: float = 0.0
    capacity_units: Optional[int] = None    # units deliverable within its quoted lead time
    blacklisted: bool = False


class RiskProfile(BaseModel):
    supplier_name: str
    risk_score: float                      # 0 (safe) .. 100 (very risky)
    band: str                              # LOW | MEDIUM | HIGH
    drivers: List[str]
    risk_premium_pct: float
    risk_adjusted_cost: Optional[float] = None
    history: Optional[SupplierHistory] = None


def load_supplier_history(path: Optional[str] = None) -> Dict[str, SupplierHistory]:
    path = path or os.path.join(DATA_DIR, "supplier_history.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {k: SupplierHistory(**v) for k, v in raw.items()}


def _lookup(history: Dict[str, SupplierHistory], name: str) -> Optional[SupplierHistory]:
    if name in history:
        return history[name]
    low = name.lower()
    for k, v in history.items():
        if k.lower() in low or low in k.lower():
            return v
    return None


def score_risk(quotations: List[SupplierQuotation], results: List[EvaluationResult],
               history: Dict[str, SupplierHistory], intel: Dict[str, SupplierIntel]) -> Dict[str, RiskProfile]:
    by_res = {r.supplier_name: r for r in results}
    out: Dict[str, RiskProfile] = {}
    for q in quotations:
        h = _lookup(history, q.supplier_name)
        drivers: List[str] = []
        if h is None:
            score = 55.0
            drivers.append("New vendor — no delivery history on record")
        else:
            late = (1 - h.on_time_rate) * 100
            score = (late * 0.45 + h.defect_rate * 100 * 2.5 + min(h.disputes, 5) * 6
                     + max(0.0, 3 - h.years_active) * 5 + (0 if h.orders_completed >= 10 else 10))
            if h.on_time_rate < 0.85:
                drivers.append(f"On-time delivery only {h.on_time_rate:.0%} (avg delay {h.avg_delay_days:.0f} days)")
            if h.defect_rate >= 0.03:
                drivers.append(f"Defect/DOA rate {h.defect_rate:.1%}")
            if h.disputes:
                drivers.append(f"{h.disputes} commercial dispute(s) on record")
            if h.years_active < 3:
                drivers.append(f"Only {h.years_active:g} year(s) in business")
            if h.orders_completed < 10:
                drivers.append(f"Limited track record ({h.orders_completed} orders)")
            if h.blacklisted:
                score = 100.0
                drivers.insert(0, "Vendor is BLACKLISTED")
            if not drivers:
                drivers.append(f"Strong track record: {h.orders_completed} orders, {h.on_time_rate:.0%} on time")
        si = intel.get(q.supplier_name)
        if si:
            crit = sum(f.severity == "CRITICAL" for f in si.flags)
            high = sum(f.severity == "HIGH" for f in si.flags)
            if crit or high:
                score += crit * 20 + high * 8
                drivers.append(f"{crit} critical / {high} high fraud-or-cost flag(s)")
        score = round(max(0.0, min(100.0, score)), 1)
        band = "LOW" if score < 30 else ("MEDIUM" if score < 60 else "HIGH")
        if h is not None:
            premium = min(0.25, (1 - h.on_time_rate) * 0.10 + h.defect_rate * 1.0 + min(h.disputes, 5) * 0.01)
        else:
            premium = 0.06
        r = by_res.get(q.supplier_name)
        adjusted = round(r.cost_breakdown.total_landed_cost * (1 + premium), 2) if r and r.cost_breakdown else None
        out[q.supplier_name] = RiskProfile(supplier_name=q.supplier_name, risk_score=score, band=band, drivers=drivers,
                                           risk_premium_pct=round(premium * 100, 2), risk_adjusted_cost=adjusted,
                                           history=h)
    return out


# ---------------------------------------------------------------------------
# #6 Total Cost of Ownership
# ---------------------------------------------------------------------------


class TCOAssumptions(BaseModel):
    lifecycle_years: int = 4
    power_watts: float = 45.0
    hours_per_day: float = 9.0
    working_days: int = 250
    electricity_rate: float = 8.0                 # INR per kWh
    annual_repair_rate_out_of_warranty: float = 0.08   # share of unit price per year
    resale_value_pct: float = 0.12
    downtime_cost_per_late_day: float = 0.0       # INR per unit per day beyond deadline (optional)


class TCOResult(BaseModel):
    supplier_name: str
    acquisition: float
    hidden_extras: float
    energy: float
    post_warranty_repairs: float
    resale_credit: float
    tco_total: float
    tco_per_unit: float
    months_out_of_warranty: int


def compute_tco(quotations: List[SupplierQuotation], results: List[EvaluationResult], requirement: PurchaseRequirement,
                assumptions: TCOAssumptions, intel: Dict[str, SupplierIntel]) -> List[TCOResult]:
    by_res = {r.supplier_name: r for r in results}
    qty = requirement.requested_quantity
    out: List[TCOResult] = []
    for q in quotations:
        r = by_res.get(q.supplier_name)
        if not r or not r.cost_breakdown or q.unit_price is None or q.warranty_months is None:
            continue  # TCO needs a confirmed warranty; unknown cover is never assumed
        acquisition = r.cost_breakdown.total_landed_cost
        hidden = (intel.get(q.supplier_name).hidden_cost_per_unit if intel.get(q.supplier_name) else 0.0) * qty * \
            (1 + q.gst_percent / 100.0)
        kwh = assumptions.power_watts / 1000.0 * assumptions.hours_per_day * assumptions.working_days * \
            assumptions.lifecycle_years
        energy = kwh * assumptions.electricity_rate * qty
        life_months = assumptions.lifecycle_years * 12
        warranty = q.warranty_months or 0
        out_months = max(0, life_months - warranty)
        repairs = q.unit_price * assumptions.annual_repair_rate_out_of_warranty * (out_months / 12.0) * qty
        resale = q.unit_price * assumptions.resale_value_pct * qty
        total = acquisition + hidden + energy + repairs - resale
        out.append(TCOResult(
            supplier_name=q.supplier_name, acquisition=round(acquisition, 2), hidden_extras=round(hidden, 2),
            energy=round(energy, 2), post_warranty_repairs=round(repairs, 2), resale_credit=round(resale, 2),
            tco_total=round(total, 2), tco_per_unit=round(total / qty, 2), months_out_of_warranty=out_months,
        ))
    out.sort(key=lambda t: t.tco_total)
    return out


# ---------------------------------------------------------------------------
# #7 What-if simulator
# ---------------------------------------------------------------------------

WHAT_IF_PARAMS = {
    "max_delivery_days": ("Max Delivery (days)", 1),
    "min_warranty_months": ("Min Warranty (months)", 6),
    "min_ram_gb": ("Min RAM (GB)", 4),
    "min_storage_gb": ("Min Storage (GB)", 128),
    "max_budget_per_unit": ("Max Budget per Unit (₹)", 2000),
}


def what_if_sweep(quotations: List[SupplierQuotation], requirement: PurchaseRequirement, param: str,
                  values: List[float]) -> List[Dict[str, Any]]:
    base = ProcureAIDecisionEngine(requirement).evaluate(quotations)
    base_leader = next((r for r in base if r.rank == 1), None)
    base_cost = base_leader.cost_breakdown.total_landed_cost if base_leader else None
    rows = []
    for v in values:
        req = requirement.model_copy(update={param: (int(v) if param != "max_budget_per_unit" else float(v))})
        res = ProcureAIDecisionEngine(req).evaluate(quotations)
        leader = next((r for r in res if r.rank == 1), None)
        cost = leader.cost_breakdown.total_landed_cost if leader else None
        rows.append({
            "value": v,
            "eligible": sum(r.status == STATUS_ELIGIBLE for r in res),
            "best_supplier": leader.supplier_name if leader else "— none —",
            "best_landed_cost": cost,
            "savings_vs_current": round(base_cost - cost, 2) if (base_cost is not None and cost is not None) else None,
            "is_current": abs(v - float(getattr(requirement, param) or 0)) < 1e-9,
        })
    return rows


def sweep_values(requirement: PurchaseRequirement, param: str) -> List[float]:
    cur = getattr(requirement, param)
    step = WHAT_IF_PARAMS[param][1]
    if param == "max_delivery_days":
        return [float(v) for v in range(max(1, cur - 5), cur + 11)]
    if param == "min_warranty_months":
        return [float(v) for v in range(max(0, cur - 24), cur + 25, step)]
    if param == "min_ram_gb":
        return [4.0, 8.0, 12.0, 16.0, 24.0, 32.0]
    if param == "min_storage_gb":
        return [128.0, 256.0, 512.0, 1024.0, 2048.0]
    if param == "max_budget_per_unit":
        base = cur or 60000
        return [float(round(base + d, -3)) for d in range(-20000, 20001, step)]
    return [float(cur)]


# ---------------------------------------------------------------------------
# #8 Split-order optimizer
# ---------------------------------------------------------------------------


class Allocation(BaseModel):
    supplier_name: str
    quantity: int
    unit_price: float
    gst_percent: float
    shipping: float
    line_cost: float


class SplitPlan(BaseModel):
    feasible: bool
    allocations: List[Allocation] = Field(default_factory=list)
    total_cost: Optional[float] = None
    single_vendor_cost: Optional[float] = None
    single_vendor_name: Optional[str] = None
    single_vendor_feasible: bool = False
    message: str = ""


def optimise_split(quotations: List[SupplierQuotation], results: List[EvaluationResult], requirement: PurchaseRequirement,
                   history: Dict[str, SupplierHistory], excluded: Optional[set] = None,
                   max_suppliers: int = 3) -> SplitPlan:
    """Minimise total landed cost to buy the full quantity from ELIGIBLE suppliers only, respecting each supplier's
    capacity within its lead time. Shipping is a fixed cost per supplier used, so every supplier subset is enumerated
    and filled greedily by unit cost (optimal for a fixed subset)."""
    excluded = excluded or set()
    qty = requirement.requested_quantity
    by_res = {r.supplier_name: r for r in results}
    cands = []
    for q in quotations:
        r = by_res.get(q.supplier_name)
        if not r or r.status != STATUS_ELIGIBLE or q.supplier_name in excluded:
            continue
        h = _lookup(history, q.supplier_name)
        cap = h.capacity_units if (h and h.capacity_units) else qty
        unit_with_tax = q.unit_price * (1 + q.gst_percent / 100.0)
        cands.append((q, cap, unit_with_tax))
    if not cands:
        return SplitPlan(feasible=False, message="No eligible supplier available for allocation.")

    best: Optional[Tuple[float, List[Allocation]]] = None
    for k in range(1, min(max_suppliers, len(cands)) + 1):
        for subset in itertools.combinations(cands, k):
            if sum(c[1] for c in subset) < qty:
                continue
            remaining, allocs, total = qty, [], 0.0
            for q, cap, unit in sorted(subset, key=lambda c: c[2]):
                take = min(cap, remaining)
                if take <= 0:
                    break
                line = take * unit + q.shipping_cost_total
                allocs.append(Allocation(supplier_name=q.supplier_name, quantity=take, unit_price=q.unit_price,
                                         gst_percent=q.gst_percent, shipping=q.shipping_cost_total, line_cost=round(line, 2)))
                total += line
                remaining -= take
            if len(allocs) < k:  # a supplier in the subset received nothing — that subset is dominated
                continue
            if best is None or total < best[0] - 1e-6:
                best = (round(total, 2), allocs)

    leader = next((r for r in results if r.rank == 1), None)
    plan = SplitPlan(feasible=best is not None)
    if leader and leader.cost_breakdown:
        lh = _lookup(history, leader.supplier_name)
        plan.single_vendor_name = leader.supplier_name
        plan.single_vendor_cost = leader.cost_breakdown.total_landed_cost
        plan.single_vendor_feasible = not (lh and lh.capacity_units and lh.capacity_units < qty)
    if best is None:
        total_cap = sum(c[1] for c in cands)
        plan.message = (f"Eligible suppliers can deliver only {total_cap} of {qty} units within their lead times. "
                        f"Consider phased delivery or relaxing the deadline.")
        return plan
    plan.total_cost, plan.allocations = best
    if len(plan.allocations) == 1:
        plan.message = f"Single-vendor award to {plan.allocations[0].supplier_name} is optimal and within capacity."
    elif not plan.single_vendor_feasible and plan.single_vendor_name:
        lh = _lookup(history, plan.single_vendor_name)
        plan.message = (f"{plan.single_vendor_name} can only supply {lh.capacity_units if lh else '?'} of {qty} units "
                        f"within its lead time — a split award is required to meet the deadline.")
    else:
        plan.message = "Splitting the order lowers total landed cost."
    return plan


# ---------------------------------------------------------------------------
# #12 Policy rules engine
# ---------------------------------------------------------------------------


class PolicyConfig(BaseModel):
    min_compliant_quotes: int = 3
    require_valid_gstin: bool = True
    max_risk_score: float = 60.0
    block_collusion_flags: bool = True
    msme_preference: bool = True
    msme_margin_pct: float = 15.0               # MSE purchase-preference band (L1 + 15%)
    max_price_vs_market: float = 1.30
    min_quote_validity_days: int = 15


class PolicyFinding(BaseModel):
    rule: str
    status: str             # PASS | WARN | FAIL | INFO
    detail: str
    blocking: bool = False


def check_policies(results: List[EvaluationResult], quotations: List[SupplierQuotation],
                   intel: Dict[str, SupplierIntel], risk: Dict[str, RiskProfile],
                   config: PolicyConfig) -> List[PolicyFinding]:
    findings: List[PolicyFinding] = []
    by_q = {q.supplier_name: q for q in quotations}
    eligible = [r for r in results if r.status == STATUS_ELIGIBLE]
    leader = next((r for r in results if r.rank == 1), None)

    n = len(eligible)
    if n >= config.min_compliant_quotes:
        findings.append(PolicyFinding(rule="Competitive quotes", status="PASS",
                                      detail=f"{n} compliant quotes received (minimum {config.min_compliant_quotes})."))
    else:
        findings.append(PolicyFinding(rule="Competitive quotes", status="WARN",
                                      detail=f"Only {n} compliant quote(s); policy requires {config.min_compliant_quotes}. "
                                             f"A limited-tender justification must be recorded by the approver."))
    if not leader:
        findings.append(PolicyFinding(rule="Recommendation available", status="FAIL", blocking=True,
                                      detail="No compliant supplier — award not permitted."))
        return findings

    name = leader.supplier_name
    q = by_q.get(name)
    si = intel.get(name)
    rp = risk.get(name)

    if config.require_valid_gstin and si:
        if si.gstin_status == "VALID":
            findings.append(PolicyFinding(rule="Valid GSTIN (recommended vendor)", status="PASS",
                                          detail=f"{name}: {q.gstin} verified."))
        else:
            findings.append(PolicyFinding(rule="Valid GSTIN (recommended vendor)", status="FAIL", blocking=True,
                                          detail=f"{name}: GSTIN {si.gstin_status.lower()} — award blocked until verified."))
    if rp:
        if rp.risk_score <= config.max_risk_score:
            findings.append(PolicyFinding(rule="Supplier risk threshold", status="PASS",
                                          detail=f"{name} risk {rp.risk_score:.0f}/100 ≤ {config.max_risk_score:.0f}."))
        else:
            findings.append(PolicyFinding(rule="Supplier risk threshold", status="FAIL", blocking=True,
                                          detail=f"{name} risk {rp.risk_score:.0f}/100 exceeds {config.max_risk_score:.0f}: "
                                                 f"{'; '.join(rp.drivers[:2])}."))
        if rp.history and rp.history.blacklisted:
            findings.append(PolicyFinding(rule="Blacklist check", status="FAIL", blocking=True,
                                          detail=f"{name} is on the debarred-vendor list."))
        else:
            findings.append(PolicyFinding(rule="Blacklist check", status="PASS", detail=f"{name} is not debarred."))
    if config.block_collusion_flags and si:
        coll = [f for f in si.flags if f.category == "Collusion"]
        established = bool(rp and rp.history and rp.history.orders_completed >= 10 and rp.band != "HIGH")
        if coll and established:
            findings.append(PolicyFinding(rule="Bid independence", status="WARN",
                                          detail=f"{name}: {coll[0].message} {name} has an established track record, "
                                                 f"so the copied bid is the likelier culprit — approver must record a "
                                                 f"declaration of bid independence."))
        elif coll:
            findings.append(PolicyFinding(rule="Bid independence", status="FAIL", blocking=True,
                                          detail=f"{name}: {coll[0].message}"))
        else:
            findings.append(PolicyFinding(rule="Bid independence", status="PASS",
                                          detail="No template-reuse or identical-price signals for the recommended vendor."))
    if si and si.price_vs_market:
        if si.price_vs_market <= config.max_price_vs_market:
            findings.append(PolicyFinding(rule="Price reasonableness", status="PASS",
                                          detail=f"{name} is at {si.price_vs_market:.0%} of market reference."))
        else:
            findings.append(PolicyFinding(rule="Price reasonableness", status="WARN",
                                          detail=f"{name} is at {si.price_vs_market:.0%} of market reference "
                                                 f"(limit {config.max_price_vs_market:.0%})."))
    if q and q.quote_validity_days is not None and q.quote_validity_days < config.min_quote_validity_days:
        findings.append(PolicyFinding(rule="Quote validity", status="WARN",
                                      detail=f"{name}'s quote is valid for {q.quote_validity_days} days only — issue PO promptly."))

    if config.msme_preference and leader.cost_breakdown:
        l1 = leader.cost_breakdown.total_landed_cost
        band = l1 * (1 + config.msme_margin_pct / 100.0)
        msme = [r for r in eligible if by_q.get(r.supplier_name) and by_q[r.supplier_name].is_msme
                and r.supplier_name != name and r.cost_breakdown and r.cost_breakdown.total_landed_cost <= band]
        if q and q.is_msme:
            findings.append(PolicyFinding(rule="MSME purchase preference", status="PASS",
                                          detail=f"{name} is itself an MSME vendor."))
        elif msme:
            names = ", ".join(r.supplier_name for r in msme)
            findings.append(PolicyFinding(rule="MSME purchase preference", status="INFO",
                                          detail=f"MSME vendor(s) {names} are within L1+{config.msme_margin_pct:.0f}% and may be "
                                                 f"offered the opportunity to match L1 ({format_inr(l1)}) for up to 25% of the order."))
        else:
            findings.append(PolicyFinding(rule="MSME purchase preference", status="INFO",
                                          detail="No eligible MSME vendor within the preference band."))
    return findings


def approval_blocked(findings: List[PolicyFinding]) -> List[PolicyFinding]:
    return [f for f in findings if f.blocking and f.status == "FAIL"]
