"""
engine.py — ProcureAI Decision Engine
=====================================

Deterministic core of ProcureAI: "Beyond the Lowest Price. Towards the Right Purchase."

Everything in this module is pure, reproducible Python business logic:
  * Pydantic schemas for requirements, quotations, cost breakdowns and results
  * Mandatory-field validation (missing data -> NEEDS_CLARIFICATION, never a default pass)
  * Hard compliance checks (RAM, storage, warranty, delivery, unit budget)
  * Anomaly detection on prices and shipping
  * Landed-cost arithmetic (base + GST + shipping)
  * Strict cost ranking of eligible vendors + multi-criteria composite score (0-100)
  * Trade-off / sensitivity analysis ("what-if" relaxations that unlock savings)
  * Clarification e-mail drafting (template based, so it is auditable and repeatable)
  * Natural-language requirement parsing (regex based)

No LLM is consulted for any compliance, cost or ranking decision.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

STATUS_ELIGIBLE = "ELIGIBLE"
STATUS_NOT_ELIGIBLE = "NOT_ELIGIBLE"
STATUS_NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"

MANDATORY_FIELDS: List[str] = [
    "unit_price",
    "ram_gb",
    "storage_gb",
    "warranty_months",
    "delivery_days",
]

FIELD_LABELS: Dict[str, str] = {
    "unit_price": "Unit Price (INR, excl. GST)",
    "ram_gb": "RAM (GB)",
    "storage_gb": "Storage (GB)",
    "warranty_months": "Warranty (Months)",
    "delivery_days": "Delivery Lead Time (Days)",
    "gst_percent": "GST (%)",
    "shipping_cost_total": "Shipping / Freight (INR, total)",
}


def format_inr(amount: Optional[float], decimals: int = 0) -> str:
    """Format a number using the Indian digit grouping (e.g. 12,34,567)."""
    if amount is None:
        return "—"
    negative = amount < 0
    amount = abs(amount)
    whole = int(amount)
    frac = amount - whole
    digits = str(whole)
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        digits = ",".join(groups) + "," + tail
    text = digits
    if decimals > 0:
        text += f"{frac:.{decimals}f}"[1:]
    return ("-" if negative else "") + "₹" + text


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class PurchaseRequirement(BaseModel):
    min_ram_gb: int
    min_storage_gb: int
    min_warranty_months: int
    max_delivery_days: int
    requested_quantity: int
    max_budget_per_unit: Optional[float] = None
    cost_weight: float = 0.50
    speed_weight: float = 0.25
    warranty_weight: float = 0.25


class SupplierQuotation(BaseModel):
    supplier_name: str
    unit_price: Optional[float] = None
    ram_gb: Optional[int] = None
    storage_gb: Optional[int] = None
    warranty_months: Optional[int] = None
    delivery_days: Optional[int] = None
    gst_percent: float = 18.0
    shipping_cost_total: float = 0.0
    # Optional commercial metadata (not used for compliance; used by risk, policy and PO modules)
    gstin: Optional[str] = None
    is_msme: Optional[bool] = None
    quote_validity_days: Optional[int] = None
    payment_terms_days: Optional[int] = None
    fine_print: List[str] = Field(default_factory=list)


class CostBreakdown(BaseModel):
    unit_price: float
    base_subtotal: float
    gst_amount: float
    shipping_total: float
    total_landed_cost: float


class EvaluationResult(BaseModel):
    supplier_name: str
    status: str
    reasons: List[str] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)
    anomalies: List[str] = Field(default_factory=list)
    cost_breakdown: Optional[CostBreakdown] = None
    rank: Optional[int] = None
    composite_score: Optional[float] = None
    suggested_clarification_email: Optional[str] = None
    raw_specs: Dict[str, Any] = Field(default_factory=dict)


class ConstraintViolation(BaseModel):
    """A single failed hard constraint, quantified for sensitivity analysis."""

    constraint: str          # e.g. "max_delivery_days"
    label: str               # human-readable label
    offered: float           # value quoted by supplier
    required: float          # threshold set by the buyer
    variance: float          # how far the threshold must move to qualify (always > 0)
    unit: str                # "days", "months", "GB", "INR"
    direction: str           # "increase" (raise a max) or "decrease" (lower a min)


class TradeOffInsight(BaseModel):
    supplier_name: str
    insight_type: str        # SINGLE_CONSTRAINT | MULTI_CONSTRAINT | PENDING_CLARIFICATION | WEIGHTED_LEADER
    constraints: List[ConstraintViolation] = Field(default_factory=list)
    potential_landed_cost: Optional[float] = None
    benchmark_supplier: Optional[str] = None
    benchmark_landed_cost: Optional[float] = None
    potential_savings: Optional[float] = None
    message: str
    priority: int = 3        # 1 = most actionable


# ---------------------------------------------------------------------------
# Decision engine
# ---------------------------------------------------------------------------


class ProcureAIDecisionEngine:
    """Deterministic evaluation of supplier quotations against a purchase requirement."""

    def __init__(self, requirement: PurchaseRequirement):
        self.requirement = requirement

    # ----------------------------- landed cost -----------------------------

    def calculate_landed_cost(self, quotation: SupplierQuotation) -> Optional[CostBreakdown]:
        """Base subtotal + GST + shipping. Returns None when unit price is unknown."""
        if quotation.unit_price is None:
            return None
        qty = self.requirement.requested_quantity
        base_subtotal = round(quotation.unit_price * qty, 2)
        gst_amount = round(base_subtotal * (quotation.gst_percent / 100.0), 2)
        shipping_total = round(quotation.shipping_cost_total, 2)
        total_landed_cost = round(base_subtotal + gst_amount + shipping_total, 2)
        return CostBreakdown(
            unit_price=round(quotation.unit_price, 2),
            base_subtotal=base_subtotal,
            gst_amount=gst_amount,
            shipping_total=shipping_total,
            total_landed_cost=total_landed_cost,
        )

    # ------------------------------ validation -----------------------------

    @staticmethod
    def find_missing_fields(quotation: SupplierQuotation) -> List[str]:
        return [f for f in MANDATORY_FIELDS if getattr(quotation, f) is None]

    def check_constraints(self, quotation: SupplierQuotation) -> List[ConstraintViolation]:
        """Evaluate every hard constraint whose value is known. Unknown values are skipped
        here (they are handled as missing fields, never as a pass)."""
        req = self.requirement
        violations: List[ConstraintViolation] = []

        if quotation.ram_gb is not None and quotation.ram_gb < req.min_ram_gb:
            violations.append(ConstraintViolation(
                constraint="min_ram_gb", label="RAM",
                offered=quotation.ram_gb, required=req.min_ram_gb,
                variance=req.min_ram_gb - quotation.ram_gb, unit="GB", direction="decrease",
            ))
        if quotation.storage_gb is not None and quotation.storage_gb < req.min_storage_gb:
            violations.append(ConstraintViolation(
                constraint="min_storage_gb", label="Storage",
                offered=quotation.storage_gb, required=req.min_storage_gb,
                variance=req.min_storage_gb - quotation.storage_gb, unit="GB", direction="decrease",
            ))
        if quotation.warranty_months is not None and quotation.warranty_months < req.min_warranty_months:
            violations.append(ConstraintViolation(
                constraint="min_warranty_months", label="Warranty",
                offered=quotation.warranty_months, required=req.min_warranty_months,
                variance=req.min_warranty_months - quotation.warranty_months, unit="months",
                direction="decrease",
            ))
        if quotation.delivery_days is not None and quotation.delivery_days > req.max_delivery_days:
            violations.append(ConstraintViolation(
                constraint="max_delivery_days", label="Delivery",
                offered=quotation.delivery_days, required=req.max_delivery_days,
                variance=quotation.delivery_days - req.max_delivery_days, unit="days",
                direction="increase",
            ))
        if (
            req.max_budget_per_unit is not None
            and quotation.unit_price is not None
            and quotation.unit_price > req.max_budget_per_unit
        ):
            violations.append(ConstraintViolation(
                constraint="max_budget_per_unit", label="Unit budget",
                offered=quotation.unit_price, required=req.max_budget_per_unit,
                variance=round(quotation.unit_price - req.max_budget_per_unit, 2), unit="INR",
                direction="increase",
            ))
        return violations

    @staticmethod
    def describe_violation(v: ConstraintViolation) -> str:
        if v.constraint == "min_ram_gb":
            return f"RAM {int(v.offered)} GB is below the mandatory minimum of {int(v.required)} GB (short by {int(v.variance)} GB)."
        if v.constraint == "min_storage_gb":
            return f"Storage {int(v.offered)} GB is below the mandatory minimum of {int(v.required)} GB (short by {int(v.variance)} GB)."
        if v.constraint == "min_warranty_months":
            return f"Warranty {int(v.offered)} months is below the mandatory minimum of {int(v.required)} months (short by {int(v.variance)} months)."
        if v.constraint == "max_delivery_days":
            return f"Delivery in {int(v.offered)} days exceeds the maximum of {int(v.required)} days (late by {int(v.variance)} days)."
        if v.constraint == "max_budget_per_unit":
            return (f"Unit price {format_inr(v.offered)} exceeds the per-unit budget of "
                    f"{format_inr(v.required)} (over by {format_inr(v.variance)}).")
        return f"{v.label}: offered {v.offered}, required {v.required}."

    def detect_anomalies(self, quotation: SupplierQuotation) -> List[str]:
        anomalies: List[str] = []
        budget = self.requirement.max_budget_per_unit
        price = quotation.unit_price
        if price is not None:
            if price <= 0:
                anomalies.append(f"Unit price {format_inr(price)} is zero or negative — likely an extraction or quoting error.")
            if budget is not None and price > budget * 1.5:
                anomalies.append(
                    f"Unit price {format_inr(price)} is more than 150% of the per-unit budget "
                    f"({format_inr(budget)}) — possible mis-quote or wrong product tier."
                )
            if price > 0 and quotation.shipping_cost_total > price * 0.2:
                anomalies.append(
                    f"Shipping {format_inr(quotation.shipping_cost_total)} exceeds 20% of the unit price "
                    f"({format_inr(price * 0.2)}) — verify freight terms."
                )
        if quotation.shipping_cost_total < 0:
            anomalies.append("Shipping cost is negative — invalid value.")
        if not (0.0 <= quotation.gst_percent <= 28.0):
            anomalies.append(f"GST of {quotation.gst_percent}% is outside the valid Indian GST range (0–28%).")
        return anomalies

    # --------------------------- clarification -----------------------------

    def draft_clarification_email(self, quotation: SupplierQuotation, missing_fields: List[str]) -> str:
        req = self.requirement
        expectations = {
            "unit_price": "Unit price per laptop (INR, exclusive of GST)"
                          + (f" — our budget ceiling is {format_inr(req.max_budget_per_unit)} per unit"
                             if req.max_budget_per_unit else ""),
            "ram_gb": f"Installed RAM in GB — our minimum requirement is {req.min_ram_gb} GB",
            "storage_gb": f"SSD storage capacity in GB — our minimum requirement is {req.min_storage_gb} GB",
            "warranty_months": f"Warranty period in months — our minimum requirement is {req.min_warranty_months} months",
            "delivery_days": f"Delivery lead time in calendar days from PO — we require delivery within {req.max_delivery_days} days",
        }
        bullet_lines = "\n".join(
            f"  {i}. {expectations.get(f, FIELD_LABELS.get(f, f))}" for i, f in enumerate(missing_fields, 1)
        )
        subject_fields = ", ".join(FIELD_LABELS.get(f, f).split(" (")[0] for f in missing_fields)
        return (
            f"Subject: Clarification required on your quotation — {subject_fields}\n\n"
            f"Dear {quotation.supplier_name} Team,\n\n"
            f"Thank you for your quotation for {req.requested_quantity} laptops. During our compliance "
            f"review we were unable to locate the following mandatory details:\n\n"
            f"{bullet_lines}\n\n"
            f"As per our procurement policy, quotations with incomplete mandatory specifications cannot "
            f"be evaluated or shortlisted. Kindly reply with the above information, ideally within 48 hours, "
            f"so that your offer can be considered in the current evaluation cycle.\n\n"
            f"Regards,\n"
            f"Procurement Desk — ProcureAI\n"
            f"(Reference generated {datetime.now().strftime('%d %b %Y, %H:%M')})"
        )

    # ------------------------------ evaluation -----------------------------

    def evaluate_single(self, quotation: SupplierQuotation) -> EvaluationResult:
        missing = self.find_missing_fields(quotation)
        violations = self.check_constraints(quotation)
        anomalies = self.detect_anomalies(quotation)
        cost = self.calculate_landed_cost(quotation)
        reasons: List[str] = []

        if violations:
            status = STATUS_NOT_ELIGIBLE
            reasons.extend(self.describe_violation(v) for v in violations)
            if missing:
                reasons.append(
                    "Also missing: " + ", ".join(FIELD_LABELS[f] for f in missing)
                    + " (not pursued — already disqualified on known values)."
                )
        elif missing:
            status = STATUS_NEEDS_CLARIFICATION
            reasons.append(
                "Mandatory specification(s) not stated: " + ", ".join(FIELD_LABELS[f] for f in missing)
                + ". No default or assumed value is applied."
            )
        elif quotation.unit_price is not None and quotation.unit_price <= 0:
            status = STATUS_NEEDS_CLARIFICATION
            missing = ["unit_price"]
            reasons.append("Unit price is zero or negative; a valid price must be confirmed before evaluation.")
        else:
            status = STATUS_ELIGIBLE
            req = self.requirement
            reasons.append(
                f"Meets all mandatory constraints: {quotation.ram_gb} GB RAM (≥{req.min_ram_gb}), "
                f"{quotation.storage_gb} GB storage (≥{req.min_storage_gb}), "
                f"{quotation.warranty_months} months warranty (≥{req.min_warranty_months}), "
                f"{quotation.delivery_days} days delivery (≤{req.max_delivery_days})."
            )
            if req.max_budget_per_unit is not None:
                reasons.append(
                    f"Unit price {format_inr(quotation.unit_price)} is within the per-unit budget of "
                    f"{format_inr(req.max_budget_per_unit)}."
                )

        email = self.draft_clarification_email(quotation, missing) if status == STATUS_NEEDS_CLARIFICATION else None

        return EvaluationResult(
            supplier_name=quotation.supplier_name,
            status=status,
            reasons=reasons,
            missing_fields=missing,
            anomalies=anomalies,
            cost_breakdown=cost,
            suggested_clarification_email=email,
            raw_specs=quotation.model_dump(),
        )

    def _normalized_weights(self) -> Tuple[float, float, float]:
        req = self.requirement
        weights = [max(0.0, req.cost_weight), max(0.0, req.speed_weight), max(0.0, req.warranty_weight)]
        total = sum(weights)
        if total <= 0:
            return 0.50, 0.25, 0.25
        return weights[0] / total, weights[1] / total, weights[2] / total

    def score_eligible(self, results: List[EvaluationResult]) -> None:
        """Composite 0–100 score using ratio-to-best normalisation for each criterion:
        cost   -> cheapest / this   (lower is better)
        speed  -> fastest  / this   (lower is better)
        warranty -> this / longest  (higher is better)
        """
        eligible = [r for r in results if r.status == STATUS_ELIGIBLE and r.cost_breakdown]
        if not eligible:
            return
        w_cost, w_speed, w_warranty = self._normalized_weights()
        min_cost = min(r.cost_breakdown.total_landed_cost for r in eligible)
        min_days = min(max(1, r.raw_specs["delivery_days"]) for r in eligible)
        max_warranty = max(r.raw_specs["warranty_months"] for r in eligible)

        for r in eligible:
            cost = r.cost_breakdown.total_landed_cost
            days = max(1, r.raw_specs["delivery_days"])
            warranty = r.raw_specs["warranty_months"]
            cost_score = (min_cost / cost) if cost > 0 else 0.0
            speed_score = min_days / days
            warranty_score = (warranty / max_warranty) if max_warranty > 0 else 1.0
            r.composite_score = round(100.0 * (w_cost * cost_score + w_speed * speed_score
                                               + w_warranty * warranty_score), 2)

    @staticmethod
    def rank_eligible(results: List[EvaluationResult]) -> None:
        """Strict ranking of ELIGIBLE vendors by lowest total landed cost.
        Ties are broken by higher composite score, then supplier name (deterministic)."""
        eligible = [r for r in results if r.status == STATUS_ELIGIBLE and r.cost_breakdown]
        eligible.sort(key=lambda r: (r.cost_breakdown.total_landed_cost,
                                     -(r.composite_score or 0.0), r.supplier_name.lower()))
        for i, r in enumerate(eligible, start=1):
            r.rank = i

    def evaluate(self, quotations: List[SupplierQuotation]) -> List[EvaluationResult]:
        results = [self.evaluate_single(q) for q in quotations]
        self.score_eligible(results)
        self.rank_eligible(results)
        order = {STATUS_ELIGIBLE: 0, STATUS_NEEDS_CLARIFICATION: 1, STATUS_NOT_ELIGIBLE: 2}
        results.sort(key=lambda r: (order.get(r.status, 3), r.rank if r.rank is not None else 10 ** 6,
                                    r.supplier_name.lower()))
        return results

    # ------------------------- sensitivity analysis ------------------------

    @staticmethod
    def _relaxation_phrase(v: ConstraintViolation) -> str:
        if v.constraint == "max_delivery_days":
            return f"relaxing the delivery deadline by +{int(v.variance)} days (to {int(v.offered)} days)"
        if v.constraint == "min_warranty_months":
            return f"accepting {int(v.variance)} months less warranty (down to {int(v.offered)} months)"
        if v.constraint == "min_ram_gb":
            return f"lowering minimum RAM by {int(v.variance)} GB (to {int(v.offered)} GB)"
        if v.constraint == "min_storage_gb":
            return f"lowering minimum storage by {int(v.variance)} GB (to {int(v.offered)} GB)"
        if v.constraint == "max_budget_per_unit":
            return f"raising the per-unit budget by {format_inr(v.variance)} (to {format_inr(v.offered)})"
        return f"relaxing {v.label}"

    def sensitivity_analysis(
        self, quotations: List[SupplierQuotation], results: List[EvaluationResult]
    ) -> List[TradeOffInsight]:
        """What-if analysis: which disqualified / pending suppliers could unlock savings,
        and exactly how far a constraint must move for them to qualify."""
        by_name = {q.supplier_name: q for q in quotations}
        leader = next((r for r in results if r.rank == 1), None)
        leader_cost = leader.cost_breakdown.total_landed_cost if leader and leader.cost_breakdown else None
        insights: List[TradeOffInsight] = []

        for r in results:
            q = by_name.get(r.supplier_name)
            if q is None or r.cost_breakdown is None:
                continue
            cost = r.cost_breakdown.total_landed_cost
            savings = round(leader_cost - cost, 2) if leader_cost is not None else None

            if r.status == STATUS_NOT_ELIGIBLE:
                violations = self.check_constraints(q)
                missing = self.find_missing_fields(q)
                if not violations:
                    continue
                if len(violations) == 1 and not missing:
                    v = violations[0]
                    phrase = self._relaxation_phrase(v)
                    if leader_cost is None:
                        msg = (f"{phrase[0].upper() + phrase[1:]} qualifies {r.supplier_name} — currently no supplier "
                               f"is eligible, so this would create a compliant option at {format_inr(cost)} landed.")
                        priority = 1
                    elif savings is not None and savings > 0:
                        msg = (f"{phrase[0].upper() + phrase[1:]} qualifies {r.supplier_name}, saving "
                               f"{format_inr(savings)} versus the current recommendation "
                               f"({leader.supplier_name}, {format_inr(leader_cost)}).")
                        priority = 1
                    else:
                        msg = (f"{phrase[0].upper() + phrase[1:]} would qualify {r.supplier_name}, but it would cost "
                               f"{format_inr(abs(savings or 0))} more than {leader.supplier_name}. "
                               f"Keeping the current constraint is recommended.")
                        priority = 3
                    insights.append(TradeOffInsight(
                        supplier_name=r.supplier_name, insight_type="SINGLE_CONSTRAINT",
                        constraints=[v], potential_landed_cost=cost,
                        benchmark_supplier=leader.supplier_name if leader else None,
                        benchmark_landed_cost=leader_cost, potential_savings=savings,
                        message=msg, priority=priority,
                    ))
                else:
                    phrases = "; ".join(self._relaxation_phrase(v) for v in violations)
                    if savings is not None and savings > 0:
                        msg = (f"{r.supplier_name} is {format_inr(savings)} cheaper than {leader.supplier_name} but fails "
                               f"{len(violations)} mandatory constraints ({phrases}). Lowest price ≠ right purchase — "
                               f"not recommended without a formal requirement change.")
                    else:
                        msg = (f"{r.supplier_name} fails {len(violations)} mandatory constraints ({phrases}) and offers "
                               f"no cost advantage. No trade-off worth considering.")
                    insights.append(TradeOffInsight(
                        supplier_name=r.supplier_name, insight_type="MULTI_CONSTRAINT",
                        constraints=violations, potential_landed_cost=cost,
                        benchmark_supplier=leader.supplier_name if leader else None,
                        benchmark_landed_cost=leader_cost, potential_savings=savings,
                        message=msg, priority=3 if (savings or 0) > 0 else 4,
                    ))

            elif r.status == STATUS_NEEDS_CLARIFICATION:
                missing_labels = ", ".join(FIELD_LABELS[f] for f in r.missing_fields)
                if leader_cost is None:
                    msg = (f"{r.supplier_name} ({format_inr(cost)} landed) could become the only eligible supplier "
                           f"once it confirms: {missing_labels}. Prioritise this clarification.")
                    priority = 1
                elif savings is not None and savings > 0:
                    msg = (f"{r.supplier_name} could save up to {format_inr(savings)} versus {leader.supplier_name} "
                           f"if its pending details ({missing_labels}) turn out compliant. Send the clarification e-mail.")
                    priority = 2
                else:
                    msg = (f"{r.supplier_name} is pending clarification ({missing_labels}) but would not beat "
                           f"{leader.supplier_name} on landed cost even if compliant.")
                    priority = 4
                insights.append(TradeOffInsight(
                    supplier_name=r.supplier_name, insight_type="PENDING_CLARIFICATION",
                    potential_landed_cost=cost,
                    benchmark_supplier=leader.supplier_name if leader else None,
                    benchmark_landed_cost=leader_cost, potential_savings=savings,
                    message=msg, priority=priority,
                ))

        # Highlight when the weighted (multi-criteria) leader differs from the cheapest vendor.
        eligible = [r for r in results if r.status == STATUS_ELIGIBLE and r.composite_score is not None]
        if leader and len(eligible) > 1:
            best_weighted = max(eligible, key=lambda r: (r.composite_score, -r.cost_breakdown.total_landed_cost))
            if best_weighted.supplier_name != leader.supplier_name:
                premium = round(best_weighted.cost_breakdown.total_landed_cost - leader_cost, 2)
                insights.append(TradeOffInsight(
                    supplier_name=best_weighted.supplier_name, insight_type="WEIGHTED_LEADER",
                    potential_landed_cost=best_weighted.cost_breakdown.total_landed_cost,
                    benchmark_supplier=leader.supplier_name, benchmark_landed_cost=leader_cost,
                    potential_savings=-premium,
                    message=(f"{best_weighted.supplier_name} has the best weighted score "
                             f"({best_weighted.composite_score:.1f} vs {leader.composite_score:.1f}) thanks to faster "
                             f"delivery / longer warranty, for a premium of {format_inr(premium)} over "
                             f"{leader.supplier_name}. Consider if speed or warranty outweighs cost."),
                    priority=2,
                ))

        insights.sort(key=lambda i: (i.priority, -(i.potential_savings or 0)))
        return insights


# ---------------------------------------------------------------------------
# Natural-language requirement parsing (regex, deterministic)
# ---------------------------------------------------------------------------


def _to_number(text: str) -> float:
    return float(text.replace(",", "").strip())


_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
         "ninety": 90}
_NUM_WORD = "|".join(sorted(list(_UNITS) + list(_TENS), key=len, reverse=True))


def _words_to_int(phrase: str) -> int:
    total, current = 0, 0
    for w in re.split(r"[\s\-]+", phrase.strip()):
        if w in _UNITS:
            current += _UNITS[w]
        elif w in _TENS:
            current += _TENS[w]
        elif w == "hundred":
            current = (current or 1) * 100
        elif w == "thousand":
            total += (current or 1) * 1000
            current = 0
    return total + current


def normalise_casual_text(text: str) -> str:
    """Rewrite everyday phrasing into the canonical forms the regex rules understand, e.g.
    'two dozen' -> '24', 'a terabyte' -> '1 tb', '32 gigs' -> '32 gb', 'three years cover' -> '3 years warranty',
    '85 thousand' -> '85000'. Deterministic and lower-cases the text."""
    t = " " + text.lower().replace("–", "-").replace("—", "-") + " "
    # "70 thousand" -> "70000" first, so a lone scale word is never read as a number on its own
    t = re.sub(r"(\d+(?:\.\d+)?)\s*thousand\b", lambda m: str(int(float(m.group(1)) * 1000)), t)
    # multi-word number phrases ("twenty five", "one hundred and twenty")
    t = re.sub(rf"\b((?:{_NUM_WORD})(?:[\s\-]+(?:{_NUM_WORD}|hundred|thousand|and))*)\b",
               lambda m: str(_words_to_int(re.sub(r"\band\b", " ", m.group(1)))), t)
    t = re.sub(r"\ba\s+(hundred|thousand)\b", lambda m: "100" if m.group(1) == "hundred" else "1000", t)
    # dozens / couples / halves
    t = re.sub(r"\bhalf\s+a\s+dozen\b", "6", t)
    t = re.sub(r"\b(\d+)\s+dozens?\b", lambda m: str(int(m.group(1)) * 12), t)
    t = re.sub(r"\b(?:a|one)\s+dozen\b", "12", t)
    t = re.sub(r"\bdozens?\b", "12", t)
    t = re.sub(r"\ba\s+couple\s+(?:of\s+)?", "2 ", t)
    t = re.sub(r"\bhalf\s+a\s+(?:terabyte|tb)\b", "512 gb", t)
    t = re.sub(r"\b(?:a|an)\s+(terabyte|tb|year|month|week|fortnight)\b", r"1 \1", t)
    # units
    t = re.sub(r"(\d)\s*(?:gigs?|gigabytes?|gbs)\b", r"\1 gb", t)
    t = re.sub(r"(\d)\s*g\b(?=\s*(?:of\s*)?(?:ram|memory))", r"\1 gb", t)
    t = re.sub(r"(\d)\s*(?:terabytes?|tbs)\b", r"\1 tb", t)
    t = re.sub(r"(\d)\s*(?:yrs?)\b", r"\1 years", t)
    t = re.sub(r"(\d)\s*(?:mths?|mos)\b", r"\1 months", t)
    t = re.sub(r"\b1\s+fortnight\b|\bfortnight\b", "2 weeks", t)
    # warranty synonyms
    t = re.sub(r"(\d+\s*(?:years?|months?))\s*(?:of\s*)?(?:on-?site\s*|onsite\s*)?(?:cover|coverage|guarantee|support|amc)\b",
               r"\1 warranty", t)
    return re.sub(r"\s+", " ", t).strip()


def _relative_deadline_days(t: str, today: Optional[datetime] = None) -> Optional[int]:
    """Convert relative deadlines ('end of next week', 'tomorrow', 'end of the month') into calendar days."""
    today = (today or datetime.now()).date()
    wd = today.weekday()  # Monday = 0
    if re.search(r"\b(?:end\s+of|before\s+the\s+end\s+of)\s+next\s+week\b", t):
        return (4 - wd) + 7 if wd <= 4 else (11 - wd)          # Friday of next week
    if re.search(r"\b(?:end\s+of\s+(?:this|the)\s+week|this\s+friday|by\s+friday)\b", t):
        return max(1, 4 - wd) if wd < 4 else 7 - wd + 4
    if re.search(r"\b(?:by|before|within|in|for)\s+next\s+week\b|\bnext\s+week\b", t):
        return 7
    if re.search(r"\bday\s+after\s+tomorrow\b", t):
        return 2
    if re.search(r"\btomorrow\b", t):
        return 1
    if re.search(r"\b(?:asap|urgently|immediately|right\s+away)\b", t):
        return 3
    if re.search(r"\bend\s+of\s+(?:the|this)\s+month\b", t):
        import calendar

        last = calendar.monthrange(today.year, today.month)[1]
        return max(1, last - today.day)
    if re.search(r"\b(?:within|in)\s+1\s+month\b|\bnext\s+month\b", t):
        return 30
    return None


def _parse_amount(num: str, suffix: str) -> float:
    value = _to_number(num)
    suffix = (suffix or "").strip()
    if suffix in {"k", "thousand"}:
        value *= 1_000
    elif suffix.startswith(("la", "l")):
        value *= 100_000
    return value


def parse_natural_language_requirement(text: str, today: Optional[datetime] = None) -> Dict[str, Any]:
    """Extract requirement fields from free text such as
    'We need 20 laptops with 16GB RAM and 512GB SSD within 10 days under 60000' or, casually,
    'about two dozen laptops, 32 gigs, a terabyte, three years cover, by end of next week, 85k each max'.
    Only fields that are actually found are returned."""
    found: Dict[str, Any] = {}
    if not text:
        return found
    t = normalise_casual_text(text)

    m = re.search(r"(\d+)\s*(?:x\s*)?(?:new\s+)?(?:business\s+|office\s+)?(?:laptops?|units?|pcs|pieces|nos\.?|machines?|"
                  r"desktops?|devices?|notebooks?|computers?|systems?)\b", t) \
        or re.search(r"\bfor\s+(?:our\s+|the\s+|about\s+|around\s+)?(\d+)\s*(?:new\s+)?(?:people|employees|staff|users|"
                     r"engineers|developers|designers|members|interns|hires|joiners|students)\b", t)
    if m:
        found["requested_quantity"] = int(m.group(1))

    m = re.search(r"(\d+)\s*gb\s*(?:of\s*)?(?:ddr\d\s*)?(?:ram|memory)", t) \
        or re.search(r"(?:ram|memory)\s*(?:of\s*|should\s+be\s*|is\s*)?[:\-]?\s*(\d+)\s*gb", t)
    if m:
        found["min_ram_gb"] = int(m.group(1))

    m = re.search(r"(\d+(?:\.\d+)?)\s*(gb|tb)\s*(?:of\s*)?(?:nvme\s*|ssd\s*)?(?:ssd|storage|hdd|disk|drive|space)", t) \
        or re.search(r"(?:ssd|storage|disk|drive)\s*(?:of\s*|should\s+be\s*|is\s*)?[:\-]?\s*(\d+(?:\.\d+)?)\s*(gb|tb)", t)
    if not m:
        # a bare "1 tb" / "512 gb" that is not the RAM figure
        for cand in re.finditer(r"(\d+(?:\.\d+)?)\s*(gb|tb)\b(?!\s*(?:of\s*)?(?:ddr\d\s*)?(?:ram|memory))", t):
            size = float(cand.group(1)) * (1024 if cand.group(2) == "tb" else 1)
            if size >= 128 and size != found.get("min_ram_gb"):
                m = cand
                break
    if m:
        value = float(m.group(1))
        found["min_storage_gb"] = int(value * 1024) if m.group(2) == "tb" else int(value)

    m = re.search(r"(\d+)\s*(months?|years?)\s*(?:of\s*)?(?:on-?site\s*|comprehensive\s*)?warranty", t) \
        or re.search(r"warranty\s*(?:of\s*|for\s*|should\s+be\s*|is\s*)?[:\-]?\s*(\d+)\s*(months?|years?)", t)
    if m:
        n = int(m.group(1))
        found["min_warranty_months"] = n * 12 if m.group(2).startswith("y") else n

    m = re.search(r"(?:within|in|by|under|max(?:imum)?\s*of|before|no\s+later\s+than)\s*(\d+)\s*(days?|weeks?)", t) \
        or re.search(r"(\d+)\s*(days?|weeks?)\s*(?:delivery|lead\s*time|turnaround)", t) \
        or re.search(r"(?:delivery|lead\s*time|deliver(?:ed)?)\s*(?:of|in|within)?\s*[:\-]?\s*(\d+)\s*(days?|weeks?)", t)
    if m:
        n = int(m.group(1))
        found["max_delivery_days"] = n * 7 if m.group(2).startswith("week") else n
    else:
        rel = _relative_deadline_days(t, today)
        if rel is not None:
            found["max_delivery_days"] = rel

    money = r"(?:₹|rs\.?|inr)?\s*([\d,]+(?:\.\d+)?)\s*(k|lakhs?|lacs?|l)?\b"
    not_other_unit = r"(?!\s*(?:days?|weeks?|gb|tb|months?|years?|laptops?|units?))"
    m = re.search(
        r"(?:under|below|budget(?:\s*(?:is|of|:|=|around|about|capped\s+at))?|max(?:imum)?(?:\s*budget)?(?:\s*of)?|"
        r"less\s*than|up\s*to|upto|within\s*budget\s*of|at\s*most|not\s+more\s+than|no\s+more\s+than|around|about|"
        r"spend(?:\s*up\s*to)?|cap(?:ped)?\s*at)\s*[:\-]?\s*" + money + not_other_unit, t)
    if not m:
        m = re.search(money + r"\s*(?:each|per\s*(?:unit|laptop|piece|pc|device|machine)|/\s*(?:unit|laptop|pc))", t) \
            or re.search(r"(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)\s*(k|lakhs?|lacs?|l)?\b", t)
    if m:
        value = _parse_amount(m.group(1), m.group(2) or "")
        if value >= 1000:  # guard against matching small numbers such as "under 10 days"
            found["max_budget_per_unit"] = float(value)
    return found


# ---------------------------------------------------------------------------
# Self-check when executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    requirement = PurchaseRequirement(
        min_ram_gb=16, min_storage_gb=512, min_warranty_months=24, max_delivery_days=10,
        requested_quantity=20, max_budget_per_unit=60000,
    )
    quotes = [
        SupplierQuotation(supplier_name="Apex Solutions", unit_price=52000, ram_gb=16, storage_gb=512,
                          warranty_months=24, delivery_days=7, shipping_cost_total=2000),
        SupplierQuotation(supplier_name="Beta Tech Enterprises", unit_price=41000, ram_gb=8, storage_gb=256,
                          warranty_months=12, delivery_days=10, shipping_cost_total=1500),
        SupplierQuotation(supplier_name="Gamma Systems", unit_price=48000, ram_gb=16, storage_gb=512,
                          shipping_cost_total=1800),
    ]
    engine = ProcureAIDecisionEngine(requirement)
    res = engine.evaluate(quotes)
    for r in res:
        print(r.supplier_name, r.status, r.rank, r.composite_score,
              r.cost_breakdown.total_landed_cost if r.cost_breakdown else None)
        for reason in r.reasons:
            print("   -", reason)
    for ins in engine.sensitivity_analysis(quotes, res):
        print("INSIGHT:", ins.message)
    print(parse_natural_language_requirement(
        "We need 20 laptops with 16GB RAM and 512GB SSD within 10 days under 60000"))
