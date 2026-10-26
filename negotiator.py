"""
negotiator.py — Agentic multi-round negotiation (Feature #1)
============================================================

Division of labour:
  * Strategy   — deterministic. The engine computes what we can credibly ask each supplier
                 for, using the competing quotes as leverage.
  * Messages   — Claude drafts the buyer's e-mail (offline: template).
  * Supplier   — Claude role-plays the supplier with a hidden floor price and persona
                 (offline: a deterministic concession model). Its counter-offer is clamped to
                 the hidden floor and re-validated by the decision engine.
  * Guardrails — the agent never accepts a deal and never asks a supplier to break a
                 mandatory constraint; a human decides whether to apply the negotiated terms.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

import llm
from engine import (
    STATUS_ELIGIBLE,
    EvaluationResult,
    ProcureAIDecisionEngine,
    PurchaseRequirement,
    SupplierQuotation,
    format_inr,
)

MAX_ROUNDS = 3

PERSONAS: Dict[str, Dict[str, float]] = {
    # max price concession, delivery days it can shave, warranty months it can add, free-shipping willingness
    "Stubborn": {"price_flex": 0.025, "delivery_flex": 1, "warranty_flex": 0, "ship_waive": 0.0, "pace": 0.35},
    "Balanced": {"price_flex": 0.055, "delivery_flex": 3, "warranty_flex": 6, "ship_waive": 0.5, "pace": 0.55},
    "Eager to win": {"price_flex": 0.09, "delivery_flex": 5, "warranty_flex": 12, "ship_waive": 1.0, "pace": 0.75},
}


class NegotiationAsk(BaseModel):
    target_unit_price: Optional[float] = None
    target_delivery_days: Optional[int] = None
    target_warranty_months: Optional[int] = None
    waive_shipping: bool = False
    leverage: List[str] = Field(default_factory=list)


class Offer(BaseModel):
    unit_price: float
    delivery_days: int
    warranty_months: int
    shipping_cost_total: float


class NegotiationRound(BaseModel):
    round_no: int
    buyer_subject: str
    buyer_message: str
    supplier_message: str
    offer: Offer
    landed_cost: float
    status_after: str


class NegotiationSession(BaseModel):
    supplier_name: str
    persona: str
    original: Offer
    original_landed_cost: float
    ask: NegotiationAsk
    rounds: List[NegotiationRound] = Field(default_factory=list)
    closed: bool = False
    applied: bool = False
    mode: str = "offline"

    @property
    def best_offer(self) -> Offer:
        return self.rounds[-1].offer if self.rounds else self.original

    @property
    def savings(self) -> float:
        if not self.rounds:
            return 0.0
        return round(self.original_landed_cost - self.rounds[-1].landed_cost, 2)


# ---------------------------------------------------------------------------
# Strategy (deterministic)
# ---------------------------------------------------------------------------


def negotiable(q: SupplierQuotation) -> bool:
    return all(v is not None for v in (q.unit_price, q.delivery_days, q.warranty_months))


def plan_ask(target: SupplierQuotation, quotations: List[SupplierQuotation], results: List[EvaluationResult],
             requirement: PurchaseRequirement) -> NegotiationAsk:
    ask = NegotiationAsk()
    by_name = {r.supplier_name: r for r in results}
    eligible_rivals = [q for q in quotations if q.supplier_name != target.supplier_name
                       and by_name.get(q.supplier_name) and by_name[q.supplier_name].status == STATUS_ELIGIBLE]
    priced_rivals = [q for q in quotations if q.supplier_name != target.supplier_name and q.unit_price]

    if eligible_rivals:
        best = min(eligible_rivals, key=lambda q: q.unit_price)
        if best.unit_price < target.unit_price:
            ask.target_unit_price = round(best.unit_price * 0.99, -2)
            ask.leverage.append(f"A compliant competitor quotes {format_inr(best.unit_price)} per unit with "
                                f"{best.warranty_months} months warranty and {best.delivery_days}-day delivery.")
        else:
            ask.target_unit_price = round(target.unit_price * 0.95, -2)
            ask.leverage.append(f"You are currently the lowest compliant bid; a 5% volume discount on "
                                f"{requirement.requested_quantity} units secures the order.")
    elif priced_rivals:
        low = min(priced_rivals, key=lambda q: q.unit_price)
        ask.target_unit_price = round(max(low.unit_price, target.unit_price * 0.93), -2)
        ask.leverage.append(f"Other suppliers have quoted from {format_inr(low.unit_price)} per unit.")
    else:
        ask.target_unit_price = round(target.unit_price * 0.95, -2)
        ask.leverage.append("A volume discount is expected for an order of this size.")

    if target.delivery_days > requirement.max_delivery_days:
        ask.target_delivery_days = requirement.max_delivery_days
        ask.leverage.append(f"Delivery within {requirement.max_delivery_days} days is a mandatory requirement.")
    if target.warranty_months < requirement.min_warranty_months:
        ask.target_warranty_months = requirement.min_warranty_months
        ask.leverage.append(f"A minimum of {requirement.min_warranty_months} months warranty is mandatory.")
    if target.shipping_cost_total > 0:
        ask.waive_shipping = True
    return ask


# ---------------------------------------------------------------------------
# Supplier simulation
# ---------------------------------------------------------------------------


def _floor(original: Offer, persona: str) -> Offer:
    p = PERSONAS[persona]
    return Offer(
        unit_price=round(original.unit_price * (1 - p["price_flex"]), 2),
        delivery_days=max(1, int(original.delivery_days - p["delivery_flex"])),
        warranty_months=int(original.warranty_months + p["warranty_flex"]),
        shipping_cost_total=round(original.shipping_cost_total * (1 - p["ship_waive"]), 2),
    )


def _clamp(offer: Offer, original: Offer, floor: Offer) -> Offer:
    """A simulated supplier may never go beyond its hidden limits, nor worsen its own quote."""
    return Offer(
        unit_price=min(original.unit_price, max(floor.unit_price, offer.unit_price)),
        delivery_days=max(floor.delivery_days, min(original.delivery_days, offer.delivery_days)),
        warranty_months=min(floor.warranty_months, max(original.warranty_months, offer.warranty_months)),
        shipping_cost_total=min(original.shipping_cost_total, max(floor.shipping_cost_total, offer.shipping_cost_total)),
    )


def _offline_counter(session: NegotiationSession, round_no: int) -> tuple[Offer, str]:
    original, current = session.original, session.best_offer
    floor = _floor(original, session.persona)
    pace = PERSONAS[session.persona]["pace"]
    ask = session.ask
    final = round_no >= MAX_ROUNDS

    price_target = max(floor.unit_price, ask.target_unit_price or current.unit_price)
    new_price = current.unit_price - (current.unit_price - price_target) * (1.0 if final else pace)
    new_price = round(max(floor.unit_price, new_price), -1)

    new_delivery = current.delivery_days
    if ask.target_delivery_days is not None:
        new_delivery = max(floor.delivery_days, ask.target_delivery_days if final or pace > 0.5 else
                           current.delivery_days - 1)
    new_warranty = current.warranty_months
    if ask.target_warranty_months is not None:
        new_warranty = min(floor.warranty_months, ask.target_warranty_months)
    new_ship = floor.shipping_cost_total if (ask.waive_shipping and (final or pace >= 0.55)) else current.shipping_cost_total

    offer = _clamp(Offer(unit_price=new_price, delivery_days=int(new_delivery), warranty_months=int(new_warranty),
                         shipping_cost_total=new_ship), original, floor)

    changes = []
    if offer.unit_price < current.unit_price:
        changes.append(f"revise the unit price to {format_inr(offer.unit_price)}")
    if offer.delivery_days < current.delivery_days:
        changes.append(f"deliver in {offer.delivery_days} days")
    if offer.warranty_months > current.warranty_months:
        changes.append(f"extend warranty to {offer.warranty_months} months")
    if offer.shipping_cost_total < current.shipping_cost_total:
        changes.append("waive freight" if offer.shipping_cost_total == 0 else
                       f"reduce freight to {format_inr(offer.shipping_cost_total)}")
    tone = {"Stubborn": "Our margins are already tight, but", "Balanced": "We value this opportunity and",
            "Eager to win": "We are keen to win this order and"}[session.persona]
    if changes:
        body = f"{tone} we can {', '.join(changes)}."
    else:
        body = "We regret that our previous offer is already our best and final position."
    if final:
        body += " This is our final offer."
    if ask.target_delivery_days is not None and offer.delivery_days > ask.target_delivery_days:
        body += f" Unfortunately {offer.delivery_days} days is the fastest we can commit to."
    if ask.target_warranty_months is not None and offer.warranty_months < ask.target_warranty_months:
        body += f" We cannot extend warranty beyond {offer.warranty_months} months."
    message = f"Dear Procurement Team,\n\n{body}\n\nRegards,\nSales Desk, {session.supplier_name}"
    return offer, message


class BuyerDraft(BaseModel):
    subject: str
    body: str


class SupplierReply(BaseModel):
    message: str = Field(description="The supplier's reply e-mail body")
    unit_price: float
    delivery_days: int
    warranty_months: int
    shipping_cost_total: float


def _offline_buyer_message(session: NegotiationSession, round_no: int, requirement: PurchaseRequirement) -> BuyerDraft:
    ask, cur = session.ask, session.best_offer
    points = []
    if ask.target_unit_price and cur.unit_price > ask.target_unit_price:
        points.append(f"a unit price of {format_inr(ask.target_unit_price)} (currently {format_inr(cur.unit_price)})")
    if ask.target_delivery_days and cur.delivery_days > ask.target_delivery_days:
        points.append(f"delivery within {ask.target_delivery_days} days (currently {cur.delivery_days})")
    if ask.target_warranty_months and cur.warranty_months < ask.target_warranty_months:
        points.append(f"{ask.target_warranty_months} months warranty (currently {cur.warranty_months})")
    if ask.waive_shipping and cur.shipping_cost_total > 0:
        points.append(f"waiver of the {format_inr(cur.shipping_cost_total)} freight charge")
    opener = {1: "Thank you for your quotation.", 2: "Thank you for your revised offer.",
              3: "We are close to finalising the award."}.get(round_no, "Following up on our discussion.")
    leverage = " ".join(session.ask.leverage)
    ask_text = "; ".join(points) if points else "your best and final commercial terms"
    body = (f"Dear {session.supplier_name} Team,\n\n{opener} For our order of {requirement.requested_quantity} laptops "
            f"we request: {ask_text}.\n\n{leverage}\n\nPlease share your revised offer at the earliest.\n\n"
            f"Regards,\nProcurement Desk — ProcureAI")
    return BuyerDraft(subject=f"Round {round_no}: revised commercial offer — {session.supplier_name}", body=body)


def _ai_buyer_message(session: NegotiationSession, round_no: int, requirement: PurchaseRequirement) -> Optional[BuyerDraft]:
    history = "\n\n".join(f"ROUND {r.round_no}\nUS: {r.buyer_message}\nSUPPLIER: {r.supplier_message}" for r in session.rounds)
    system = ("You are a professional procurement negotiator for an Indian company. Write concise, courteous, firm "
              "negotiation e-mails. Use only the facts and targets provided. Never accept a deal, never promise an "
              "award, and never ask the supplier to compromise mandatory specifications.")
    prompt = (f"Round {round_no} of {MAX_ROUNDS} with {session.supplier_name}.\nQuantity: {requirement.requested_quantity}\n"
              f"Their current offer: {session.best_offer.model_dump()}\nOur targets: {session.ask.model_dump()}\n"
              f"Leverage facts: {session.ask.leverage}\nPrevious rounds:\n{history or 'none'}\n\n"
              "Write the next e-mail (subject + body, under 150 words).")
    return llm.structured_call(system, prompt, BuyerDraft, effort="low", max_tokens=4000)


def _ai_supplier_reply(session: NegotiationSession, buyer: BuyerDraft, round_no: int) -> Optional[SupplierReply]:
    floor = _floor(session.original, session.persona)
    system = (f"You role-play the sales manager of {session.supplier_name}, an Indian laptop supplier, in a simulation. "
              f"Persona: {session.persona}. Your original offer: {session.original.model_dump()}. "
              f"HIDDEN LIMITS you must never cross: unit price not below {floor.unit_price}, delivery not faster than "
              f"{floor.delivery_days} days, warranty not above {floor.warranty_months} months, freight not below "
              f"{floor.shipping_cost_total}. Concede gradually; round {round_no} of {MAX_ROUNDS}. "
              "Reply with a short e-mail and the exact numbers of your revised offer.")
    return llm.structured_call(system, f"Buyer e-mail:\nSubject: {buyer.subject}\n\n{buyer.body}",
                               SupplierReply, effort="low", max_tokens=4000)


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


def default_persona(supplier_name: str) -> str:
    keys = list(PERSONAS)
    return keys[int(hashlib.md5(supplier_name.encode()).hexdigest(), 16) % len(keys)]


def start_session(target: SupplierQuotation, quotations: List[SupplierQuotation], results: List[EvaluationResult],
                  requirement: PurchaseRequirement, persona: str) -> NegotiationSession:
    engine = ProcureAIDecisionEngine(requirement)
    original = Offer(unit_price=target.unit_price, delivery_days=target.delivery_days,
                     warranty_months=target.warranty_months, shipping_cost_total=target.shipping_cost_total)
    cb = engine.calculate_landed_cost(target)
    return NegotiationSession(
        supplier_name=target.supplier_name, persona=persona, original=original,
        original_landed_cost=cb.total_landed_cost if cb else 0.0,
        ask=plan_ask(target, quotations, results, requirement),
    )


def run_round(session: NegotiationSession, base_quote: SupplierQuotation, requirement: PurchaseRequirement,
              use_ai: bool) -> NegotiationSession:
    if session.closed:
        return session
    round_no = len(session.rounds) + 1
    buyer = _ai_buyer_message(session, round_no, requirement) if use_ai else None
    mode = "ai" if buyer else "offline"
    buyer = buyer or _offline_buyer_message(session, round_no, requirement)

    offer, message = None, None
    if use_ai and mode == "ai":
        reply = _ai_supplier_reply(session, buyer, round_no)
        if reply:
            floor = _floor(session.original, session.persona)
            offer = _clamp(Offer(unit_price=reply.unit_price, delivery_days=reply.delivery_days,
                                 warranty_months=reply.warranty_months, shipping_cost_total=reply.shipping_cost_total),
                           session.original, floor)
            message = reply.message
    if offer is None:
        offer, message = _offline_counter(session, round_no)
        mode = "offline" if mode == "offline" else "ai+offline"

    engine = ProcureAIDecisionEngine(requirement)
    updated = base_quote.model_copy(update=offer.model_dump())
    res = engine.evaluate_single(updated)
    session.rounds.append(NegotiationRound(
        round_no=round_no, buyer_subject=buyer.subject, buyer_message=buyer.body, supplier_message=message,
        offer=offer, landed_cost=res.cost_breakdown.total_landed_cost if res.cost_breakdown else 0.0,
        status_after=res.status,
    ))
    session.mode = mode
    no_progress = len(session.rounds) >= 2 and session.rounds[-1].offer == session.rounds[-2].offer
    if round_no >= MAX_ROUNDS or no_progress:
        session.closed = True
    return session
