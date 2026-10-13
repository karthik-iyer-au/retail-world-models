"""Audit of plan-lift explanations (paper 2, docs/PAPER2_PLAN.md).

The audit reads fitted models; it never changes their settings. Each function
returns plain numbers or frames so results can be written to one record per
model version.
"""
from rwm.audit.lifts import (
    construction_checks,
    lever_contributions,
    lift_vs_actual,
    off_plan,
    price_monotonicity,
    randomised_copies,
    rank_agreement,
    regular_prices,
)

__all__ = [
    "construction_checks", "lever_contributions", "lift_vs_actual", "off_plan", "price_monotonicity",
    "randomised_copies", "rank_agreement", "regular_prices",
]
