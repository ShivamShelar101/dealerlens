"""Deal score: is this acquisition price a good buy given the estimated market value?"""
import math

Z90 = 1.2816  # standard normal 90th percentile


def deal_score(value, low, high, price, recon=0.0, target_margin=0.08):
    """Treat [low, high] as the 10th-90th percentile of a log-normal market value.

    Returns expected margin %, probability market value exceeds total cost, and a label.
    """
    cost = float(price) + float(recon)
    margin = (value - cost) / value
    sigma = max((math.log(high) - math.log(low)) / (2 * Z90), 1e-6)
    z = (math.log(value) - math.log(cost)) / sigma
    p_above = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    if margin >= target_margin and p_above >= 0.75:
        label = "Buy"
    elif cost > value:
        label = "Pass"
    else:
        label = "Negotiate"
    target_price = value * (1 - target_margin) - recon
    return {"total_cost": round(cost), "expected_margin_pct": round(margin * 100, 1),
            "prob_value_above_cost": round(p_above, 3), "label": label,
            "max_price_for_target_margin": round(target_price)}
