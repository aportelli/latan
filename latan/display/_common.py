import numpy as np
from scipy import stats


def p_value_colour(p_value: float) -> tuple[float, str]:
    significance = float(stats.norm.isf(p_value / 2) if p_value > 0 else np.inf)
    if significance < 2:
        colour = "#2e7d32"
    elif significance < 3:
        colour = "#ed6c02"
    else:
        colour = "#c62828"
    return significance, colour
