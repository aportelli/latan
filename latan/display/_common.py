import numpy as np
import numpy.typing as npt
from scipy import stats

from latan.statistics.bootstrap import BootstrapArray

NORMALITY_P_THRESHOLD = 2.0 * stats.norm.sf(2.0)


def p_value_colour(p_value: float) -> tuple[float, str]:
    significance = float(stats.norm.isf(p_value / 2) if p_value > 0 else np.inf)
    if significance < 2:
        colour = "#2e7d32"
    elif significance < 3:
        colour = "#ed6c02"
    else:
        colour = "#c62828"
    return significance, colour


def bootstrap_normality(
    data: BootstrapArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from latan.statistics.normality import normality_test

    test = normality_test(data)
    lower, upper = np.quantile(
        data.samples, (stats.norm.cdf(-1.0), stats.norm.cdf(1.0)), axis=0
    )
    p_value = np.asarray(test.p_value)
    non_gaussian = p_value < NORMALITY_P_THRESHOLD
    return lower, upper, non_gaussian, p_value


def bootstrap_error_html(
    value: float,
    error: float,
    lower: float,
    upper: float,
    non_gaussian: bool,
    attributes: str = "",
) -> str:
    if not non_gaussian:
        return f"<td{attributes}>{error:.4g}</td>"
    return (
        f'<td{attributes}><small style="display:inline-block;vertical-align:middle;'
        'line-height:0.85;text-align:left">'
        f'<span style="display:block">+{upper - value:.4g}</span>'
        f'<span style="display:block">−{value - lower:.4g}</span>'
        "</small></td>"
    )


def non_gaussian_attributes(p_values: npt.ArrayLike) -> str:
    p_values = np.asarray(p_values)
    flagged = p_values[p_values < NORMALITY_P_THRESHOLD]
    if flagged.size == 0:
        return ""
    _, colour = p_value_colour(float(flagged.min()))
    return f' class="latan-ng" style="--latan-ng-colour:{colour}"'


def normality_css() -> str:
    return """
        <style>
          td.latan-ng { position:relative; overflow:visible; }
          td.latan-ng::after {
            content:"NN";
            position:absolute;
            left:calc(100% + 0.35em);
            top:50%;
            transform:translateY(-50%);
            color:var(--latan-ng-colour);
            font-weight:bold;
            white-space:nowrap;
          }
        </style>
    """
