import numpy as np
import numpy.typing as npt
from scipy import stats

from latan.statistics.bootstrap import BootstrapArray

NORMALITY_P_THRESHOLD = 2.0 * stats.norm.sf(2.0)


def p_value_colour(p_value: float) -> tuple[float, str]:
    significance = normality_significance(p_value)
    if significance < 2:
        colour = "#2e7d32"
    elif significance < 3:
        colour = "#ed6c02"
    else:
        colour = "#c62828"
    return significance, colour


def normality_significance(p_value: float) -> float:
    return float(stats.norm.isf(p_value / 2) if p_value > 0 else np.inf)


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
    error: float,
    attributes: str = "",
    annotation: str = "",
) -> str:
    return f"<td{attributes}>{error:.4g}{annotation}</td>"


def bootstrap_error_text(
    value: float,
    error: float,
) -> str:
    return f"{value:.4g} ± {error:.4g}"


def asymmetric_error_text(value: float, lower: float, upper: float, label: str = "") -> str:
    prefix = f"err_{label}" if label else "err"
    return f"{prefix} = +{upper - value:.4g} −{value - lower:.4g}"


def non_gaussian_html(
    p_values: npt.ArrayLike,
    errors: tuple[tuple[str, float, float], ...] = (),
) -> tuple[str, str]:
    p_values = np.asarray(p_values)
    flagged = p_values[p_values < NORMALITY_P_THRESHOLD]
    if flagged.size == 0:
        return "", ""
    _, colour = p_value_colour(float(flagged.min()))
    items = "&nbsp;&amp;&nbsp;".join(
        f'<span class="latan-ng-item">{label} =&nbsp;'
        f'<small style="display:inline-block;vertical-align:middle;'
        'line-height:0.85;text-align:left">'
        f'<span style="display:block">+{upper:.4g}</span>'
        f'<span style="display:block">−{lower:.4g}</span>'
        "</small></span>"
        for label, upper, lower in errors
    )
    annotation = f'<span class="latan-ng" style="--latan-ng-colour:{colour}">'
    annotation += f"<b>NN</b> ({items})</span>"
    return ' class="latan-ng-holder"', annotation


def non_gaussian_text(
    p_values: npt.ArrayLike,
    *,
    sigma: bool = False,
    errors: tuple[str, ...] = (),
) -> str:
    p_values = np.asarray(p_values)
    flagged = p_values[p_values < NORMALITY_P_THRESHOLD]
    if flagged.size == 0:
        return ""
    if sigma:
        return f" NN ({normality_significance(float(flagged.min())):.2g}σ)"
    if errors:
        return f" NN ({' & '.join(errors)})"
    return " NN"


def normality_css() -> str:
    return """
        <style>
          td.latan-ng-holder { position:relative; overflow:visible; }
          span.latan-ng {
            position:absolute;
            left:calc(100% + 0.35em);
            top:50%;
            transform:translateY(-50%);
            color:var(--latan-ng-colour);
            display:inline-flex;
            align-items:center;
            white-space:nowrap;
          }
          span.latan-ng-item {
            display:inline-flex;
            align-items:center;
          }
          span.latan-ng b { margin-right:0.25em; }
        </style>
    """
