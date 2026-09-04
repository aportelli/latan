from latan.physics.laplace_filter.filter import lfilter, lfilter_factor
from latan.physics.laplace_filter.spectrum import LaplaceFilterEnergies
from latan.statistics.bootstrap import BootstrapArray


def filter_excited(
    correlator: BootstrapArray,
    energies: LaplaceFilterEnergies[BootstrapArray],
    n_excited_states: int,
) -> BootstrapArray:
    lambs = energies.lambdas.central
    e = energies.energies
    states = slice(1, n_excited_states + 1)
    correlator_f = lfilter(correlator, lambs[states], dim=-1)
    factor = lfilter_factor(lambs[states], BootstrapArray(e[:, 0:1]))
    return BootstrapArray(correlator_f / factor)
