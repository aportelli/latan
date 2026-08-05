import glob
import hashlib
import os
import tarfile
import threading
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import cast
from urllib.request import urlretrieve

import h5py
import numpy as np
import numpy.typing as npt
from numpy.lib import recfunctions
from tqdm.auto import tqdm

from latan.datasets.arXiv_2406_19193.locations import (
    ARCHIVES,
    IRREP_ARCHIVES,
    RECORD_URL,
    Archive,
)

_progress_lock = threading.Lock()


class _Progress:
    def __init__(self, enabled: bool, total: int | None) -> None:
        self.enabled = enabled
        self.current: dict[str, int] = {}
        # tqdm.auto automatically selects the Jupyter renderer when available.
        self.bar = tqdm(
            total=total,
            desc="Downloading data",
            unit="B",
            unit_scale=True,
            disable=not enabled,
        )

    def report(self, name: str, block: int, block_size: int, total: int) -> None:
        if not self.enabled or total <= 0:
            return
        current = min(block * block_size, total)
        with _progress_lock:
            previous = self.current.get(name, 0)
            self.current[name] = current
            self.bar.update(current - previous)

    def close(self) -> None:
        self.bar.close()


def _download_archive(destination: str, info: Archive, progress: _Progress) -> str:
    archive = os.path.join(destination, info.name)
    temporary = f"{archive}.part"
    if os.path.exists(temporary):
        os.remove(temporary)
    urlretrieve(
        f"{RECORD_URL}/{info.name}?download=1",
        temporary,
        reporthook=lambda block, block_size, total: progress.report(
            info.name, block, block_size, total
        ),
    )
    if _md5(temporary) != info.md5:
        os.remove(temporary)
        raise RuntimeError(f"checksum mismatch after downloading '{info.name}'")
    os.replace(temporary, archive)
    return archive


def _md5(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract(archive: str, destination: str) -> None:
    destination = os.path.realpath(destination)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = os.path.realpath(os.path.join(destination, member.name))
            if os.path.commonpath((destination, target)) != destination:
                raise RuntimeError(
                    f"unsafe path in archive '{os.path.basename(archive)}': {member.name}"
                )
        tar.extractall(destination, filter="data")


def download(
    path: str,
    archives: Iterable[str],
    *,
    extract: bool = True,
    workers: int = 1,
    progress: bool = True,
) -> dict[str, str]:
    """Download named release archives and optionally extract them into `path`.

    An existing destination is left untouched and causes the download to be
    skipped. Distinct archives can transfer concurrently; extraction remains
    sequential.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    destination = path
    if os.path.exists(destination):
        print(f"Download skipped: destination already exists: '{destination}'")
        return {}
    os.makedirs(destination)
    names = tuple(archives)
    if len(set(names)) != len(names):
        raise ValueError("archives must not contain duplicates")
    for name in names:
        if name not in ARCHIVES:
            choices = ", ".join(ARCHIVES)
            raise ValueError(f"unknown archive '{name}' (choose from: {choices})")

    sizes = [ARCHIVES[name].size for name in names]
    known_sizes = [size for size in sizes if size is not None]
    total = sum(known_sizes) if len(known_sizes) == len(sizes) else None
    reporter = _Progress(progress, total)
    try:
        with ThreadPoolExecutor(max_workers=min(workers, len(names) or 1)) as executor:
            downloaded = list(
                executor.map(
                    lambda name: _download_archive(
                        destination, ARCHIVES[name], reporter
                    ),
                    names,
                )
            )
    finally:
        reporter.close()
    result = dict(zip(names, downloaded, strict=True))
    if extract:
        for archive in result.values():
            _extract(archive, destination)
    return result


def _momentum(momentum: Sequence[int]) -> str:
    return "_".join(str(item) for item in momentum)


def _neg(momentum: Sequence[int]) -> tuple[int, ...]:
    return tuple(-item for item in momentum)  # type: ignore[return-value]


def _filename(fields: Sequence[tuple[str, str, Sequence[int]]], trajectory: int) -> str:
    prefix = "__".join(
        f"{flavour}_{gamma}_p{_momentum(momentum)}"
        for flavour, gamma, momentum in fields
    )
    return f"{prefix}.{trajectory}.h5"


def _contraction_file(
    path: str,
    corrtype: str,
    trajectory: int,
    fields: Sequence[tuple[str, str, Sequence[int]]],
    momentum_squared: int | None = None,
) -> str:
    directory = os.path.join(path, "K-Pi", corrtype)
    if momentum_squared is not None:
        directory = os.path.join(directory, f"P{momentum_squared}")
    filename = os.path.join(
        directory, f"data.{trajectory}", _filename(fields, trajectory)
    )
    if not os.path.isfile(filename):
        raise FileNotFoundError(
            f"missing contraction file '{filename}'; download and extract the required archive"
        )
    return filename


def load_diagram(filename: str, diagram: str) -> npt.NDArray:
    """Read one diagram and average its translated source times.

    The result has shape `(nt,)`, indexed by source-sink separation.
    """
    with h5py.File(filename, "r") as file:
        correlators = file["DistillationContraction/Correlators"]
        if not isinstance(correlators, h5py.Group):
            raise TypeError(f"correlator path in '{filename}' is not a group")
        if diagram not in correlators:
            raise RuntimeError(f"no diagram '{diagram}' in file '{filename}'")
        sources = correlators[diagram]
        if not isinstance(sources, h5py.Group):
            raise TypeError(f"diagram '{diagram}' in '{filename}' is not a group")
        translated: list[npt.NDArray] = []
        for source in sorted(sources.keys(), key=int):
            source_data = sources[source]
            if isinstance(source_data, h5py.Group):
                dataset = source_data["double_array"]
            elif isinstance(source_data, h5py.Dataset):
                dataset = source_data
            else:
                raise TypeError(f"source '{source}' in '{filename}' is not readable")
            if not isinstance(dataset, h5py.Dataset):
                raise TypeError(f"source '{source}' in '{filename}' is not a dataset")
            array = np.asarray(dataset)
            if array.dtype.names == ("re", "im") and array.ndim == 1:
                components = recfunctions.structured_to_unstructured(
                    cast(npt.NDArray[np.void], array)
                )
                data = components[:, 0] + 1j * components[:, 1]
            elif array.ndim == 2 and array.shape[1] == 2:
                data = array[:, 0] + 1j * array[:, 1]
            else:
                raise ValueError(
                    f"'{filename}' has incompatible shape and dtype: {array.shape}, {array.dtype}"
                )
            translated.append(np.roll(data, -int(source)))
    if not translated:
        raise RuntimeError(f"diagram '{diagram}' in '{filename}' has no source times")
    return np.mean(translated, axis=0)


def trajectories(path: str) -> tuple[int, ...]:
    """Return sorted trajectory numbers available for zero-momentum Kpi V_V data."""
    directory = os.path.join(path, "K-Pi", "V_V", "P0")
    values = []
    for item in glob.glob(os.path.join(directory, "data.*")):
        try:
            values.append(int(os.path.basename(item).removeprefix("data.")))
        except ValueError:
            continue
    if not values:
        archive = IRREP_ARCHIVES["K-Pi"]["T1u[000]"][-1]
        raise FileNotFoundError(
            f"no trajectories found under '{directory}'; extract {archive}"
        )
    return tuple(sorted(values))


def _shell(norm_squared: int) -> tuple[tuple[int, int, int], ...]:
    extent = int(np.sqrt(norm_squared))
    momenta = [
        (x, y, z)
        for x in range(-extent, extent + 1)
        for y in range(-extent, extent + 1)
        for z in range(-extent, extent + 1)
        if x * x + y * y + z * z == norm_squared
    ]
    return tuple(momenta)


def _t1u_weights(norm_squared: int, axis: int) -> dict[tuple[int, int, int], float]:
    """Weights for one Cartesian row of the T1u cubic harmonic."""
    # Dudek et al., Phys. Rev. D 82, 034508 (2010), App. A, gives the
    # J=1 -> T1 subduction. On a complete |d|^2 orbit, its real Cartesian
    # rows are d_i / |d|; normalize them so that sum_d |w(d)|^2 = 1.
    values = {
        momentum: momentum[axis] / np.sqrt(norm_squared)
        for momentum in _shell(norm_squared)
    }
    normalization = np.sqrt(sum(value * value for value in values.values()))
    return {
        momentum: value / normalization for momentum, value in values.items() if value
    }


def _vv(path: str, trajectory: int, gamma: str) -> npt.NDArray:
    zero = (0, 0, 0)
    file = _contraction_file(
        path,
        "V_V",
        trajectory,
        (("rpl", gamma, zero), ("rps", gamma, zero)),
        0,
    )
    # Appendix C, Fig. 12.
    return -load_diagram(file, "connected")


def _kpi_v(
    path: str, trajectory: int, momentum: Sequence[int], gamma: str
) -> npt.NDArray:
    zero = (0, 0, 0)
    file = _contraction_file(
        path,
        "Kpi_V",
        trajectory,
        (
            ("rpl", "Gamma5", momentum),
            ("rpl", "Gamma5", _neg(momentum)),
            ("rps", gamma, zero),
        ),
        0,
    )
    # Appendix C, Fig. 13; its global phase makes the triangle real.
    return np.sqrt(3.0 / 2.0) * np.imag(load_diagram(file, "triangle"))


def _kpi_kpi(
    path: str,
    trajectory: int,
    sink_momentum: Sequence[int],
    source_momentum: Sequence[int],
) -> npt.NDArray:
    file = _contraction_file(
        path,
        "Kpi_Kpi",
        trajectory,
        (
            ("rpl", "Gamma5", sink_momentum),
            ("rpl", "Gamma5", _neg(sink_momentum)),
            ("rps", "Gamma5", source_momentum),
            ("rpl", "Gamma5", _neg(source_momentum)),
        ),
        0,
    )
    # Appendix C, Fig. 14.
    return (
        load_diagram(file, "direct")
        - 1.5 * load_diagram(file, "rectangle")
        + 0.5 * load_diagram(file, "cross")
    )


def _load_kpi_t1u(path: str, trajectory: int) -> npt.NDArray:
    rows: list[npt.NDArray] = []
    for axis, gamma in enumerate(("GammaX", "GammaY", "GammaZ")):
        shell_weights = [
            _t1u_weights(norm_squared, axis) for norm_squared in (1, 2, 3, 4)
        ]
        vector = _vv(path, trajectory, gamma)
        sample = np.empty((5, 5, vector.size), dtype=vector.dtype)
        sample[0, 0] = vector
        for index, weights in enumerate(shell_weights, start=1):
            value = sum(
                (
                    weight * _kpi_v(path, trajectory, momentum, gamma)
                    for momentum, weight in weights.items()
                ),
                start=np.zeros_like(vector),
            )
            sample[index, 0] = value
            sample[0, index] = value.conj()
        for sink_index, sink_weights in enumerate(shell_weights, start=1):
            for source_index, source_weights in enumerate(
                shell_weights[sink_index - 1 :], start=sink_index
            ):
                value = sum(
                    (
                        sink_weight
                        * source_weight
                        * _kpi_kpi(path, trajectory, sink_momentum, source_momentum)
                        for sink_momentum, sink_weight in sink_weights.items()
                        for source_momentum, source_weight in source_weights.items()
                    ),
                    start=np.zeros_like(vector),
                )
                sample[sink_index, source_index] = value
                if sink_index != source_index:
                    sample[source_index, sink_index] = value.conj()
        rows.append(0.5 * (sample + sample.swapaxes(0, 1).conj()))
    # Section II D: average the three T1u irrep rows.
    return np.mean(rows, axis=0)


def process_kpi_t1u(
    path: str,
    trajectory_list: Sequence[int] | None = None,
    *,
    workers: int = 1,
    progress: bool = True,
) -> str:
    """Process Kpi T1u[000] correlators into a single HDF5 dataset.

    The five operators are the vector bilinear and Kpi cubic harmonics on
    momentum shells |d|^2 = 1, 2, 3, and 4. The three T1u rows are averaged,
    as in the paper. The output is `path` with an `.h5` suffix and contains
    a `kpi_t1u` dataset with shape `(n_trajectory, 5, 5, nt)`.

    Set `workers` above one to process trajectories in parallel.
    """
    if trajectory_list is None:
        trajectory_list = trajectories(path)
    if workers < 1:
        raise ValueError("workers must be positive")
    selected_trajectories = tuple(trajectory_list)
    samples: dict[int, npt.NDArray] = {}

    with tqdm(
        total=len(selected_trajectories),
        desc="Loading Kpi T1u",
        unit="traj",
        disable=not progress,
    ) as bar:
        if workers == 1:
            for row, trajectory in enumerate(selected_trajectories):
                samples[row] = _load_kpi_t1u(path, trajectory)
                bar.update()
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_load_kpi_t1u, path, trajectory): row
                    for row, trajectory in enumerate(selected_trajectories)
                }
                for future in as_completed(futures):
                    samples[futures[future]] = future.result()
                    bar.update()
    data = np.stack([samples[row] for row in range(len(selected_trajectories))])
    output = f"{path}.h5"
    with h5py.File(output, "w") as file:
        dataset = file.create_dataset(
            "kpi_t1u",
            data=data,
            chunks=(1, *data.shape[1:]),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            fletcher32=True,
        )
        dataset.attrs["axes"] = [
            "trajectory",
            "sink_operator",
            "source_operator",
            "time",
        ]
        dataset.attrs["operator_basis"] = [
            "T1u vector bilinear, x/y/z rows averaged",
            "Kpi T1u |d|^2 = 1, x/y/z rows averaged",
            "Kpi T1u |d|^2 = 2, x/y/z rows averaged",
            "Kpi T1u |d|^2 = 3, x/y/z rows averaged",
            "Kpi T1u |d|^2 = 4, x/y/z rows averaged",
        ]
    return output
