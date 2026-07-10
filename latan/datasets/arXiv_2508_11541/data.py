import os
from urllib.request import urlretrieve

import h5py
import numpy as np
import numpy.typing as npt


# This function downloads data from the paper "Application of Laplace filters to the
# analysis of lattice time correlators" (https://arxiv.org/abs/2508.11541) hosted on
# Zenodo.
#
# NB: By downloading this data you agree with the Creative Commons Attribution 4.0
# International Licence under which they are shared.
def download(path: str) -> str:
    filename = "data_2508.11541.h5"
    if not os.path.exists(f"{path}/{filename}"):
        print("Downloading data...")
        url = f"https://zenodo.org/records/16921526/files/{filename}?download=1"
        urlretrieve(url, filename)
        print("Done!")
    else:
        print("Data already downloaded")
    return filename


def load_correlator(filename: str, name: str, verbose: bool = False) -> npt.NDArray:
    file = h5py.File(filename)
    if name not in file.keys():
        raise RuntimeError(f"no dataset '{name}' in file '{filename}'")
    dset = file[f"{name}/dataset/data"]
    if isinstance(dset, h5py.Dataset):
        buf = np.array(dset[()])
        if verbose:
            nconf = buf.shape[0]
            nsrc = buf.shape[1]
            nt = buf.shape[2]
            print(f"nConf: {nconf}")
            print(f" nSrc: {nsrc}")
            print(f"   nt: {nt}")
        return buf
    else:
        raise RuntimeError(
            f"cannot read dataset '{name}/dataset/data' in file '{filename}'"
        )


def bin_correlator(
    data: npt.NDArray, bin_size: int, verbose: bool = False
) -> npt.NDArray:
    nconf = data.shape[0]
    nsrc = data.shape[1]
    nt = data.shape[2]
    nbin = (nconf * nsrc) // bin_size
    if verbose:
        print(f" size: {nconf * nsrc}")
        print(f" nBin: {nbin}")
    return data.reshape(nbin, bin_size, nt).mean(axis=1)
