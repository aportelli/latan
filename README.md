# Latan
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

*Statistical data analysis library for lattice field theory*

Disclaimer: this package is still in early development and is expected to receive significant updates.

## Installation

Latan requires Python 3.12 or later.

```shell
pip install latan
```

## Getting started

The package includes [example notebooks](https://github.com/aportelli/latan/tree/main/notebooks) and interfaces for [public datasets](https://github.com/aportelli/latan/tree/main/latan/datasets).

## Development

[Install uv](https://docs.astral.sh/uv/getting-started/installation/) if you have not already done so. Then, at the root of the repository, execute:
```shell
uv sync
```
This will create a Python environment in `.venv`, which should be used to run code from this repository. This environment can be activated in a shell with
```shell
source .venv/bin/activate
```

## Citation policy

If you use Latan in research that leads to a publication, please cite this repository or the corresponding software release.

If contributors to this repository make a substantial intellectual contribution to your research beyond providing the software, please consider co-authorship in line with standard authorship guidelines.

## AI policy

*Latan is not vibe-coded and does not accept vibe-coded contributions*. Nevertheless, LLMs can be effective productivity tools and have been used at intermediate stages of development, including:

- Mechanical coding tasks, such as find-and-replace or propagation of trivial API changes;
- prototyping, testing, and benchmarking during the early stages of a feature;
- first drafts of API documentation;
- low-stakes plotting helpers.

New contributions must respect the following rules:

- All commits must be authored and submitted by a human contributor, who takes responsibility for the submitted code.
- AI-assisted prototyping is permitted. Before code from such a prototype is imported into Latan, a human contributor must review it end-to-end, understand its behavior, and validate it appropriately.
- Unit tests must be curated and validated by a human. Numerical changes should be compared with a trusted reference or independent implementation where practical.
- Documentation may begin as an AI-assisted draft, but must be reviewed, corrected, and approved by a human contributor.
