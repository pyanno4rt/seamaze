[![PyPI](https://img.shields.io/badge/PyPI-seamaze-orange.svg)](https://pypi.org/project/seamaze/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/seamaze)
[![Coverage Status](https://coveralls.io/repos/github/pyanno4rt/seamaze/badge.svg)](https://coveralls.io/github/pyanno4rt/seamaze)
![GitHub Repo stars](https://img.shields.io/github/stars/pyanno4rt/seamaze)
![GitHub forks](https://img.shields.io/github/forks/pyanno4rt/seamaze)
[![GitHub Downloads](https://img.shields.io/github/downloads/pyanno4rt/seamaze/total)](https://github.com/pyanno4rt/seamaze/releases) 
![visitors](https://visitor-badge.laobi.icu/badge?page_id=pyanno4rt.seamaze)
[![GitHub Release](https://img.shields.io/github/v/release/pyanno4rt/seamaze)](https://github.com/pyanno4rt/seamaze/releases)
[![GitHub Discussions](https://img.shields.io/github/discussions/pyanno4rt/seamaze)](https://github.com/pyanno4rt/seamaze/discussions)
[![GitHub Issues](https://img.shields.io/github/issues/pyanno4rt/seamaze)](https://github.com/pyanno4rt/seamaze/issues)
[![GitHub Contributors](https://img.shields.io/github/contributors/pyanno4rt/seamaze)](https://github.com/pyanno4rt/seamaze/graphs/contributors)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/pyanno4rt/seamaze/blob/develop/logo/logo_white.png?raw=true">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/pyanno4rt/seamaze/blob/develop/logo/logo_black.png?raw=true">
    <img alt="logo" src="https://github.com/pyanno4rt/seamaze/blob/develop/logo/logo_white.png?raw=true" width="600">
  </picture>
</p>

<h3 align='center'>A Python Library for Classical, Limited-Memory, and Dynamical Low-Rank CMA-ES</h3>

---

# General :earth_americas:

*seamaze* is a Python library for classical, limited-memory and Dynamical Low-Rank (DLR) variants of the Covariance Matrix Adaptation Evolution Strategy (CMA-ES). It provides state-of-the-art, derivative-free algorithms designed for continuous, non-linear, and non-convex real-parameter optimization, excelling in ill-conditioned, non-separable, or rugged fitness landscapes. By leveraging limited-memory and DLR approximations, *seamaze* maintains computational efficiency even on high-dimensional black-box problems. This implementation further incorporates first-order information, constraint handling, and multi-stage restart mechanisms.

# Installation :computer:

### Python distribution

You can install the latest distribution via:

```bash
pip install seamaze
```

### Source code

You can check the latest source code via:

```bash
git clone https://github.com/pyanno4rt/seamaze.git
```

### Usage

*seamaze* has three main classes which provide a classical, a limited-memory, and a dynamical low-rank CMA-ES variant:

###### Classical CMA-ES

```python
from seamaze.optimizers.evolutionary import CMAES
```

###### Limited-memory CMA-ES

```python
from seamaze.optimizers.evolutionary import LMCMAES
```

###### Dynamical low-rank CMA-ES

```python
from seamaze.optimizers.low_rank import DLRCMAES
```

### Dependencies

| Name                           | Version                               |
| -----------------------------: | :------------------------------------ |
| `python`                       | <font size="3"> >=3.11, <4.0 </font>  |
| `numpy`                        | <font size="3"> >=2.4.4 </font>       |
| `scipy`                        | <font size="3"> >=1.17.1 </font>      |
| `numba`                        | <font size="3"> >=0.65.1 </font>      |
| `matplotlib`                   | <font size="3"> >=3.10.9 </font>      |
| `seaborn`                      | <font size="3"> >=0.13.2 </font>      |

# Development :rocket:

### Important links

* [Github](https://github.com/pyanno4rt/seamaze)
* [PyPI](https://pypi.org/project/seamaze/)
* [Coveralls](https://coveralls.io/github/pyanno4rt/seamaze)
* [Issue tracker](https://github.com/pyanno4rt/seamaze/issues)

# Help and Support :busts_in_silhouette:

### Resources

* [Github Discussions](https://github.com/pyanno4rt/seamaze/discussions)
* [Github Issues](https://github.com/pyanno4rt/seamaze/issues)

### Contact

* [Github Page](https://tortka.github.io)
* [Mail](mailto:tim.ortkamp@gmx.de?subject=Request (seamaze))
* [LinkedIn](https://www.linkedin.com/in/tim-ortkamp)

### Citation

To cite *seamaze*, either use the link in the right sidebar of the Github landing page labeled "Cite this repository" or copy the short-form bib-style paragraph below:

```tex
@software{seamaze,
  title = {{seamaze}: a python library for classical, limited-memory, and dynamical low-rank CMA-ES},
  author = {Ortkamp, Tim and Patwardhan, Chinmay and Stammer, Pia},
  version = {0.0.3},
  license = {MIT},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/pyanno4rt/seamaze}
}
```

