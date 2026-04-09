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

<h3 align='center'>A Python Library for Classical and Dynamical Low-Rank CMA-ES</h3>

---

# General :earth_americas:

*seamaze* is a Python library for classical and Dynamical Low-Rank (DLR) CMA-ES variants. It is designed to navigate complex, high-dimensional fitness landscapes by iteratively adapting a multivariate Gaussian search space to the objective's local topography. By leveraging DLR approximations, seamaze remains computationally efficient even on ill-conditioned or rugged black-box problems. This implementation further extends to the integration of first-order information, constraints, and robust restart mechanisms.

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

*seamaze* has two main classes which provide a classical and a dynamical low-rank CMA-ES variant:

###### Classical CMA-ES

```python
from seamaze.cmaes import CMAES
```

###### Dynamical low-rank CMA-ES

```python
from seamaze.dlrcmaes import DLRCMAES
```

### Dependencies

| Name                           | Version                               |
| -----------------------------: | :------------------------------------ |
| `python`                       | <font size="3"> >=3.11, <4.0 </font>  |
| `numpy`                        | <font size="3"> >=2.4.4 </font>       |
| `scipy`                        | <font size="3"> >=1.17.1 </font>      |
| `numba`                        | <font size="3"> >=0.65.0 </font>      |
| `matplotlib`                   | <font size="3"> >=3.10.8 </font>      |
| `seaborn`                      | <font size="3"> >=0.13.2 </font>      |

Moreover, we are using **Python v3.11.11** and **Spyder IDE v6.1.4** for development.

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
  title = {{seamaze}: a python library for classical and dynamical low-rank CMA-ES},
  author = {Ortkamp, Tim and Patwardhan, Chinmay and Stammer, Pia},
  version = {0.0.1},
  license = {MIT},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/pyanno4rt/seamaze}
}
```

