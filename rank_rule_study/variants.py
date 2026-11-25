"""Build the rank-rule variants from the optimizer of this repository.

'current' is the unchanged optimizer (0.02 soft filter + relative energy
rule). The other rules replace only the rank proposal in _adaptive_bug_step:

- 'soft_eps': soft filter with the noise level eps instead of 0.02,
- 'matrix_bernstein': number of eigenvalues above the 99 % matrix
  Gaussian/Bernstein bound sqrt(2 (m+1) eps^2 log(2m/0.01)) (Tropp 2012),
- 'bernstein_age': the same bound with an age-dependent noise level per
  direction (see _build_age).

eps^2 = (c_mu^2 sum w^2 + c_1^2 (1+a)/(1-a)) / (1 - beta^2) is the stationary
per-entry noise variance of the learned covariance under random selection,
beta = 1 - c_1 - c_mu, a = (1 - c_c)^2 beta.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(REPO, 'seamaze', 'optimizers', '_dlrcmaes.py')

FILTER = """        # Apply a soft noise filter
        threshold = 0.02
"""

ENERGY = """        sigma_off = sign(sigma_off) * maximum(nabs(sigma_off) - threshold, 0.0)

        # Sort by absolute eigenvalue magnitude
        sigma_off = sort(nabs(sigma_off))[::-1]

        # Select the rank from the relative spectral energy
        rank_energy = _energy_rank_selection(
            eigenvalues=sigma_off,
            energy_fraction=1.0-low_rank_energy_tolerance,
            min_rank=1
            )
"""

NOISE = """        # Stationary per-entry noise variance of the learned covariance:
        # rank-mu term plus rank-one term with AR(1) evolution path
        decay_noise = 1.0 - lr_rank_one - lr_rank_mu
        path_corr = (1.0 - lr_cov)**2 * decay_noise
        noise_var = (
            lr_rank_mu**2 * nsum(weights**2)
            + lr_rank_one**2 * (1.0 + path_corr) / (1.0 - path_corr)
            ) / (1.0 - decay_noise**2)
"""

SOFT_EPS = NOISE + """
        # Apply a soft noise filter with the noise level eps
        threshold = sqrt(noise_var)
"""

BERNSTEIN = NOISE + """        size = sigma_off.size

        # Count the eigenvalues above the 99% matrix Gaussian/Bernstein
        # bound sqrt(2 v log(2m/delta)), v = (m+1) eps^2 (Tropp 2012)
        threshold = sqrt(
            2.0 * (size + 1.0) * noise_var * log(2.0 * size / 0.01))
        rank_energy = max(1, int(nsum(nabs(sigma_off) > threshold)))
"""

LABEL = {'current': 'current (0.02 soft filter)',
         'soft_eps': 'soft filter with ε',
         'matrix_bernstein': 'matrix Bernstein with ε',
         'bernstein_age': 'matrix Bernstein with age-dependent ε',
         'mp_hysteresis': 'Marchenko-Pastur filter + hysteresis',
         'bootstrap': 'Rademacher bootstrap filter + hysteresis',
         'bootstrap_nohyst': 'Rademacher bootstrap filter, no hysteresis'}


def _patch(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'{SOURCE} differs from the version this study was written for '
                           f'(not found: {old.strip()[:60]!r}); update variants.py.')
    return text.replace(old, new)


def _build_age(src):
    """Bernstein bound with age-dependent noise: every entry of the whitened
    core is normalized by its own noise level, which depends on how many
    updates the two directions have accumulated. Noise after k updates
    (random selection, whitened covariance):
        Var E_k = c_mu^2 sum w^2 V_k + c_1^2 R_k,
        V_{k+1} = beta^2 V_k + 1,
        R_{k+1} = beta^2 R_k + 1 + 2 beta rho C_k,   C_{k+1} = 1 + a C_k,
    rho = (1 - c_c)^2, a = rho beta, V_0 = R_0 = C_0 = 0; k -> inf gives the
    stationary eps^2. (V, R, C) is stored per basis direction and mixed with
    squared coefficients when directions are rotated or truncated."""
    t = src
    t = _patch(t, "        self._core = zeros(self.rank, dtype=float64)\n",
               "        self._core = zeros(self.rank, dtype=float64)\n"
               "        self._noise_state = zeros((self.rank, 3), dtype=float64)\n")
    t = _patch(t, "            basis_new, core_new, psi_new, rank_new = _adaptive_bug_step(\n"
                  "                self._basis,\n                self._core,\n                self._psi,\n",
               "            (basis_new, core_new, psi_new, rank_new,\n"
               "             noise_new) = _adaptive_bug_step(\n"
               "                self._basis,\n                self._core,\n                self._psi,\n"
               "                self._noise_state,\n")
    t = _patch(t, "            self._psi[:] = psi_new\n",
               "            self._psi[:] = psi_new\n            self._noise_state = noise_new\n")
    t = _patch(t, "    types.Tuple((f8_2d, f8_1d, f8_1d, i8))(\n"
                  "        # Return: Tuple(basis_new, core_new, psi_new, rank_new)\n",
               "    types.Tuple((f8_2d, f8_1d, f8_1d, i8, f8_2d))(\n"
               "        # Return: Tuple(basis_new, core_new, psi_new, rank_new, noise_new)\n")
    t = _patch(t, "        f8_1d,          # psi\n        f8_2d,          # steps\n",
               "        f8_1d,          # psi\n        f8_2d,          # noise_state\n"
               "        f8_2d,          # steps\n")
    t = _patch(t, "    basis, core, psi, steps, sorting, weights, path_cov, lr_cov, lr_rank_one,\n",
               "    basis, core, psi, noise_state, steps, sorting, weights, path_cov, lr_cov,\n"
               "    lr_rank_one,\n")
    t = _patch(t, "    # Get the transposed basis\n    uhat_aug_tr = uhat_aug.T\n",
               "    # Get the transposed basis\n    uhat_aug_tr = uhat_aug.T\n\n"
               "    # Carry the noise state of the old directions into the augmented\n"
               "    # basis (new directions start without accumulated noise) and advance\n"
               "    # it by one update: rank-mu (white) and rank-one (AR(1) path) parts\n"
               "    noise_aug = ((uhat_aug_tr @ basis)**2) @ noise_state\n"
               "    decay_noise = 1.0 - lr_decay\n"
               "    path_rho = (1.0 - lr_cov)**2\n"
               "    noise_next = zeros(noise_aug.shape)\n"
               "    noise_next[:, 0] = decay_noise**2 * noise_aug[:, 0] + 1.0\n"
               "    noise_next[:, 1] = (decay_noise**2 * noise_aug[:, 1] + 1.0\n"
               "                        + 2.0 * decay_noise * path_rho * noise_aug[:, 2])\n"
               "    noise_next[:, 2] = 1.0 + path_rho * decay_noise * noise_aug[:, 2]\n"
               "    noise_dir = (lr_rank_mu**2 * nsum(weights**2) * noise_next[:, 0]\n"
               "                 + lr_rank_one**2 * noise_next[:, 1])\n")
    t = _patch(t, FILTER + ENERGY,
               "        # Normalize each entry by its own noise level; an entry\n"
               "        # accumulates only while both directions exist\n"
               "        noise_pair = minimum(noise_dir[:, None], noise_dir[None, :])\n"
               "        shat_norm = shat_off / sqrt(maximum(noise_pair, 1e-300))\n"
               "        shat_norm = 0.5 * (shat_norm + shat_norm.T)\n"
               "        sigma_norm, _ = eigh(shat_norm)\n"
               "        size = sigma_norm.size\n\n"
               "        # Count the eigenvalues above the 99% matrix Gaussian/Bernstein\n"
               "        # bound for unit noise, sqrt(2 (m+1) log(2m/delta)) (Tropp 2012)\n"
               "        threshold = sqrt(2.0 * (size + 1.0) * log(2.0 * size / 0.01))\n"
               "        rank_energy = max(1, int(nsum(nabs(sigma_norm) > threshold)))\n")
    t = _patch(t, "    return basis_new, core_new, psi_new, rank_new\n",
               "    noise_new = (basis_sigma[:, :rank_new]**2).T @ noise_next\n\n"
               "    return basis_new, core_new, psi_new, rank_new, noise_new\n")
    return t


def build(out_dir):
    """Write one optimizer module per rule into out_dir; return {rule: path}.
    Files are only rewritten if their content changes, so a study that is
    already running can keep loading them."""
    src = open(SOURCE).read()
    texts = {'current': src,
             'soft_eps': _patch(src, FILTER, SOFT_EPS),
             'matrix_bernstein': _patch(src, FILTER + ENERGY, BERNSTEIN),
             'bernstein_age': _build_age(src)}
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for rule, text in texts.items():
        paths[rule] = os.path.join(out_dir, f'{rule}.py')
        if not os.path.exists(paths[rule]) or open(paths[rule]).read() != text:
            tmp = f'{paths[rule]}.{os.getpid()}.tmp'    # unique per process (parallel parts)
            with open(tmp, 'w') as fh:
                fh.write(text)
            os.replace(tmp, paths[rule])
    return paths
