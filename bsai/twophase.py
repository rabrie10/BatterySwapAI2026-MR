"""Two-phase Wiener degradation with a random changepoint (plateau -> plunge).

The margin ``m(t) = smooth_v(t) - 2.4`` of a healthy cell sits on a plateau:
daily increments concentrate near zero. At a latent onset day tau (geometric
hazard rho per calendar day, absorbing) the dynamics switch to the plunge
regime with drift mu2 << mu1 (measured on the 82 crossed devices: median final
42-day drift -2.7 mV/d against a plateau core near -0.1). This is the standard
two-phase change-point Wiener of the degradation literature (RESS 2019; Kong,
Balakrishnan & Cui 2017; PMC10781245) in the discrete daily form the smoothed
series supports; see outputs/litreview_methods.md P3-1.

Emission model -- measured, not assumed. Three measured pathologies rule out
the plain two-Gaussian emission, and each is handled structurally:

* Daily increments are extremely heavy-tailed (pooled std 5.96 mV against a
  robust MAD-sigma of 0.51 mV), driven by gap re-entries and re-smoothing
  steps. A plain fit splits on VARIANCE, not drift (measured: sigma2/sigma1 =
  3.7 with mu2 barely below mu1). Handled by a jump component (lam, sigma_J)
  in the PLATEAU emission only. Keeping the plunge emission pure-core is what
  encodes irreversibility as evidence: a knee in a seven-day-median series is
  a steep SMOOTH fall, so a large recovery day is strong evidence against an
  ongoing plunge. With the jump shared by both states (the first attempt),
  recoveries were absorbed as "artifact" under either hypothesis and carried
  no evidence, so the sawtooth of a floor-sitter -- descend, recover, descend
  -- ratcheted the onset posterior to 1.0 (measured: mean pi 0.97-1.0 in
  every dwell band).
* 35% of raw daily increments are exact zeros (rolling-median holds), a
  discrete atom any Gaussian mixture collapses onto (measured: sigma_core ->
  floor, lambda -> cap). Handled upstream: hold runs are merged so increments
  are (dm != 0 over dt days); a completed hold still scores as strong plateau
  evidence through its tiny dm over a long dt.
* Per-device wiggle scale spans a decade (robust daily scale 0.3 to 6 mV; the
  five audit zombies sit at 3-6 mV), so with any global emission scale the
  second state degenerates into "the volatile devices" and their onset
  posterior saturates (measured: pi = 1.0 across all dwell bands). Handled by
  a per-device, causally-estimated observation scale s_d(t) (expanding robust
  MAD of the device's own increments -- the CMVN / per-group standardization
  precedent, computable at plan time):

    f_1(x | dt, s) = (1-lam) N(x; mu1 dt, (sigma1 s)^2 dt)
                   + lam     N(x; mu1 dt, (sigma_J s)^2 dt)
    f_2(x | dt, s) =          N(x; mu2 dt, (sigma2 s)^2 dt)

  The drifts stay PHYSICAL (V/day): the same -2.7 mV/d knee is overwhelming
  evidence on a quiet device and weak evidence on a noisy one, which is the
  honest inference under heteroscedastic observation noise.

EM stays closed-form (nested responsibilities; ECM for the common-mean
update).

The passage law, by contrast, uses the FLEET reference scale (sigma_s *
scale_ref), not the device's own s_d: the zombies' measured floor behaviour --
years of +-3-6 mV daily wiggle at 2-16 mV margin with no crossing -- says the
per-device excess wiggle is anti-persistent observation noise that does not
accumulate into the level, and pricing it as Brownian state diffusion is
exactly the incumbent's certainty-manufacturing failure. The honest check on
this choice is the dwell-table gate (g1) and the sum-p gate (g4).

Why this attacks the measured zombie inversion: the per-device posterior
``pi_t = P(plunge | increments to t)`` is a likelihood statement. A fresh
sustained fall (z = -2.7/0.6 per day against the plateau core) drives pi up in
two or three days; once the fall stops on a floor, flat days crush pi back
down, because an absorbed plunge state would have kept falling. Long low-margin
dwell without sustained falls is evidence FOR the plateau state -- the dwell
table (realized 0.80 / 0.29 / 0.18) becomes a likelihood statement instead of a
post-hoc p-knockdown (which measured dead five times through the planner).

Fitting is exact EM. The chain is absorbing with two states, so the
forward-backward reduces to enumerating the changepoint boundary per device:
posterior over tau is prior (geometric over calendar days, plus a point mass
pi0 for "already in plunge at first observation") times the product of segment
likelihoods, computable with two cumulative sums per device. Missing days enter
through dt-scaling of the emission and the transition 1-(1-rho)^dt, exactly as
Wiener increments demand.

Prediction from margin m and posterior pi over a horizon of h days:

    p(h) = pi * IG(m; mu2, sigma2, h)
         + (1-pi) * [ sum_{k=1..h} rho (1-rho)^(k-1) *
                        ( IG(m; mu1, sigma1, k-1)
                          + (1 - IG(m; mu1, sigma1, k-1)) *
                            IG(m + mu1 (k-1); mu2, sigma2, h-k+1) )
                      + (1-rho)^h * IG(m; mu1, sigma1, h) ]

where IG is the closed-form first-passage law in ``bsai.wiener``. Two
deliberate choices:

* The two-segment crossing is decomposed conditionally rather than by pooling
  the total drop into one Gaussian: a pooled N(mu_mix, var_mix) counts paths
  absorbed during the plateau as still alive at onset and misprices the
  reflection term with the mixed variance. The conditional form is exact on
  the tau-mixture up to one approximation -- the phase-2 segment restarts from
  the mean plateau path m + mu1*(k-1) instead of the survival-conditioned
  density (the literature's refinement; second-order here, and checked by the
  degradation test: with mu1=mu2, sigma1=sigma2 the mixture telescopes to the
  single-phase law, so the incumbent Wiener is the exact special case).
* The passage law uses the CORE sigmas, not the jump-inflated pooled std. The
  jump days are artifacts of re-smoothing after data gaps, not sustained state
  motion, and the EOL record requires the smoothed level itself to sit below
  the threshold; pricing jump variance into the barrier law is precisely the
  incumbent's failure mode (volatility + small margin read as certain touch).
  The honest check on this choice is the dwell-table and sum-p gates.

``barrier_shift`` applies the Broadie-Glasserman-Kou continuity correction for
discretely (daily) monitored barriers: the recorded EOL fires when the DAILY
smoothed value is below 2.4, not when a continuous path touches it, so the
continuous law overstates touch probability; the standard correction moves the
barrier 0.5826 * sigma * sqrt(1 day) away, applied per segment with that
segment's sigma.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from .features import FEATURE_NAMES, FeatureContext
from .hazard import HORIZON_GRID
from .margin import EOL_THRESHOLD
from .wiener import first_passage_probability

VOLTAGE_FEATURE = FEATURE_NAMES.index("voltage")

MIN_SIGMA = 1e-5
MIN_RHO = 1e-7
MAX_RHO = 0.25
MIN_PI0 = 1e-6
MAX_PI0 = 0.5
MIN_LAMBDA = 1e-4
MAX_LAMBDA = 0.6
# Broadie-Glasserman-Kou barrier offset for daily monitoring, in units of
# sigma * sqrt(1 day).
BGK_BETA = 0.5826



@dataclass
class TwoPhaseParams:
    """Population parameters of the two-phase Wiener with geometric onset."""

    mu1: float
    sigma1: float  # relative to the per-device observation scale
    mu2: float
    sigma2: float  # relative to the per-device observation scale
    rho: float
    pi0: float
    lam: float = 0.0  # plateau jump-component weight
    sigma_j: float = 1.0  # plateau jump-component scale (relative)
    scale_ref: float = 1.0  # fleet reference scale (V per sqrt-day)
    # Plateau passage diffusion after per-device drift (see trailing_drift):
    # the global sigma1 pools cross-device trend heterogeneity, which the
    # random-drift passage law prices through the per-device trailing drift
    # instead; the residual near-floor core diffusion is what remains for the
    # Brownian term. Measured by the fitting tool; NaN falls back to sigma1.
    mu1_nf: float = float("nan")  # unused fallback slot, kept for reporting
    sigma1_nf: float = float("nan")
    n_nf: int = 0
    # Conditioned onset hazard (the covariate iteration). Bins: (floor &
    # fresh dwell, floor & chronic dwell, mid margin, high margin), where
    # floor is margin < 50 mV and chronic means the device first went below
    # 2.45 V more than rho_dwell_split days ago. The dwell split carries the
    # measured frailty: among stalled floor rows the 42-day onset mass is
    # ~0.12-0.17 -- margin-independent, because a knee covers any sub-50 mV
    # distance -- while the chronic floor-survivors (the audit zombies, with
    # years of exposure and no onset) drag a pooled floor hazard to 3.7e-4/d.
    # A device that has survived a full seasonal cycle at the floor is its own
    # evidence of a lower onset rate; this is the dwell table 0.80/0.29/0.18
    # expressed as exposure statistics instead of a post-hoc knockdown.
    rho_edges: tuple[float, ...] = (0.05, 0.15)
    rho_dwell_split: float = 365.0
    rho_values: tuple[float, ...] = ()
    # Conditional plateau rebound at the floor (V/day at margin 0, tapered
    # linearly to 0 at 50 mV). The plateau branch is by construction the
    # NO-ONSET path, and the measured no-onset forward mean of stalled floor
    # rows is +12.8 mV over 42 days -- seasonal recovery off the post-knee
    # flat. Without it no centered Gaussian endpoint law can price a cell
    # sitting 2-9 mV above the barrier below ~0.4, which is the zombie
    # inversion all over again.
    rebound: float = 0.0
    loglik: float = float("nan")
    n_increments: int = 0
    n_devices: int = 0
    n_iter: int = 0

    def rho_for(self, margin: np.ndarray, dwell: np.ndarray | None = None) -> np.ndarray:
        margin = np.asarray(margin, dtype=float)
        if not self.rho_values:
            return np.full(margin.shape, self.rho)
        values = np.asarray(self.rho_values, dtype=float)
        band = np.searchsorted(np.asarray(self.rho_edges), margin, side="right")
        out = values[np.clip(band + 1, 1, 3)]  # mid/high occupy slots 2..3
        floor = band == 0
        if floor.any():
            chronic = (
                np.zeros(margin.shape, dtype=bool)
                if dwell is None
                else np.asarray(dwell, dtype=float) > self.rho_dwell_split
            )
            out = np.where(floor, np.where(chronic, values[1], values[0]), out)
        return out

    @property
    def mu1_passage(self) -> float:
        return self.mu1_nf if np.isfinite(self.mu1_nf) else self.mu1

    @property
    def sigma1_passage(self) -> float:
        sigma = self.sigma1_nf if np.isfinite(self.sigma1_nf) else self.sigma1
        return sigma * self.scale_ref

    @property
    def sigma2_passage(self) -> float:
        return self.sigma2 * self.scale_ref

    def as_dict(self) -> dict:
        return {
            "mu1": self.mu1,
            "sigma1_rel": self.sigma1,
            "mu1_passage": self.mu1_passage,
            "sigma1_passage": self.sigma1_passage,
            "mu2": self.mu2,
            "sigma2_rel": self.sigma2,
            "sigma2_passage": self.sigma2_passage,
            "rho": self.rho,
            "pi0": self.pi0,
            "lam": self.lam,
            "sigma_j_rel": self.sigma_j,
            "scale_ref": self.scale_ref,
            "n_nf": self.n_nf,
            "loglik": self.loglik,
            "n_increments": self.n_increments,
            "n_devices": self.n_devices,
            "n_iter": self.n_iter,
        }


@dataclass
class DeviceTrack:
    """Valid (pre-crossing, hold-merged) days of one device's smoothed margin."""

    device: str
    building: str
    days: np.ndarray  # day ordinals of valid observations, increasing
    margin: np.ndarray  # smoothed margin at those days
    scale: np.ndarray | None = None  # per-increment causal observation scale

    @property
    def dm(self) -> np.ndarray:
        return np.diff(self.margin)

    @property
    def dt(self) -> np.ndarray:
        return np.diff(self.days).astype(float)

    @property
    def s2dt(self) -> np.ndarray:
        """Per-increment variance base (scale^2 * dt)."""
        dt = self.dt
        if self.scale is None:
            return dt
        return self.scale * self.scale * dt


def causal_scales(
    dm: np.ndarray,
    dt: np.ndarray,
    *,
    min_count: int = 30,
    refresh: int = 28,
    floor: float = 5e-5,
) -> np.ndarray:
    """Expanding robust MAD of |dm|/sqrt(dt), refreshed every ``refresh``
    increments, using strictly past increments only. Early increments (before
    ``min_count``) are NaN and are filled with the fleet median by the caller.
    """
    z = np.abs(dm) / np.sqrt(dt)
    out = np.full(dm.shape[0], np.nan)
    current = np.nan
    next_update = min_count
    for i in range(dm.shape[0]):
        if i >= next_update or (i >= min_count and np.isnan(current)):
            past = z[:i]
            current = max(1.4826 * float(np.median(np.abs(past - np.median(past)))), floor)
            next_update = i + refresh
        out[i] = current
    return out


def _log_normal(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    return -0.5 * (np.log(2.0 * np.pi * var) + (x - mean) ** 2 / var)


def _log_emission(
    dm: np.ndarray, dt: np.ndarray, s2dt: np.ndarray,
    mu: float, sigma: float, lam: float, sigma_j: float,
) -> np.ndarray:
    """Marginal log density of the core+jump scale mixture."""
    core = _log_normal(dm, mu * dt, sigma * sigma * s2dt)
    if lam <= MIN_LAMBDA:
        return core
    jump = _log_normal(dm, mu * dt, sigma_j * sigma_j * s2dt)
    return np.logaddexp(np.log1p(-lam) + core, np.log(lam) + jump)


def _jump_responsibility(
    dm: np.ndarray, dt: np.ndarray, s2dt: np.ndarray,
    mu: float, sigma: float, lam: float, sigma_j: float,
) -> np.ndarray:
    if lam <= MIN_LAMBDA:
        return np.zeros_like(dm)
    core = np.log1p(-lam) + _log_normal(dm, mu * dt, sigma * sigma * s2dt)
    jump = np.log(lam) + _log_normal(dm, mu * dt, sigma_j * sigma_j * s2dt)
    return np.exp(jump - np.logaddexp(core, jump))


def _changepoint_posterior(
    l1: np.ndarray, l2: np.ndarray, dt: np.ndarray, elapsed: np.ndarray, p: TwoPhaseParams
) -> tuple[np.ndarray, float]:
    """Posterior over the onset boundary for one device.

    Atoms 0..N are ``tau = boundary j`` (increments 1..j plateau, j+1..N
    plunge; the transition is applied after a boundary's emission so the
    boundary's own increment stays plateau). Atom N+1 is "no onset within the
    observed span". Returns (posterior, log marginal likelihood).
    """
    n = l1.shape[0]
    c1 = np.concatenate([[0.0], np.cumsum(l1)])  # c1[j] = sum_{i<=j} l1
    c2_tail = np.concatenate([np.cumsum(l2[::-1])[::-1], [0.0]])  # sum_{i>j} l2
    ll = np.empty(n + 2)
    ll[: n + 1] = c1 + c2_tail
    ll[n + 1] = c1[n]  # never switching: all increments plateau

    log_keep = np.log1p(-p.rho)
    lp = np.empty(n + 2)
    lp[0] = np.log(max(p.pi0, MIN_PI0))
    if n:
        # P(tau = t_j) = (1-pi0) (1-rho)^{e_{j-1}} (1 - (1-rho)^{dt_j})
        lp[1 : n + 1] = (
            np.log1p(-p.pi0)
            + elapsed[:-1] * log_keep
            + np.log(-np.expm1(dt * log_keep))
        )
    lp[n + 1] = np.log1p(-p.pi0) + elapsed[-1] * log_keep

    joint = lp + ll
    z = float(logsumexp(joint))
    return np.exp(joint - z), z


def fit_em(
    tracks: list[DeviceTrack],
    *,
    init: TwoPhaseParams | None = None,
    max_iter: int = 300,
    tol: float = 1e-9,
    mu2_ceiling: float | None = None,
    sigma2_cap_ratio: float | None = 1.5,
) -> TwoPhaseParams:
    """EM for the two-phase Wiener with the shared-jump emission.

    The E-step is the exact forward-backward of the absorbing 2-state chain
    (changepoint enumeration) plus nested jump responsibilities; the M-step is
    closed form (ECM for the common-mean updates).

    Two identity constraints keep state 2 the KNEE rather than the dominant
    "fleet seasonal decline" regime the unconstrained likelihood prefers
    (measured: mu2 -0.6 mV/d, i.e. barely steeper than the plateau, with pi
    saturating for every device that drifted recently):

    * ``mu2_ceiling`` caps the plunge drift from above -- the physics prior on
      minimum knee steepness (the crossed devices' final-42-day drift has
      q90 = -1.27 mV/d, median -2.7).
    * ``sigma2_cap_ratio`` caps sigma2 at that multiple of sigma1, so state 2
      cannot balloon into a wide catch-all for mobile days: a knee in the
      smoothed series is a steep SMOOTH fall, not a high-variance regime.

    Either can be None to fit freely.
    """
    data = []
    device_scales = []
    for track in tracks:
        dm, dt = track.dm, track.dt
        if dm.shape[0] == 0:
            continue
        elapsed = np.concatenate([[0.0], np.cumsum(dt)])
        data.append((dm, dt, track.s2dt, elapsed, track.margin[:-1]))
        if track.scale is not None:
            device_scales.append(float(np.median(track.scale)))
    if not data:
        raise ValueError("no increments to fit")
    n_inc = int(sum(d[0].shape[0] for d in data))
    scale_ref = float(np.median(device_scales)) if device_scales else 1.0

    if init is None:
        # Anchor the plunge at the measured knee steepness (median final-42-day
        # drift of the crossed devices is about -2.7 mV/d) so EM starts at the
        # physical optimum rather than a variance split; the zero-merge in the
        # track builder removes the median-hold atom that otherwise captures
        # the core component, and the per-device scales make sigma relative.
        params = TwoPhaseParams(
            mu1=0.0,
            sigma1=1.0,
            mu2=-2.7e-3,
            sigma2=2.0,
            rho=3e-4,
            pi0=0.02,
            lam=0.10,
            sigma_j=6.0,
            scale_ref=scale_ref,
        )
    else:
        params = TwoPhaseParams(**{
            k: getattr(init, k)
            for k in ("mu1", "sigma1", "mu2", "sigma2", "rho", "pi0", "lam", "sigma_j")
        })
        params.scale_ref = scale_ref

    previous = -np.inf
    iteration = 0
    for iteration in range(1, max_iter + 1):
        # accumulators: per state, core/jump weighted sums
        prec_x = np.zeros(2)  # sum w * x * omega   (omega = precision mix)
        prec_t = np.zeros(2)  # sum w * dt * omega
        core_sq = np.zeros(2)  # sum w (1-r) (x - mu dt)^2 / dt
        core_n = np.zeros(2)
        jump_sq = 0.0
        jump_n = 0.0
        weight_n = 0.0
        onsets = 0.0
        exposure = 0.0
        pi0_mass = 0.0
        total_ll = 0.0

        inv_core = np.array([1.0 / params.sigma1**2, 1.0 / params.sigma2**2])
        inv_jump = 1.0 / params.sigma_j**2
        mus = (params.mu1, params.mu2)
        sigmas = (params.sigma1, params.sigma2)

        for dm, dt, s2dt, elapsed, _ in data:
            # Jump component in the plateau emission only: a knee is a steep
            # smooth fall, so recoveries and spikes must count against it.
            l1 = _log_emission(dm, dt, s2dt, params.mu1, params.sigma1, params.lam, params.sigma_j)
            l2 = _log_emission(dm, dt, s2dt, params.mu2, params.sigma2, 0.0, params.sigma_j)
            post, z = _changepoint_posterior(l1, l2, dt, elapsed, params)
            total_ll += z
            n = dm.shape[0]
            boundary = post[: n + 1]
            w2 = np.cumsum(boundary)[:n]  # P(increment i is plunge)
            weights = (1.0 - w2, w2)
            for state in (0, 1):
                w = weights[state]
                lam_state = params.lam if state == 0 else 0.0
                r = _jump_responsibility(
                    dm, dt, s2dt, mus[state], sigmas[state], lam_state, params.sigma_j
                )
                # var = sigma^2 * s^2 * dt, so the WLS weight of (dm - mu dt)
                # carries dt/s2dt = 1/s^2 into both sums.
                omega = ((1.0 - r) * inv_core[state] + r * inv_jump) * (dt / s2dt)
                prec_x[state] += float((w * omega) @ dm)
                prec_t[state] += float((w * omega) @ dt)
                resid = (dm - mus[state] * dt) ** 2 / s2dt
                core_sq[state] += float((w * (1.0 - r)) @ resid)
                core_n[state] += float((w * (1.0 - r)).sum())
                jump_sq += float((w * r) @ resid)
                jump_n += float((w * r).sum())
                if state == 0:
                    weight_n += float(w.sum())
            onsets += float(boundary[1:].sum())
            exposure += float(boundary @ elapsed) + float(post[n + 1] * elapsed[-1])
            pi0_mass += float(post[0])

        mu = np.where(prec_t > 1e-12, prec_x / np.maximum(prec_t, 1e-12), 0.0)
        mu1, mu2 = float(mu[0]), float(mu[1])
        var = np.where(
            core_n > 1e-9, core_sq / np.maximum(core_n, 1e-9), MIN_SIGMA**2
        )
        sg1, sg2 = np.sqrt(np.maximum(var, MIN_SIGMA**2))
        sg1, sg2 = float(sg1), float(sg2)
        if mu2 > mu1:  # keep state 2 the steeper-falling one
            mu1, mu2, sg1, sg2 = mu2, mu1, sg2, sg1
        if mu2_ceiling is not None:
            mu2 = min(mu2, mu2_ceiling)
        if sigma2_cap_ratio is not None:
            sg2 = min(sg2, sigma2_cap_ratio * sg1)
        sigma_j = float(
            np.sqrt(max(jump_sq / max(jump_n, 1e-9), (1.5 * max(sg1, sg2)) ** 2))
        )
        lam = float(np.clip(jump_n / max(weight_n, 1e-9), MIN_LAMBDA, MAX_LAMBDA))
        params = TwoPhaseParams(
            mu1=mu1,
            sigma1=sg1,
            mu2=mu2,
            sigma2=sg2,
            rho=float(np.clip(onsets / max(exposure, 1.0), MIN_RHO, MAX_RHO)),
            pi0=float(np.clip(pi0_mass / len(data), MIN_PI0, MAX_PI0)),
            lam=lam,
            sigma_j=sigma_j,
            scale_ref=scale_ref,
            loglik=total_ll,
            n_increments=n_inc,
            n_devices=len(data),
            n_iter=iteration,
        )
        if abs(total_ll - previous) < tol * (1.0 + abs(total_ll)):
            break
        previous = total_ll
    return params


def plateau_weights(
    track: DeviceTrack, p: TwoPhaseParams
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed plateau responsibility and jump responsibility per increment."""
    dm, dt, s2dt = track.dm, track.dt, track.s2dt
    elapsed = np.concatenate([[0.0], np.cumsum(dt)])
    l1 = _log_emission(dm, dt, s2dt, p.mu1, p.sigma1, p.lam, p.sigma_j)
    l2 = _log_emission(dm, dt, s2dt, p.mu2, p.sigma2, 0.0, p.sigma_j)
    post, _ = _changepoint_posterior(l1, l2, dt, elapsed, p)
    n = dm.shape[0]
    w1 = 1.0 - np.cumsum(post[: n + 1])[:n]
    r = _jump_responsibility(dm, dt, s2dt, p.mu1, p.sigma1, p.lam, p.sigma_j)
    return w1, r


def onset_hazard_bins(
    tracks: list[DeviceTrack],
    p: TwoPhaseParams,
    *,
    edges: tuple[float, ...] = (0.05, 0.15),
    dwell_split: float = 365.0,
    min_exposure: float = 600.0,
) -> tuple[float, ...]:
    """Posterior onset events over plateau exposure, binned by (margin, dwell).

    The changepoint posterior localizes each device's onset; assigning its
    mass to the (margin, floor-dwell) bin of the increment it precedes and
    dividing by the plateau-weighted exposure in that bin gives the
    conditioned geometric hazard. Bin order: (floor fresh, floor chronic,
    mid margin, high margin). Bins with thin exposure keep the homogeneous
    rho.
    """
    onsets = np.zeros(4)
    exposure = np.zeros(4)
    for track in tracks:
        dm, dt, s2dt = track.dm, track.dt, track.s2dt
        if dm.shape[0] == 0:
            continue
        elapsed = np.concatenate([[0.0], np.cumsum(dt)])
        l1 = _log_emission(dm, dt, s2dt, p.mu1, p.sigma1, p.lam, p.sigma_j)
        l2 = _log_emission(dm, dt, s2dt, p.mu2, p.sigma2, 0.0, p.sigma_j)
        post, _ = _changepoint_posterior(l1, l2, dt, elapsed, p)
        n = dm.shape[0]
        cp = np.cumsum(post[: n + 1])
        w1 = 1.0 - cp[:n]  # plateau probability of increment i
        m_start = track.margin[:-1]
        below = track.margin < edges[0]
        first_below = float(track.days[np.argmax(below)]) if below.any() else np.inf
        dwell = track.days[:-1] - first_below
        band = np.searchsorted(np.asarray(edges), m_start, side="right")
        bins = np.where(
            band == 0,
            np.where(dwell > dwell_split, 1, 0),
            band + 1,
        )
        np.add.at(exposure, bins, w1 * dt)
        np.add.at(onsets, bins, post[1 : n + 1])
    values = []
    for onset, expo in zip(onsets, exposure):
        if expo < min_exposure:
            values.append(p.rho)
        else:
            values.append(float(np.clip(onset / expo, MIN_RHO, MAX_RHO)))
    return tuple(values)


def trailing_drift(
    days: np.ndarray,
    margin: np.ndarray,
    prior: float,
    *,
    window: float = 42.0,
    kappa: float = 21.0,
) -> np.ndarray:
    """Causal empirical-Bayes drift of one device at each boundary.

    The standard two-phase Wiener literature gives each unit its own drift
    (unit-to-unit variability); here the plateau passage drift is the device's
    trailing ``window``-day slope shrunk toward the population plateau drift
    with prior strength ``kappa`` days of exposure. The window equals the
    decision horizon so that a device whose fall STALLED more than a window
    ago reads as flat -- the measured discriminator between the slow-crossers
    (touch rate 0.36-0.87 by margin) and the floor-sitters at the same margin.
    The window start brackets across holds: the anchor is the last observation
    at or before ``t - window`` (a level that has not moved for sixty days has
    drift zero, not "no data").
    """
    out = np.empty(days.shape[0])
    for j in range(days.shape[0]):
        start = int(np.searchsorted(days, days[j] - window, side="right")) - 1
        if start < 0:
            start = 0
        span = float(days[j] - days[start])
        if span <= 0:
            out[j] = prior
            continue
        slope = (margin[j] - margin[start]) / span
        weight = span / (span + kappa)
        out[j] = weight * slope + (1.0 - weight) * prior
    return out


def single_phase_loglik(
    tracks: list[DeviceTrack], *, max_iter: int = 200
) -> tuple[float, float, float]:
    """One-state counterpart with the same core+jump emission, for the LR test."""
    dm = np.concatenate([t.dm for t in tracks if t.dm.shape[0]])
    dt = np.concatenate([t.dt for t in tracks if t.dt.shape[0]])
    s2dt = np.concatenate([t.s2dt for t in tracks if t.dm.shape[0]])
    mu = float(np.median(dm / dt))
    sigma, lam, sigma_j = 1.0, 0.15, 6.0
    ll = -np.inf
    for _ in range(max_iter):
        r = _jump_responsibility(dm, dt, s2dt, mu, sigma, lam, sigma_j)
        omega = ((1.0 - r) / sigma**2 + r / sigma_j**2) * (dt / s2dt)
        mu = float((omega * dm).sum() / max((omega * dt).sum(), 1e-12))
        resid = (dm - mu * dt) ** 2 / s2dt
        sigma = float(
            np.sqrt(max(((1.0 - r) * resid).sum() / max((1.0 - r).sum(), 1e-9), MIN_SIGMA**2))
        )
        sigma_j = float(
            np.sqrt(max((r * resid).sum() / max(r.sum(), 1e-9), (1.5 * sigma) ** 2))
        )
        lam = float(np.clip(r.mean(), MIN_LAMBDA, MAX_LAMBDA))
        new_ll = float(_log_emission(dm, dt, s2dt, mu, sigma, lam, sigma_j).sum())
        if abs(new_ll - ll) < 1e-9 * (1.0 + abs(new_ll)):
            ll = new_ll
            break
        ll = new_ll
    return mu, sigma, ll


def forward_pi(track: DeviceTrack, p: TwoPhaseParams) -> np.ndarray:
    """Causal posterior P(plunge | data so far) at each observation boundary.

    Convention matches the EM: the emission of an increment is scored before
    the transition at its end boundary, so ``out[j]`` conditions on increments
    1..j and allows onset at any boundary <= j.
    """
    dm, dt, s2dt = track.dm, track.dt, track.s2dt
    n = dm.shape[0]
    out = np.empty(n + 1)
    out[0] = p.pi0
    if n == 0:
        return out
    l1 = _log_emission(dm, dt, s2dt, p.mu1, p.sigma1, p.lam, p.sigma_j)
    l2 = _log_emission(dm, dt, s2dt, p.mu2, p.sigma2, 0.0, p.sigma_j)
    log_keep = np.log1p(-p.rho)
    a1 = np.log1p(-p.pi0)
    a2 = np.log(max(p.pi0, MIN_PI0))
    for i in range(n):
        a1 += l1[i]
        a2 += l2[i]
        # transition at this boundary, over the increment's own gap
        stay = dt[i] * log_keep
        switch = np.log(-np.expm1(stay))
        a2 = float(np.logaddexp(a2, a1 + switch))
        a1 = float(a1 + stay)
        norm = float(np.logaddexp(a1, a2))
        a1 -= norm
        a2 -= norm
        out[i + 1] = np.exp(a2)
    return np.clip(out, 0.0, 1.0)


def _passage(
    margin: np.ndarray,
    drop: np.ndarray,
    sigma: np.ndarray,
    *,
    per_day_sigma: float,
    barrier_shift: float,
    reflection_weight: float,
) -> np.ndarray:
    if barrier_shift:
        margin = margin + barrier_shift * per_day_sigma
    return first_passage_probability(margin, drop, sigma, reflection_weight)


def mixture_crossing(
    margin: np.ndarray,
    pi: np.ndarray,
    horizon: float,
    p: TwoPhaseParams,
    *,
    mu1_row: np.ndarray | None = None,
    dwell_row: np.ndarray | None = None,
    reflection_weight: float = 1.0,
    barrier_shift: float = 0.0,
) -> np.ndarray:
    """P(first passage to 0 within ``horizon`` days) for each (margin, pi) row.

    ``mu1_row`` is the per-device plateau drift (random-drift Wiener) and
    ``dwell_row`` the days since the device first went below 2.45 V (for the
    dwell-conditioned floor hazard); None uses the population values.
    """
    margin = np.asarray(margin, dtype=float)
    pi = np.clip(np.asarray(pi, dtype=float), 0.0, 1.0)
    h = float(horizon)
    if h <= 0.0:
        return np.where(margin <= 0.0, 1.0, 0.0)

    kwargs = dict(barrier_shift=barrier_shift, reflection_weight=reflection_weight)
    mu1 = (
        np.full_like(margin, p.mu1_passage)
        if mu1_row is None
        else np.asarray(mu1_row, dtype=float)
    )
    if p.rebound:
        mu1 = mu1 + p.rebound * np.clip(1.0 - margin / 0.05, 0.0, 1.0)
    sigma1 = p.sigma1_passage
    sigma2 = p.sigma2_passage
    sqrt_h = np.sqrt(h)
    p_plunge = _passage(
        margin, np.full_like(margin, -p.mu2 * h), np.full_like(margin, sigma2 * sqrt_h),
        per_day_sigma=sigma2, **kwargs,
    )

    steps = max(int(round(h)), 1)
    a = np.arange(steps, dtype=float)  # plateau days before onset (k-1)
    b = h - a  # plunge days after onset
    m_col = margin[:, None]
    mu1_col = mu1[:, None]
    survive_plateau = _passage(
        m_col, -mu1_col * a[None, :], np.maximum(sigma1 * np.sqrt(a), MIN_SIGMA)[None, :],
        per_day_sigma=sigma1, **kwargs,
    )
    from_onset = _passage(
        m_col + mu1_col * a[None, :], -p.mu2 * b[None, :], (sigma2 * np.sqrt(b))[None, :],
        per_day_sigma=sigma2, **kwargs,
    )
    per_onset = survive_plateau + (1.0 - survive_plateau) * from_onset
    rho = p.rho_for(margin, dwell_row)  # conditioned onset hazard per row
    weights = rho[:, None] * np.power(1.0 - rho[:, None], a[None, :])
    never_weight = np.power(1.0 - rho, h)
    p_plateau_only = _passage(
        margin, -mu1 * h, np.full_like(margin, sigma1 * sqrt_h),
        per_day_sigma=sigma1, **kwargs,
    )
    p_plateau = (per_onset * weights).sum(axis=1) + never_weight * p_plateau_only

    out = pi * p_plunge + (1.0 - pi) * p_plateau
    return np.clip(np.where(margin <= 1e-4, 1.0, out), 0.0, 1.0)


@dataclass
class TwoPhaseModel:
    """Fold-dispatching two-phase Wiener with per-device onset posteriors.

    Presents the ``WienerModel`` interface (predict_grid / cdf_at / calibration
    / model_version) so ``HazardForecaster`` and the Task 2 planner run
    unchanged. The dispatch is leave-building-out honest by construction: each
    device's parameters and its stored pi table come from the fold whose
    training data excluded that device's building, so handing this single
    object to ``validate_v6.py --production`` still scores every device with a
    model that never saw its building. Unknown devices fall back to the
    production (all-buildings) parameters with the onset prior pi0.
    """

    params_by_fold: dict[int, TwoPhaseParams]
    fold_of_device: dict[str, int]
    production_params: TwoPhaseParams
    # device -> (observation-day ordinals, causal pi, causal trailing drift),
    # computed with the device's own out-of-fold parameters.
    pi_tables: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
    end_ordinal: dict[str, int]
    climatology: np.ndarray
    # device -> day ordinal of the first smoothed value below 2.45 V (-1 if
    # never); feeds the dwell-conditioned floor hazard.
    first_below: dict[str, int] = field(default_factory=dict)
    calibration: object | None = None  # RemainingCalibration or {fold: RemainingCalibration}
    reflection_weight: float = 1.0
    barrier_shift: float = 0.0
    horizons: tuple[int, ...] = HORIZON_GRID
    feature_names: tuple[str, ...] = tuple(FEATURE_NAMES)
    model_version: str = "bsai-twophase/v1"

    def context(self) -> FeatureContext:
        return FeatureContext(climatology=self.climatology)

    def _params_for(self, fold: int) -> TwoPhaseParams:
        return self.params_by_fold.get(fold, self.production_params)

    def state_at(
        self, device: str, origin_ordinal: float, p: TwoPhaseParams
    ) -> tuple[float, float]:
        """(onset posterior, trailing plateau drift) at a cutoff ordinal."""
        table = self.pi_tables.get(device)
        if table is None:
            return p.pi0, p.mu1
        days, pis, drifts = table
        index = int(np.searchsorted(days, origin_ordinal, side="right")) - 1
        if index < 0:
            return p.pi0, p.mu1
        pi = float(pis[index])
        gap = float(origin_ordinal) - float(days[index])
        if gap > 0:
            # No observations since: onset may have happened unseen.
            pi = 1.0 - (1.0 - pi) * (1.0 - p.rho) ** gap
        return pi, float(drifts[index])

    def pi_at(self, device: str, origin_ordinal: float, p: TwoPhaseParams) -> float:
        return self.state_at(device, origin_ordinal, p)[0]

    def _calibration_for(self, fold: int):
        calibration = self.calibration
        if isinstance(calibration, dict):
            return calibration.get(fold)
        return calibration

    def predict_rows(
        self,
        margin: np.ndarray,
        remaining: np.ndarray,
        devices: np.ndarray,
        horizons: np.ndarray | None = None,
    ) -> np.ndarray:
        """CDF grid from raw margins; the workhorse behind ``predict_grid``."""
        margin = np.asarray(margin, dtype=float)
        remaining = np.asarray(remaining, dtype=float)
        grid = (
            np.asarray(self.horizons, dtype=float)
            if horizons is None
            else np.asarray(horizons, dtype=float)
        )
        rows = margin.shape[0]
        out = np.zeros((rows, grid.shape[0]))
        if rows == 0:
            return out

        names = np.asarray([str(d) for d in devices])
        folds = np.asarray([self.fold_of_device.get(name, -1) for name in names])
        for fold in np.unique(folds):
            params = self._params_for(int(fold))
            in_fold = folds == fold
            fold_rows = np.flatnonzero(in_fold)
            pi = np.empty(fold_rows.shape[0])
            drift = np.empty(fold_rows.shape[0])
            dwell = np.full(fold_rows.shape[0], -1.0)
            for slot, row in enumerate(fold_rows):
                end = self.end_ordinal.get(names[row])
                if end is None:
                    pi[slot], drift[slot] = params.pi0, params.mu1
                    continue
                origin = float(end) - float(round(remaining[row]))
                pi[slot], drift[slot] = self.state_at(names[row], origin, params)
                below = self.first_below.get(names[row], -1)
                if 0 <= below <= origin:
                    dwell[slot] = origin - below
            for rem in np.unique(remaining[in_fold]):
                sub = in_fold & (remaining == rem)
                sub_rows = np.flatnonzero(sub)
                sub_slots = np.searchsorted(fold_rows, sub_rows)
                cache: dict[float, np.ndarray] = {}
                for column, h in enumerate(grid):
                    effective = float(max(min(h, rem), 0.0))
                    if effective <= 0.0:
                        continue
                    if effective not in cache:
                        cache[effective] = mixture_crossing(
                            margin[sub],
                            pi[sub_slots],
                            effective,
                            params,
                            mu1_row=drift[sub_slots],
                            dwell_row=dwell[sub_slots],
                            reflection_weight=self.reflection_weight,
                            barrier_shift=self.barrier_shift,
                        )
                    out[sub_rows, column] = cache[effective]
            calibration = self._calibration_for(int(fold))
            if calibration is not None:
                out[fold_rows] = calibration.apply(out[fold_rows], remaining[fold_rows])
        return np.maximum.accumulate(np.clip(out, 0.0, 1.0), axis=1)

    def predict_grid(
        self,
        features: np.ndarray,
        remaining: np.ndarray,
        devices: np.ndarray | None = None,
    ) -> np.ndarray:
        rows = features.shape[0]
        if rows == 0:
            return np.zeros((0, len(self.horizons)))
        if devices is None:
            devices = np.asarray([""] * rows)
        margin = features[:, VOLTAGE_FEATURE].astype(float) - EOL_THRESHOLD
        return self.predict_rows(margin, remaining, devices)

    def cdf_at(self, grid_values: np.ndarray, days: np.ndarray) -> np.ndarray:
        grid = np.asarray(self.horizons, dtype=float)
        days = np.asarray(days, dtype=float)
        if grid_values.shape[0] == 0:
            return np.zeros((0, days.shape[0]))
        anchored_x = np.concatenate([[0.0], grid])
        anchored_y = np.hstack([np.zeros((grid_values.shape[0], 1)), grid_values])
        out = np.empty((grid_values.shape[0], days.shape[0]))
        for row in range(grid_values.shape[0]):
            out[row] = np.interp(days, anchored_x, anchored_y[row])
        return out


# The pi-hybrid: the changepoint filter's causal state as two extra GBDT
# features (litreview P3-1 stage 2). Appended at the END of the base feature
# vector, so every index-based reader of the incumbent pipeline (voltage 0,
# days_below_2.45 32, beta_30 55) is untouched.
PI_FEATURE_NAMES = ("pi_posterior", "pi_drift")


def pi_feature_lookup(
    tables: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    device: str,
    day_ordinal: float,
    default: tuple[float, float] = (0.0, -2.0e-4),
) -> tuple[float, float]:
    """(pi_posterior, pi_drift) at the last observation at or before a day.

    No staleness propagation: the training frame and the deployed forecaster
    must see the same value for the same (device, day), and the training
    cutoffs sit on observation days; the propagation term is O(rho * gap),
    a <=3% effect the GBDT's staleness features already carry.
    """
    table = tables.get(str(device))
    if table is None:
        return default
    days, pi, drift = table
    index = int(np.searchsorted(days, day_ordinal, side="right")) - 1
    if index < 0:
        return default
    return float(pi[index]), float(drift[index])


@dataclass
class PiHybridModel:
    """Incumbent Wiener GBDT over base features + the filter state, with
    leave-building-out dispatch and the per-device pi lookup inside.

    Presents the ``WienerModel``/``OofHazardModel`` interface. ``devices`` is
    required (the forecaster passes it in production mode); each row's pi
    features come from the filter whose EM never saw the device's building,
    and the row is scored by the fold GBDT that never saw it either.
    """

    by_building: dict[str, object]  # building -> fold WienerModel
    building_of: dict[str, str]
    # device -> (day ordinals, pi_posterior, pi_drift), out-of-fold filters.
    pi_tables: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
    end_ordinal: dict[str, int]
    climatology: np.ndarray
    horizons: tuple[int, ...] = HORIZON_GRID
    feature_names: tuple[str, ...] = tuple(FEATURE_NAMES) + PI_FEATURE_NAMES
    model_version: str = "bsai-wiener/v1+pi"

    def context(self) -> FeatureContext:
        return FeatureContext(climatology=self.climatology)

    def extend_features(
        self, features: np.ndarray, remaining: np.ndarray, devices: np.ndarray
    ) -> np.ndarray:
        rows = features.shape[0]
        extra = np.empty((rows, 2), dtype=features.dtype)
        for row in range(rows):
            device = str(devices[row])
            end = self.end_ordinal.get(device)
            if end is None:
                extra[row] = (0.0, -2.0e-4)
                continue
            day = float(end) - float(round(remaining[row]))
            extra[row] = pi_feature_lookup(self.pi_tables, device, day)
        return np.hstack([features, extra])

    def predict_grid(
        self,
        features: np.ndarray,
        remaining: np.ndarray,
        devices: np.ndarray | None = None,
    ) -> np.ndarray:
        if devices is None:
            raise ValueError("PiHybridModel needs device ids for the pi lookup")
        rows = features.shape[0]
        if rows == 0:
            return np.zeros((0, len(self.horizons)))
        remaining = np.asarray(remaining, dtype=float)
        extended = self.extend_features(features, remaining, np.asarray(devices))
        out = np.zeros((rows, len(self.horizons)))
        buildings = np.asarray(
            [self.building_of.get(str(d), "") for d in devices], dtype=object
        )
        for building in np.unique(buildings):
            model = self.by_building.get(str(building))
            if model is None:
                raise KeyError(f"No out-of-fold model for building {building!r}")
            mask = buildings == building
            out[mask] = model.predict_grid(extended[mask], remaining[mask])
        return out

    def cdf_at(self, grid_values: np.ndarray, days: np.ndarray) -> np.ndarray:
        return next(iter(self.by_building.values())).cdf_at(grid_values, days)


class PiFilterCache:
    """Incremental forward filter over the smoothed margin, per device.

    Rides the forecaster's ``SmoothingCache`` (no re-smoothing): each
    ``update_from`` consumes the grid days added since the last call and
    advances the per-device changepoint filter exactly as the batch
    ``make_tracks`` + ``causal_scales`` + ``forward_pi`` pipeline does --
    hold-merge on exact repeats, causal expanding-MAD observation scale
    (refresh schedule 30/+28, fleet fallback before 30 increments), plateau
    jump mixture, pure-core plunge, transition after emission.

    One deliberate deviation: the LAST grid day is deferred until a later
    update finalizes it, because a scenario cut can land mid-day and the
    boundary day's median is provisional (SmoothingCache re-reads it); a
    sequential filter cannot roll back a consumed observation. The pi feature
    therefore lags at most one observation day behind the batch tables.

    Post-EOL behaviour needs no special casing: a replacement battery's
    recovery jump is large upward evidence, which the pure-core plunge
    emission converts into a fast collapse of the onset posterior.
    """

    def __init__(self, params: TwoPhaseParams, fleet_scale: float) -> None:
        self.params = params
        self.fleet_scale = float(fleet_scale)
        self.devices: dict[str, dict] = {}

    def _new_state(self) -> dict:
        p = self.params
        return {
            "next_index": 0,
            "days": [],
            "margins": [],
            "zhist": [],
            "scale": float("nan"),
            "next_refresh": 30,
            "a1": float(np.log1p(-p.pi0)),
            "a2": float(np.log(max(p.pi0, MIN_PI0))),
            "pi": float(p.pi0),
        }

    def update_from(self, smoothing) -> None:
        for device_id, series in smoothing.devices.items():
            state = self.devices.get(device_id)
            if state is None:
                state = self._new_state()
                self.devices[device_id] = state
            self._advance(state, series)

    def _advance(self, state: dict, series) -> None:
        p = self.params
        values = series.smooth_voltage
        # Defer the (possibly provisional) last grid day.
        stop = len(values) - 1
        index = state["next_index"]
        while index < stop:
            value = values[index]
            if np.isfinite(value):
                margin = float(value) - EOL_THRESHOLD
                day = int(series.origin + index)
                if not state["days"]:
                    state["days"].append(day)
                    state["margins"].append(margin)
                elif margin != state["margins"][-1]:
                    dm = margin - state["margins"][-1]
                    dt = float(day - state["days"][-1])
                    count = len(state["zhist"])  # increments before this one
                    if count >= state["next_refresh"] or (
                        count >= 30 and np.isnan(state["scale"])
                    ):
                        past = np.asarray(state["zhist"], dtype=float)
                        med = float(np.median(past))
                        state["scale"] = max(
                            1.4826 * float(np.median(np.abs(past - med))), 5e-5
                        )
                        state["next_refresh"] = count + 28
                    scale = (
                        state["scale"]
                        if np.isfinite(state["scale"])
                        else self.fleet_scale
                    )
                    s2dt = scale * scale * dt
                    l1 = float(
                        _log_emission(
                            np.asarray([dm]), np.asarray([dt]), np.asarray([s2dt]),
                            p.mu1, p.sigma1, p.lam, p.sigma_j,
                        )[0]
                    )
                    l2 = float(
                        _log_emission(
                            np.asarray([dm]), np.asarray([dt]), np.asarray([s2dt]),
                            p.mu2, p.sigma2, 0.0, p.sigma_j,
                        )[0]
                    )
                    a1 = state["a1"] + l1
                    a2 = state["a2"] + l2
                    stay = dt * np.log1p(-p.rho)
                    switch = np.log(-np.expm1(stay))
                    a2 = float(np.logaddexp(a2, a1 + switch))
                    a1 = float(a1 + stay)
                    norm = float(np.logaddexp(a1, a2))
                    state["a1"] = a1 - norm
                    state["a2"] = a2 - norm
                    state["pi"] = float(np.exp(state["a2"]))
                    state["zhist"].append(abs(dm) / np.sqrt(dt))
                    state["days"].append(day)
                    state["margins"].append(margin)
            index += 1
        state["next_index"] = max(state["next_index"], stop)

    def state_of(self, device: str) -> tuple[float, float]:
        """(pi_posterior, pi_drift) at the last finalized observation."""
        p = self.params
        state = self.devices.get(str(device))
        if state is None or not state["days"]:
            return 0.0, -2.0e-4
        days = state["days"]
        margins = state["margins"]
        import bisect

        last = days[-1]
        anchor = bisect.bisect_right(days, last - 42.0) - 1
        anchor = max(anchor, 0)
        span = float(last - days[anchor])
        if span <= 0:
            drift = p.mu1
        else:
            slope = (margins[-1] - margins[anchor]) / span
            weight = span / (span + 21.0)
            drift = weight * slope + (1.0 - weight) * p.mu1
        pi = state["pi"]
        return pi, pi * p.mu2 + (1.0 - pi) * drift


@dataclass
class ProductionPiHybrid:
    """Deployment wrapper: production 66-feature Wiener GBDT + live pi filter.

    Loads through ``script.py``'s unchanged path
    (``HazardForecaster(joblib.load(path))``): the forecaster sees the
    ``pi_cache`` attribute and feeds it the smoothing cache each ``predict``
    (the ``resurrection_gate`` attachment precedent), so the pi features are
    computed incrementally from the split's own data -- no train-time tables.
    """

    inner: object  # production WienerModel (66 features, calibration inside)
    pi_cache: PiFilterCache
    climatology: np.ndarray
    horizons: tuple[int, ...] = HORIZON_GRID
    feature_names: tuple[str, ...] = tuple(FEATURE_NAMES) + PI_FEATURE_NAMES
    model_version: str = "bsai-wiener/v1+pi-prod"

    @property
    def calibration(self):
        return getattr(self.inner, "calibration", None)

    def context(self) -> FeatureContext:
        return FeatureContext(climatology=self.climatology)

    def predict_grid(
        self,
        features: np.ndarray,
        remaining: np.ndarray,
        devices: np.ndarray | None = None,
    ) -> np.ndarray:
        rows = features.shape[0]
        if rows == 0:
            return np.zeros((0, len(self.horizons)))
        if devices is None:
            raise ValueError("ProductionPiHybrid needs device ids for the pi filter")
        extra = np.empty((rows, 2), dtype=features.dtype)
        for row in range(rows):
            extra[row] = self.pi_cache.state_of(str(devices[row]))
        extended = np.hstack([features, extra])
        return self.inner.predict_grid(extended, remaining)

    def cdf_at(self, grid_values: np.ndarray, days: np.ndarray) -> np.ndarray:
        return self.inner.cdf_at(grid_values, days)
