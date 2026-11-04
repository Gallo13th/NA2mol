"""Radar-chart normalisation, implemented to the manuscript's Methods specification.

Source: main text, Methods, "Radar-chart normalisation". Verbatim requirements:

* The radial coordinate of metric *m* is ``r_m = 1 + 2 s_m``, mapping the inner ring
  (r = 1) to s_m = 0 and the outer ring (r = 3) to s_m = 1, where ``s_m`` in [0, 1] is
  direction-corrected so that a larger value always means better performance.
* Six property-retention axes **pool molecules**::

      s_m = 1/(u-l) * integral_l^u R(t) dt

  with ``R(t) = P(value <= t)`` for a lower-is-better metric (docking score, RMSD,
  SA score, strain) and ``R(t) = P(value >= t)`` for a higher-is-better metric
  (QED, SEI).  Strain energy is evaluated on a log10 scale.  Threshold ranges:
  docking [-11, -3] kcal/mol; RMSD [0, 3] A; QED [0, 1]; SA [1, 7]; SEI [0, 0.3];
  log10(strain) [1, 4] kcal/mol, i.e. 10 to 1e4.  A molecule whose strain exceeds 1e4,
  or which has no computed strain, contributes R(t) = 0 at every threshold and so stays
  in the denominator rather than leaving it.
* The ECFP4 axis **weights targets equally**.  ``s_m`` is a recovery gain relative to
  randomly reassigned target labels::

      s_m = (recovery_obs - recovery_rand) / (recovery_best - recovery_rand)

  clipped to [0, 1], where recovery is the mean over tau in [0.25, 0.5] of the fraction
  of targets with at least one molecule reaching ECFP4-TS >= tau, and the null shuffles
  molecule-to-target labels (1,000 permutations).

Implementation note on the integral
-----------------------------------
``s_m`` has an exact closed form, so no numerical integration is needed.  For one
molecule of value v and a lower-is-better metric::

    (1/(u-l)) * integral_l^u 1[v <= t] dt
        = 1                if v <= l
        = (u - v)/(u - l)  if l < v < u
        = 0                if v >= u

so ``s_m`` is simply the mean over molecules of ``clip((u - v)/(u - l), 0, 1)``.  The
higher-is-better case is the mirror image, ``clip((v - l)/(u - l), 0, 1)``.  Both are
exactly the manuscript's integral; ``integral_form`` below verifies this numerically.
"""
from __future__ import annotations

from typing import NamedTuple
import numpy as np

# ── axis definitions ────────────────────────────────────────────────────────
# (frame column, manuscript label, (lo, hi), direction, log10-scale?)
class Axis(NamedTuple):
    column: str
    label: str
    lo: float
    hi: float
    higher_is_better: bool
    log10: bool = False


RADAR_AXES: tuple[Axis, ...] = (
    Axis('qed',      'QED',    0.0, 1.0,  higher_is_better=True),
    Axis('log10_strain', 'Strain', 1.0, 4.0, higher_is_better=False, log10=True),
    Axis('sa_score', 'SA',     1.0, 7.0,  higher_is_better=False),
    Axis('rmsd',     'RMSD',   0.0, 3.0,  higher_is_better=False),
    Axis('affinity', 'Dock.', -11.0, -3.0, higher_is_better=False),
)
# ECFP4 and SEI complete the seven; drawn clockwise from the top in this order.
AXIS_ORDER = ('qed', 'log10_strain', 'sa_score', 'rmsd', 'affinity', 'ecfp4',
              'ecfp4_specificity', 'sei')
ECFP4_LABEL = 'ECFP4'
SEI_LABEL = 'SEI'
SEI_RANGE = (0.0, 0.3)
ECFP4_RANGE = (0.25, 0.50)
# Target specificity of the ECFP4 recovery: observed minus the permutation null, so 0 is
# "no better than a random target assignment". Symmetric range with 0 at mid-ring.
ECFP4_SPEC_RANGE = (-0.15, 0.15)

# The strain axis sweeps a retention curve over log10(strain) in STRAIN_SWEEP. Above
# the upper bound a pose is no longer meaningfully "retained at a stricter threshold",
# so values are censored there rather than allowed to run to 1e12 and be clipped by
# accident. Molecules whose strain could not be computed (PoseCheck returned nothing or
# exceeded the per-molecule time budget) pass no threshold and score 0, which keeps them
# in the denominator instead of disappearing from it.
STRAIN_SWEEP = (1.0, 4.0)                  # log10 kcal/mol, i.e. 10 to 1e4
STRAIN_CENSOR = 10.0 ** STRAIN_SWEEP[1]    # 1e4: above this a pose is a hard failure
STRAIN_HARD_FAIL = STRAIN_CENSOR           # strain above this is too large to be meaningful
# Operational cut used for the tabulated pass rate and the final-tier cascade.
STRAIN_FAIL_CUT = 200.0

# docking score [-11,-3]; RMSD [0,3]; QED [0,1]; SA [1,7]; SEI [0,0.3]; log10 strain [1,5]
RANGES = {
    'affinity': (-11.0, -3.0),
    'rmsd': (0.0, 3.0),
    'qed': (0.0, 1.0),
    'sa_score': (1.0, 7.0),
    'sei': SEI_RANGE,
    'log10_strain': (1.0, 4.0),
    'ecfp4': ECFP4_RANGE,
    'ecfp4_specificity': ECFP4_SPEC_RANGE,
}
HIGHER_IS_BETTER = {'qed': True, 'sei': True, 'ecfp4': True, 'ecfp4_specificity': True,
                    'affinity': False, 'rmsd': False, 'sa_score': False,
                    'log10_strain': False}
# 'Strain' is one label but the manuscript counts SEI and ECFP4 among the seven.
SEVEN_LABELS = {'qed': 'QED', 'log10_strain': 'Strain', 'sa_score': 'SA', 'rmsd': 'RMSD',
                'affinity': 'Dock.', 'ecfp4': 'ECFP4',
                'ecfp4_specificity': 'ECFP4 spec.', 'sei': 'SEI'}


def radial(s: np.ndarray | float) -> np.ndarray | float:
    """r_m = 1 + 2 s_m, mapping s=0 -> inner ring (r=1), s=1 -> outer ring (r=3)."""
    return 1.0 + 2.0 * np.asarray(s, dtype=float)


def retention_score(values, lo: float, hi: float,
                    higher_is_better: bool, valid=None) -> float:
    """Exact closed form of ``1/(u-l) * integral_l^u R(t) dt`` (see module docstring).

    ``valid`` marks which values may contribute; invalid entries score 0, i.e. they
    pass no threshold.  Returns a value in [0, 1].
    """
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v) if valid is None else (np.isfinite(v) & np.asarray(valid, bool))
    span = float(hi - lo)
    if higher_is_better:
        frac = (v - lo) / span
    else:
        frac = (hi - v) / span
    frac = np.clip(frac, 0.0, 1.0)
    frac = np.where(ok, frac, 0.0)
    if frac.size == 0:
        return float('nan')
    return float(frac.mean())


def integral_form(values, lo: float, hi: float, higher_is_better: bool,
                  n: int = 20001) -> float:
    """Numerical evaluation of the manuscript's integral, for cross-checking.

    Kept separate from :func:`retention_score` so the closed form can be validated
    against the definition it replaces.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float('nan')
    t = np.linspace(lo, hi, n)
    if higher_is_better:
        R = (v[None, :] >= t[:, None]).mean(axis=1)
    else:
        R = (v[None, :] <= t[:, None]).mean(axis=1)
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(_trapz(R, t) / (hi - lo))


# ── per-method scores ───────────────────────────────────────────────────────
def sei(frame):
    """SEI = -docking score / TPSA (kcal/mol per A^2).

    TPSA = 0 makes the ratio undefined; those molecules get SEI = 0, the worst
    attainable value, rather than +inf which would score them as best.
    """
    tpsa = np.asarray(frame['tpsa'], dtype=float)
    aff = np.asarray(frame['affinity'], dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(tpsa > 0, -aff / np.where(tpsa > 0, tpsa, np.nan), 0.0)
    return np.where(np.isfinite(out), out, 0.0)


def strain_pass_rate(frame, cut: float = STRAIN_FAIL_CUT) -> float:
    """Fraction of molecules whose computed strain is at or below `cut`.

    A molecule counts as not passing if it has no strain value --- PoseCheck did not
    finish within the per-molecule budget --- or if its strain exceeds the hard-failure
    bound, or simply if it is above `cut`. All of these stay in the denominator.
    This is the tabulated Strain number; the radar axis is the retention-curve
    integral above, which is a function of the whole distribution.
    """
    raw = np.asarray(frame['strain'], dtype=float)
    if raw.size == 0:
        return float('nan')
    return float(np.mean(np.isfinite(raw) & (raw > 0) & (raw <= cut)))


def property_scores(frame) -> dict[str, float]:
    """The six property-retention axes for one method, pooling all its molecules."""
    vals = {}
    for ax in RADAR_AXES:
        if ax.column == 'log10_strain':
            raw = np.asarray(frame['strain'], dtype=float)
            # a pose above the hard-failure bound is not a very strained pose, it is a failed
            # one: it passes no threshold and stays in the denominator
            valid = np.isfinite(raw) & (raw > 0) & (raw <= STRAIN_HARD_FAIL)
            v = np.where(valid, np.log10(np.where(valid, raw, 10.0)), np.nan)
            vals[ax.column] = retention_score(v, ax.lo, ax.hi, ax.higher_is_better, valid)
        else:
            v = np.asarray(frame[ax.column], dtype=float)
            vals[ax.column] = retention_score(v, ax.lo, ax.hi, ax.higher_is_better)
    vals['sei'] = retention_score(sei(frame), *SEI_RANGE, HIGHER_IS_BETTER['sei'])
    return vals



def property_summaries(frame) -> dict[str, float]:
    """Single-number summaries of the six property metrics, pooling molecules.

    The radar's property axes are retention-curve integrals, i.e. functions of the
    whole distribution, so they cannot be read back as one metric value. These pooled
    medians are the matching summaries for the companion source table (they use the
    same molecule pool as the integrals, not per-target medians).
    """
    out: dict[str, float] = {}
    out['median_docking'] = float(np.nanmedian(np.asarray(frame['affinity'], dtype=float)))
    out['median_rmsd'] = float(np.nanmedian(np.asarray(frame['rmsd'], dtype=float)))
    out['median_qed'] = float(np.nanmedian(np.asarray(frame['qed'], dtype=float)))
    out['median_sa'] = float(np.nanmedian(np.asarray(frame['sa_score'], dtype=float)))
    out['median_sei'] = float(np.nanmedian(sei(frame)))
    strain = np.asarray(frame['strain'], dtype=float)
    # success: a strain value was computed and it is below the hard-failure bound. A
    # molecule that exceeded the budget, or whose strain exceeds 1e4, is a failure and is
    # counted against the method in all three rates below, since the denominator is the
    # full molecule pool either way.
    ok = np.isfinite(strain) & (strain > 0) & (strain <= STRAIN_HARD_FAIL)
    if ok.any():
        out['median_strain'] = float(np.nanmedian(strain[ok]))
        out['median_log10_strain'] = float(np.nanmedian(np.log10(strain[ok])))
    else:
        out['median_strain'] = out['median_log10_strain'] = float('nan')
    out['n_strain_valid'] = int(ok.sum())
    total = strain.size if strain.size else 1
    out['strain_success'] = float(ok.sum() / total)
    out['strain_pass500'] = float(np.sum(ok & (strain <= 500.0)) / total)
    out['strain_pass200'] = float(np.sum(ok & (strain <= 200.0)) / total)
    return out


# ── ECFP4 recovery gain ─────────────────────────────────────────────────────
def ecfp4_specificity_score(specificity: float) -> float:
    """Map observed-minus-null recovery onto [0, 1] over ECFP4_SPEC_RANGE.

    s = 0.5 means the method recovers its own targets no better than randomly assigned
    ones; s = 1 means it beats the null by the top of the range.
    """
    lo, hi = ECFP4_SPEC_RANGE
    return float(np.clip((float(specificity) - lo) / (hi - lo), 0.0, 1.0))


def recovery_from_maxima(maxima: np.ndarray) -> float:
    """Mean over targets of the recovery curve over tau in [0.25, 0.50].

    ``maxima[j]`` is the best ECFP4-TS any molecule assigned to target j reaches
    against that target's actives, or -inf when the target has no output.  Closing
    the integral exactly (higher-is-better, [0.25, 0.50]) gives
    ``clip((max - 0.25)/0.25, 0, 1)`` per target.
    """
    m = np.asarray(maxima, dtype=float)
    return float(np.clip((m - ECFP4_RANGE[0]) / (ECFP4_RANGE[1] - ECFP4_RANGE[0]),
                         0.0, 1.0).mean())


def scatter_maxima(S: np.ndarray, assign: np.ndarray, n_targets: int) -> np.ndarray:
    """Best similarity per target: ``max over i of S[i, assign[i]]``.

    ``S[i, j]`` is molecule i's ECFP4-TS against target j's actives.  Targets that
    receive no molecule keep -inf, which maps to zero recovery.
    """
    maxima = np.full(n_targets, -np.inf)
    np.maximum.at(maxima, assign, S[np.arange(S.shape[0]), assign])
    return maxima


def ecfp4_gain(S: np.ndarray, assign: np.ndarray, n_targets: int,
               permutations: int = 1000, seed: int = 0) -> dict:
    """Recovery and the permutation-standardised gain for one method.

    Returns the observed recovery, the mean over ``permutations`` shuffles of the
    molecule-to-target assignment, and the raw gain (before the cross-method
    normalisation that :func:`normalise_ecfp4` applies).
    """
    obs = recovery_from_maxima(scatter_maxima(S, assign, n_targets))
    rng = np.random.default_rng(seed)
    n = assign.shape[0]
    null = np.empty(permutations, dtype=float)
    for k in range(permutations):
        null[k] = recovery_from_maxima(scatter_maxima(S, rng.permutation(assign), n_targets))
    return {'recovery_observed': obs,
            'recovery_random': float(null.mean()),
            'recovery_random_sd': float(null.std(ddof=1)) if permutations > 1 else float('nan'),
            'n_permutations': permutations}


def normalise_ecfp4(recovery_observed: dict[str, float],
                    recovery_random: dict[str, float]) -> dict[str, float]:
    """``s_m = (obs - rand) / (best - rand)`` clipped to [0, 1], per method.

    ``recovery_best`` is the largest observed value among the compared methods, so
    s_m = 0 at the random baseline and s_m = 1 for the best method, exactly as the
    manuscript specifies.
    """
    best = max(recovery_observed.values())
    out = {}
    for m, obs in recovery_observed.items():
        rand = recovery_random[m]
        denom = best - rand
        s = 0.0 if denom <= 0 else (obs - rand) / denom
        out[m] = float(np.clip(s, 0.0, 1.0))
    return out


# ── drawing ─────────────────────────────────────────────────────────────────
RING_DARK = '#68717A'    # heavy rings at r=1 and r=3
RING_LIGHT = '#DCE0E3'   # spokes and the r=2 ring


def draw_radar(ax, scores_by_method: dict[str, dict], title: str = '',
               colors: dict | None = None, labels: dict | None = None,
               legend: bool = False, fontsize: float = 7.0,
               title_pad: float = 24.0, markers: dict | None = None,
               linestyles: dict | None = None, emphasis: str | None = None,
               fill_alpha: float = 0.08, emphasis_fill_alpha: float = 0.22):
    """Draw the manuscript's seven-axis radar on ``ax`` (a polar axes).

    ``scores_by_method[method]`` maps axis key -> s_m in [0, 1]; the radius drawn is
    r_m = 1 + 2 s_m, so the inner ring is 0 and the outer ring is 1.  Axes run
    clockwise from the top, matching the published panels.
    """
    import numpy as np

    theta = np.linspace(0.0, 2 * np.pi, len(AXIS_ORDER), endpoint=False)
    closed_theta = np.append(theta, theta[0])
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(theta)
    ax.set_xticklabels([SEVEN_LABELS[a] for a in AXIS_ORDER], fontsize=fontsize)
    ax.set_ylim(0.0, 3.0)
    # The reference panels carry no radial tick numbers: the rings are told apart by
    # weight, with the inner (r=1) and outer (r=3) rings heavy and the r=2 ring light.
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels([])
    ax.grid(False)
    for th in theta:
        ax.plot([th, th], [0.0, 3.0], color=RING_LIGHT, linewidth=0.6, zorder=0)
    # the rings are circles, not the 7-gon the axis vertices would produce
    ring_theta = np.linspace(0.0, 2 * np.pi, 361)
    ax.plot(ring_theta, np.full(ring_theta.shape, 2.0),
            color=RING_LIGHT, linewidth=0.6, zorder=0)
    # the r=1 ring marks s=0 but sits inside the filled polygons, where a heavy stroke reads
    # as a hole in the shading; drawn light, it stays readable without that
    ax.plot(ring_theta, np.full(ring_theta.shape, 1.0),
            color=RING_LIGHT, linewidth=0.6, zorder=1)
    ax.spines['polar'].set_color(RING_DARK)
    ax.spines['polar'].set_linewidth(1.8)

    colors = colors or {}
    labels = labels or {}
    markers = markers or {}
    linestyles = linestyles or {}
    closed = closed_theta
    # Fills first, all of them, then the lines: a later series' fill would otherwise
    # wash out an earlier series' line. The emphasis method is drawn last in both passes so
    # its outline and its shadow sit above the others.
    order = [m for m in scores_by_method if m != emphasis]
    if emphasis in scores_by_method:
        order.append(emphasis)

    def _polygon(method):
        sc = scores_by_method[method]
        s = np.array([sc.get(a, 0.0) for a in AXIS_ORDER], dtype=float)
        r = np.asarray(radial(np.nan_to_num(s, nan=0.0)))
        return np.append(r, r[0])

    for method in order:
        r = _polygon(method)
        colour = colors.get(method, '#606060')
        alpha = emphasis_fill_alpha if method == emphasis else fill_alpha
        ax.fill(closed, r, color=colour, alpha=alpha,
                linewidth=0.0, zorder=2 if method == emphasis else 1)

    for method in order:
        r = _polygon(method)
        colour = colors.get(method, '#606060')
        hero = method == emphasis
        ax.plot(closed, r, color=colour,
                linewidth=1.95 if hero else 1.15,
                marker=markers.get(method, 'o'),
                markersize=3.8 if hero else 2.6,
                linestyle=linestyles.get(method, 'solid'),
                label=labels.get(method, method),
                zorder=6 if hero else 3)
    if title:
        # pad clears the topmost axis label, which sits above the circle
        ax.set_title(title, fontsize=fontsize * 1.15, pad=title_pad)
    if legend:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06),
                  ncol=min(4, max(1, len(scores_by_method))), frameon=False,
                  fontsize=fontsize * 0.85, handlelength=1.2)
    return ax
