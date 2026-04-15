"""
Gaussian Process Regression — OVID-19 Outbreak Animation
=========================================================
Renders an 8-scene MP4 that walks through every step of GP regression:

  1. Wide prior (no data)
  2. Early wave observations arrive
  3. Work from home (week 16) — posterior bends down
  4. Omicron variant detected (week 34) — surge observed
  5. Post-Omicron decline — full arc in view
  6. Interpolation query (week 25) — tight CI
  7. Extrapolation query (week 50) — wide CI
  8. Final annotated 52-week posterior

Output: outbreak_animation.mp4  (written to the current working directory)

Run via pixi:
    pixi run render

Or directly:
    python ovid_forecasting_example.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these values to customise the animation
# ═══════════════════════════════════════════════════════════════════════════════

# ── Observations ───────────────────────────────────────────────────────────────
# Each row is [week, daily_cases].  Add, remove, or edit rows to change the data.
# The scene list below references observations by their 0-based row index.
OBS = np.array([
    [ 2,    8],   # 0  first confirmed cases (Mar 12)
    [ 4,  220],   # 1  first wave climbing rapidly (Mar 26 — WFH order week)
    [ 7,  820],   # 2  first wave peak ~7-day avg (Apr 16 — mask mandate week)
    [11,  480],   # 3  post-peak decline (May 14 — Stage 1 reopen)
    [18,  590],   # 4  summer plateau (Jul 2)
    [25,  510],   # 5  late summer (Aug 20)
    [32,  950],   # 6  fall surge building (Oct 8)
    [39, 2200],   # 7  Thanksgiving surge (Nov 26)
    [44, 3800],   # 8  winter peak (Dec 31 / Jan 7, 2021)
    [47, 2500],   # 9  post-peak declining (Jan 28, 2021)
], dtype=float)

NOISE             = 150.0  # observation noise std — reflects reporting variability at scale
WFH_WEEK          = 4      # Mar 23 2020: non-essential businesses closed / stay-at-home
OMI_WEEK          = 11     # May 15 2020: Stage 1 reopening (stay-at-home order lifted)
HOSPITAL_CAPACITY = 2500   # daily case rate at which MD hospitals were severely strained (Dec 2020)
ICU_CAPACITY      = 1200   # total licensed ICU beds in Maryland (MHCC FY2020)
VAX_WEEK          = 42     # Dec 14 2020: first COVID vaccines administered in Maryland
MASK_WEEK         = 7      # Apr 15 2020: statewide mask mandate issued
SCHOOL_WEEK       = 27     # Sep 4 2020: Stage 3 reopening (schools authorized to resume)
N_SAMPLES         = 5      # number of GP sample curves to draw (illustrates distribution over functions)
SAMPLE_SEED       = 42     # RNG seed — change to see different plausible sample paths

# ── GP kernel ─────────────────────────────────────────────────────────────────
RBF_LENGTH_SCALE = 6.0    # smoothness: larger → longer correlation length
RBF_SIGNAL_STD   = 2000.0 # amplitude: scaled for Maryland year-1 case range (0–4000)
PRIOR_MEAN       = 1000.0 # prior mean used when no data has been observed yet
PRIOR_STD        = 1500.0 # prior std  used when no data has been observed yet

# ── Output ────────────────────────────────────────────────────────────────────
CI_MULTIPLIER = 1.96   # z-score for the confidence band  (1.96 → 95 % CI)
FPS           = 30     # output frame rate
TRANSITION    = 25     # morph frames between scenes
OUT_PATH           = 'outbreak_animation.mp4'
DPI                = 120
SHOW_KERNEL_MATRIX = False   # set True to show the k(xᵢ,xⱼ) heatmap inset

# ═══════════════════════════════════════════════════════════════════════════════

# ── Colour palette ─────────────────────────────────────────────────────────────
BG      = '#0d1117'   # figure + axes background
SURFACE = '#161b22'   # legend panel
TRUE_C  = '#e2e8f0'   # ground-truth curve (near-white)
OBS_C   = '#f0a500'   # observation dots   (amber)
MEAN_C  = '#3fb950'   # GP mean line       (green)
CI_C    = '#3fb950'   # CI band            (same green)
WFH_C   = '#79c0ff'   # work from home marker     (bright blue)
OMI_C   = '#bc8cff'   # Omicron marker     (vibrant purple)
QRY_C   = '#ff7b72'   # query marker       (coral)
BED_C   = '#f85149'   # hospital capacity  (red)
ICU_C   = '#dc2626'   # ICU capacity       (deep red — more urgent)
VAX_C   = '#06b6d4'   # vaccine rollout    (cyan — hopeful)
MASK_C  = '#f97316'   # mask mandate off   (orange — caution)
SCH_C   = '#eab308'   # school reopening   (yellow)
TXT_C   = '#e6edf3'   # title text
MUT_C   = '#8b949e'   # axis labels + desc

# ── Derived ────────────────────────────────────────────────────────────────────
obsX = OBS[:, 0]
obsY = OBS[:, 1]
n    = len(obsX)

# ── GP helpers ─────────────────────────────────────────────────────────────────
def build_cov_matrix(x):
    """Squared-exponential covariance matrix with observation noise on the diagonal."""
    dists = np.subtract.outer(x, x) / RBF_LENGTH_SCALE
    return RBF_SIGNAL_STD**2 * np.exp(-0.5 * dists**2) + np.eye(len(x)) * NOISE**2


def gp_predict(xs, obs_mask):
    """
    Return posterior mean and std at locations xs, conditioned on the
    observations selected by boolean/index array obs_mask.
    Falls back to a wide prior when no observations are selected.
    """
    ox = obsX[obs_mask]
    oy = obsY[obs_mask]
    if len(ox) == 0:
        return np.full_like(xs, PRIOR_MEAN), np.full_like(xs, PRIOR_STD)
    Km    = build_cov_matrix(ox)
    alm   = np.linalg.solve(Km, oy)
    Kim   = np.linalg.inv(Km)
    dists = np.subtract.outer(xs, ox) / RBF_LENGTH_SCALE
    kstar = RBF_SIGNAL_STD**2 * np.exp(-0.5 * dists**2)
    mu    = kstar @ alm
    var   = np.array([RBF_SIGNAL_STD**2 - kstar[i] @ Kim @ kstar[i]
                      for i in range(len(xs))])
    return mu, np.sqrt(np.maximum(0, var))


def gp_sample(xs, obs_mask):
    """
    Draw N_SAMPLES functions from the GP posterior (or prior when no data).
    Each sample is a plausible smooth epidemic curve — the CI band is their envelope.
    Uses SAMPLE_SEED so the same curves appear every scene (no flickering).
    """
    rng = np.random.RandomState(SAMPLE_SEED)
    # Prior covariance over test points (no observation noise — we sample f, not y)
    dists_ss = np.subtract.outer(xs, xs) / RBF_LENGTH_SCALE
    Kss = RBF_SIGNAL_STD**2 * np.exp(-0.5 * dists_ss**2) + np.eye(len(xs)) * 1e-6

    ox = obsX[obs_mask]
    oy = obsY[obs_mask]
    if len(ox) == 0:
        return rng.multivariate_normal(np.full(len(xs), PRIOR_MEAN), Kss, size=N_SAMPLES)

    Km    = build_cov_matrix(ox)
    Kim   = np.linalg.inv(Km)
    dists = np.subtract.outer(xs, ox) / RBF_LENGTH_SCALE
    kstar = RBF_SIGNAL_STD**2 * np.exp(-0.5 * dists**2)
    mu    = kstar @ np.linalg.solve(Km, oy)
    cov   = Kss - kstar @ Kim @ kstar.T
    cov   = 0.5 * (cov + cov.T)           # enforce symmetry
    cov  += np.eye(len(xs)) * 1e-2        # jitter ensures positive-definiteness at larger amplitudes
    return rng.multivariate_normal(mu, cov, size=N_SAMPLES)


def remove_errbar(eb):
    """Remove a matplotlib errorbar container and all of its child artists."""
    eb[0].remove()
    for artist in eb[1]: artist.remove()
    for artist in eb[2]: artist.remove()


def true_curve(x):
    """
    Idealised noise-free epidemic curve approximating Maryland's year-1 COVID arc.
    Each reported data point ≈ true_curve(week) + Gaussian(0, NOISE).
    Three components: spring first wave, summer plateau, and winter surge.
    """
    first_wave = 800  * np.exp(-0.5 * ((x -  7.0) / 3.5)**2)   # peak ~Apr 16
    summer     = 500  * np.exp(-0.5 * ((x - 21.0) / 9.0)**2)   # broad summer plateau
    winter     = 3800 * np.exp(-0.5 * ((x - 43.5) / 6.0)**2)   # peak ~Dec 31/Jan 7
    return np.maximum(0, 20 + first_wave + summer + winter)


# ── Scene definitions ──────────────────────────────────────────────────────────
# obs_revealed: row indices from OBS that are visible in this scene (0-based)
# query:        a week number to show a forecast marker at, or None
XMIN, XMAX = 1,  52   # one year: week 1 = Mar 5 2020, week 52 = Mar 4 2021
YMIN, YMAX = 0, 5000

scenes = [
    {
        'title':        'Step 1 · Prior — Before Any Data  (Maryland, Mar 2020)',
        'desc':         'GP prior: 5 plausible epidemic curves — broad uncertainty before any reports',
        'obs_revealed': [],
        'show_wfh':     False,
        'show_omi':     False,
        'show_vax':     False,
        'show_mask_off': False,
        'show_school':  False,
        'query':        None,
        'hold_frames':  60,
    },
    {
        'title':        'Step 2 · First Cases + Stay-at-Home Order  (W2–W4)',
        'desc':         'Early seeding; non-essential businesses close Mar 23 — samples compress toward low counts',
        'obs_revealed': [0, 1],
        'show_wfh':     True,    # WFH order at W4, same week as obs[1]
        'show_omi':     False,
        'show_vax':     False,
        'show_mask_off': False,
        'show_school':  False,
        'query':        None,
        'hold_frames':  55,
    },
    {
        'title':        'Step 3 · First Wave Peak + Mask Mandate  (W7)',
        'desc':         'Cases hit ~820/day by Apr 16; mask mandate issued — GP posterior captures sharp rise and fall',
        'obs_revealed': [0, 1, 2, 3],
        'show_wfh':     True,
        'show_omi':     True,    # Stage 1 reopen at W11 = obs[3]
        'show_vax':     False,
        'show_mask_off': True,   # mask mandate issued W7 = obs[2]
        'show_school':  False,
        'query':        None,
        'hold_frames':  60,
    },
    {
        'title':        'Step 4 · Summer Plateau  (W18–W25)',
        'desc':         'Stage 3 reopening (Sep 4) keeps cases at 500–600/day; uncertainty narrows between data points',
        'obs_revealed': [0, 1, 2, 3, 4, 5],
        'show_wfh':     True,
        'show_omi':     True,
        'show_vax':     False,
        'show_mask_off': True,
        'show_school':  True,    # Stage 3 reopen W27, shown here as obs[5] at W25 is very close
        'query':        None,
        'hold_frames':  60,
    },
    {
        'title':        'Step 5 · Fall / Thanksgiving Surge  (W32–W39)',
        'desc':         'Fall mixing accelerates spread — GP posterior pivots sharply upward; samples diverge',
        'obs_revealed': [0, 1, 2, 3, 4, 5, 6, 7],
        'show_wfh':     True,
        'show_omi':     True,
        'show_vax':     False,
        'show_mask_off': True,
        'show_school':  True,
        'query':        None,
        'hold_frames':  60,
    },
    {
        'title':        'Step 6 · Winter Peak + Vaccine Rollout  (W44, vaccine W42)',
        'desc':         'Cases peaked ~3,800/day around Jan 7 2021; vaccines began Dec 14 — turning point visible',
        'obs_revealed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'show_wfh':     True,
        'show_omi':     True,
        'show_vax':     True,    # vaccines at W42 — revealed once obs[8] at W44 is in view
        'show_mask_off': True,
        'show_school':  True,
        'query':        None,
        'hold_frames':  65,
    },
    {
        'title':        'Step 7 · Predict: Interpolation  (Week 15)',
        'desc':         'Between first wave and summer plateau — GP confident here, narrow 95 % CI',
        'obs_revealed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'show_wfh':     True,
        'show_omi':     True,
        'show_vax':     True,
        'show_mask_off': True,
        'show_school':  True,
        'query':        15,
        'hold_frames':  60,
    },
    {
        'title':        'Step 8 · Predict: Extrapolation  (Week 52)',
        'desc':         'Beyond last observation — samples fan out; uncertainty quantifies what we do not yet know',
        'obs_revealed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'show_wfh':     True,
        'show_omi':     True,
        'show_vax':     True,
        'show_mask_off': True,
        'show_school':  True,
        'query':        52,
        'hold_frames':  70,
    },
    {
        'title':        'Final Posterior — Maryland Year 1  (Mar 2020 – Mar 2021)',
        'desc':         'Confident reconstruction over the observed period; honest uncertainty past last data point',
        'obs_revealed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'show_wfh':     True,
        'show_omi':     True,
        'show_vax':     True,
        'show_mask_off': True,
        'show_school':  True,
        'query':        None,
        'final':        True,
        'hold_frames':  90,
    },
]

# ── Build frame list ───────────────────────────────────────────────────────────
frames = []
for si, scene in enumerate(scenes):
    for _ in range(scene['hold_frames']):
        frames.append({'scene': si, 't': 1.0})
    if si < len(scenes) - 1:
        for fi in range(TRANSITION):
            t = fi / TRANSITION
            t = t * t * (3 - 2 * t)   # smoothstep: eases in and out so transitions don't look abrupt
            frames.append({'scene': si, 'next_scene': si + 1, 't': t, 'transitioning': True})

# ── Figure setup ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(XMIN, XMAX)
ax.set_ylim(YMIN, YMAX)
ax.spines[['top', 'right']].set_visible(False)
ax.spines['left'].set_color('#30363d')
ax.spines['bottom'].set_color('#30363d')
ax.tick_params(colors=MUT_C, labelsize=9)
ax.set_xlabel('Week', color=MUT_C, fontsize=10)
ax.set_ylabel('Daily Cases', color=MUT_C, fontsize=10)
ax.xaxis.set_tick_params(color='#30363d')
ax.yaxis.set_tick_params(color='#30363d')
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_color(MUT_C)
ax.grid(axis='y', color='#21262d', linewidth=0.7, alpha=0.8)
ax.set_xticks([XMIN, WFH_WEEK, MASK_WEEK, OMI_WEEK, 20, SCHOOL_WEEK, 38, VAX_WEEK, XMAX])
ax.set_xticklabels(
    [f'W{w}' for w in [XMIN, WFH_WEEK, MASK_WEEK, OMI_WEEK, 20, SCHOOL_WEEK, 38, VAX_WEEK, XMAX]],
    fontsize=8,
)

# Pre-compute the static ground-truth curve
xs_true = np.linspace(XMIN, XMAX, 500)
ys_true = true_curve(xs_true)

# Persistent plot artists
true_line,  = ax.plot(xs_true, ys_true, color=TRUE_C, linewidth=1.4,
                      linestyle='--', alpha=0.55, zorder=2, label='Simulated truth')
ci_band     = ax.fill_between([], [], [], color=CI_C, alpha=0.15, zorder=3)
mean_line,  = ax.plot([], [], color=MEAN_C, linewidth=2.2, zorder=4)
obs_scatter = ax.scatter([], [], color=OBS_C, s=70, zorder=6,
                         edgecolors=BG, linewidths=1.4)
wfh_line    = ax.axvline(WFH_WEEK, color=WFH_C, linewidth=1.5, linestyle='--', alpha=0, zorder=5)
omi_line    = ax.axvline(OMI_WEEK, color=OMI_C, linewidth=1.5, linestyle='--', alpha=0, zorder=5)
wfh_label   = ax.text(WFH_WEEK + 0.3, YMAX * 0.92, 'WFH',  color=WFH_C,
                      fontsize=8.5, fontweight='bold', alpha=0, zorder=7)
omi_label   = ax.text(OMI_WEEK + 0.3, YMAX * 0.77, 'Reopen\nS1', color=OMI_C,
                      fontsize=8.5, fontweight='bold', alpha=0, zorder=7)
qry_line    = ax.axvline(0, color=QRY_C, linewidth=1.2, linestyle=':', alpha=0, zorder=5)
qry_errbar  = ax.errorbar([], [], yerr=[], fmt='o', color=QRY_C,
                           capsize=5, capthick=1.5, elinewidth=2, markersize=7, zorder=7)
qry_label   = ax.text(0, 0, '', color=QRY_C, fontsize=9, ha='center', zorder=8,
                      fontfamily='monospace')
fore_shade  = ax.axvspan(obsX.max(), XMAX, alpha=0, color=MEAN_C, zorder=1)
bed_line    = ax.axhline(HOSPITAL_CAPACITY, color=BED_C, linewidth=1.8,
                         linestyle='-.', alpha=0.85, zorder=5,
                         label=f'Hospital capacity ({HOSPITAL_CAPACITY:,})')
ax.text(XMAX - 0.3, HOSPITAL_CAPACITY + 20, f'Hospital surge threshold\n({HOSPITAL_CAPACITY:,} cases/day)',
        color=BED_C, fontsize=8, ha='right', va='bottom', alpha=0.85, zorder=7)
icu_line    = ax.axhline(ICU_CAPACITY, color=ICU_C, linewidth=1.5,
                          linestyle=':', alpha=0.85, zorder=5)
ax.text(XMIN + 0.5, ICU_CAPACITY + 18, f'ICU beds ({ICU_CAPACITY:,} statewide)',
        color=ICU_C, fontsize=8, ha='left', va='bottom', alpha=0.85, zorder=7)
# Intervention verticals — start invisible (alpha=0), revealed per scene
vax_line    = ax.axvline(VAX_WEEK,    color=VAX_C,  linewidth=1.5, linestyle='--', alpha=0, zorder=5)
mask_line   = ax.axvline(MASK_WEEK,   color=MASK_C, linewidth=1.5, linestyle='--', alpha=0, zorder=5)
school_line = ax.axvline(SCHOOL_WEEK, color=SCH_C,  linewidth=1.5, linestyle='--', alpha=0, zorder=5)
vax_label   = ax.text(VAX_WEEK    + 0.3, YMAX * 0.77, 'Vaccine',    color=VAX_C,
                      fontsize=8.5, fontweight='bold', alpha=0, zorder=7)
mask_label  = ax.text(MASK_WEEK   + 0.3, YMAX * 0.62, 'Mask\nissued', color=MASK_C,
                      fontsize=8.5, fontweight='bold', alpha=0, zorder=7)
school_label = ax.text(SCHOOL_WEEK + 0.3, YMAX * 0.77, 'Reopen\nS3', color=SCH_C,
                       fontsize=8.5, fontweight='bold', alpha=0, zorder=7)
# GP sample curves — faint plausible functions drawn from the posterior distribution
sample_lines = [ax.plot([], [], color=MEAN_C, linewidth=0.8, alpha=0.25, zorder=2)[0]
                for _ in range(N_SAMPLES)]
title_txt   = ax.text(0.5, 1.04, '', transform=ax.transAxes,
                      ha='center', va='bottom', color=TXT_C, fontsize=13, fontweight='bold')
desc_txt    = ax.text(0.5, -0.13, '', transform=ax.transAxes,
                      ha='center', va='top', color=MUT_C, fontsize=9.5, style='italic')

leg_elements = [
    mlines.Line2D([0],[0], color=TRUE_C, linewidth=1.4, linestyle='--',
                  alpha=0.7, label='Simulated truth'),
    mlines.Line2D([0],[0], marker='o', color='w', markerfacecolor=OBS_C,
                  markersize=8, label='Reported cases'),
    mlines.Line2D([0],[0], marker='o', color='w', markerfacecolor=QRY_C,
                  markersize=7, label='Forecast query'),
]
ax.legend(handles=leg_elements, loc='upper left', fontsize=8,
          framealpha=0.3, facecolor=SURFACE, edgecolor='#30363d', labelcolor=TXT_C)

plt.tight_layout(rect=[0, 0.06, 1, 0.91])

# ── Kernel matrix inset (optional) ────────────────────────────────────────────
# Positioned upper-right; shows k(xᵢ,xⱼ) = exp(−½ Δt²/ℓ²) as a live heatmap.
# Each cell is the normalised RBF correlation between two observed weeks.
# Diagonal = 1 (a point is perfectly correlated with itself); off-diagonal values
# shrink toward 0 as weeks grow farther apart — revealing the GP's smoothness prior.
if SHOW_KERNEL_MATRIX:
    k_ax = ax.inset_axes((0.655, 0.53, 0.335, 0.44))
    k_ax.set_facecolor(SURFACE)
    for sp in k_ax.spines.values():
        sp.set_color('#30363d')
    k_ax.set_title(f'Kernel  k(xᵢ,xⱼ) = exp(−½ Δt²/{RBF_LENGTH_SCALE:.0f}²)',
                   fontsize=6, color=MUT_C, pad=3)

# Cache GP predictions and samples per scene to avoid recomputing every frame
xs_full = np.linspace(XMIN, XMAX, 300)
scene_cache  = {}
sample_cache = {}
for si, sc in enumerate(scenes):
    mask = np.array([i in sc['obs_revealed'] for i in range(n)])
    scene_cache[si]  = gp_predict(xs_full, mask)
    sample_cache[si] = gp_sample(xs_full, mask)

def lerp(a, b, t):
    return a * (1 - t) + b * t

_last_k_mask = None   # cache so we only redraw the heatmap when obs_mask changes

def update_k_matrix(obs_mask):
    """Redraw the kernel matrix inset for the current set of observations."""
    global _last_k_mask
    key = frozenset(obs_mask)
    if key == _last_k_mask:
        return
    _last_k_mask = key

    k_ax.clear()
    k_ax.set_facecolor(SURFACE)
    for sp in k_ax.spines.values():
        sp.set_color('#30363d')
    k_ax.set_title(f'Kernel  k(xᵢ,xⱼ) = exp(−½ Δt²/{RBF_LENGTH_SCALE:.0f}²)',
                   fontsize=6, color=MUT_C, pad=3)

    if not obs_mask:
        k_ax.text(0.5, 0.5, 'no observations yet', transform=k_ax.transAxes,
                  ha='center', va='center', color=MUT_C, fontsize=7, style='italic')
        k_ax.set_xticks([])
        k_ax.set_yticks([])
        return

    ox    = obsX[sorted(obs_mask)]
    dists = np.subtract.outer(ox, ox) / RBF_LENGTH_SCALE
    K     = np.exp(-0.5 * dists**2)          # normalised RBF: diagonal = 1
    weeks = [f'W{int(w)}' for w in ox]

    im = k_ax.imshow(K, vmin=0, vmax=1, cmap='YlGn',
                     aspect='auto', interpolation='nearest')
    k_ax.set_xticks(range(len(ox)))
    k_ax.set_xticklabels(weeks, fontsize=5.5, rotation=45, ha='right', color=MUT_C)
    k_ax.set_yticks(range(len(ox)))
    k_ax.set_yticklabels(weeks, fontsize=5.5, color=MUT_C)
    k_ax.tick_params(colors=MUT_C, length=2)

    # Annotate each cell with its value
    for i in range(len(ox)):
        for j in range(len(ox)):
            k_ax.text(j, i, f'{K[i,j]:.2f}', ha='center', va='center',
                      fontsize=4.5 if len(ox) > 5 else 6,
                      color='#0d1117' if K[i,j] > 0.5 else MUT_C)

# ── Per-frame render function ──────────────────────────────────────────────────
def render_frame(frame_info):
    global ci_band, qry_errbar

    si    = frame_info['scene']
    t     = frame_info.get('t', 1.0)
    trans = frame_info.get('transitioning', False)
    scene = scenes[si]

    if trans:
        nsi    = frame_info['next_scene']
        nscene = scenes[nsi]
        mu_a, sig_a = scene_cache[si]
        mu_b, sig_b = scene_cache[nsi]
        mu       = lerp(mu_a,  mu_b,  t)
        sig      = lerp(sig_a, sig_b, t)
        show_wfh      = scene['show_wfh']      or (t > 0.5 and nscene['show_wfh'])
        show_omi      = scene['show_omi']      or (t > 0.5 and nscene['show_omi'])
        show_vax      = scene['show_vax']      or (t > 0.5 and nscene['show_vax'])
        show_mask_off = scene['show_mask_off'] or (t > 0.5 and nscene['show_mask_off'])
        show_school   = scene['show_school']   or (t > 0.5 and nscene['show_school'])
        obs_mask = set(nscene['obs_revealed'] if t > 0.5 else scene['obs_revealed'])
        query    = nscene['query']  if t > 0.5 else scene['query']
        is_final = nscene.get('final', False) and t > 0.5
        title    = nscene['title'] if t > 0.5 else scene['title']
        desc     = nscene['desc']  if t > 0.5 else scene['desc']
        samp_a, samp_b = sample_cache[si], sample_cache[nsi]
        for sl, sa, sb in zip(sample_lines, samp_a, samp_b):
            sl.set_data(xs_full, np.maximum(0, lerp(sa, sb, t)))
    else:
        mu, sig  = scene_cache[si]
        show_wfh      = scene['show_wfh']
        show_omi      = scene['show_omi']
        show_vax      = scene['show_vax']
        show_mask_off = scene['show_mask_off']
        show_school   = scene['show_school']
        obs_mask = set(scene['obs_revealed'])
        query    = scene.get('query')
        is_final = scene.get('final', False)
        title    = scene['title']
        desc     = scene['desc']
        for sl, samp in zip(sample_lines, sample_cache[si]):
            sl.set_data(xs_full, np.maximum(0, samp))

    ci_band.remove()
    ci_band = ax.fill_between(xs_full,
                               mu - CI_MULTIPLIER * sig,
                               mu + CI_MULTIPLIER * sig,
                               color=CI_C, alpha=0.14, zorder=3)
    mean_line.set_data(xs_full, mu)

    rx = obsX[list(obs_mask)] if obs_mask else np.array([])
    ry = obsY[list(obs_mask)] if obs_mask else np.array([])
    obs_scatter.set_offsets(np.c_[rx, ry] if len(rx) else np.empty((0, 2)))

    wfh_alpha    = 0.75 if show_wfh      else 0.0
    omi_alpha    = 0.75 if show_omi      else 0.0
    vax_alpha    = 0.75 if show_vax      else 0.0
    mask_alpha   = 0.75 if show_mask_off else 0.0
    school_alpha = 0.75 if show_school   else 0.0
    wfh_line.set_alpha(wfh_alpha);       wfh_label.set_alpha(wfh_alpha)
    omi_line.set_alpha(omi_alpha);       omi_label.set_alpha(omi_alpha)
    vax_line.set_alpha(vax_alpha);       vax_label.set_alpha(vax_alpha)
    mask_line.set_alpha(mask_alpha);     mask_label.set_alpha(mask_alpha)
    school_line.set_alpha(school_alpha); school_label.set_alpha(school_alpha)

    fore_shade.set_alpha(0.06 if is_final else 0.0)

    if query is not None:
        qmu, qsig = gp_predict(
            np.array([float(query)]),
            np.array([i in obs_mask for i in range(n)])
        )
        qmu, qsig = float(qmu[0]), float(qsig[0])
        half = CI_MULTIPLIER * qsig
        qry_line.set_xdata([query, query])
        qry_line.set_alpha(0.6)
        remove_errbar(qry_errbar)
        qry_errbar = ax.errorbar(query, qmu, yerr=half, fmt='o', color=QRY_C,
                                  capsize=6, capthick=1.8, elinewidth=2.2,
                                  markersize=8, zorder=7)
        qry_label.set_position((query, qmu + half + 55))
        qry_label.set_text(f'W{query}: ~{int(qmu)}±{int(half)}')
        qry_label.set_alpha(1.0)
    else:
        qry_line.set_alpha(0.0)
        try:
            remove_errbar(qry_errbar)
        except Exception:
            pass
        qry_errbar = ax.errorbar([], [], yerr=[], fmt='o', color=QRY_C,
                                  capsize=5, capthick=1.5, elinewidth=2,
                                  markersize=7, zorder=7)
        qry_label.set_alpha(0.0)

    title_txt.set_text(title)
    desc_txt.set_text(desc)
    if SHOW_KERNEL_MATRIX:
        update_k_matrix(obs_mask)

# ── Render ─────────────────────────────────────────────────────────────────────
writer = FFMpegWriter(fps=FPS, bitrate=2500,
                      extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])

with writer.saving(fig, OUT_PATH, dpi=DPI):
    for frame_info in tqdm(frames, desc=f"Rendering → {OUT_PATH}", unit="frame"):
        render_frame(frame_info)
        writer.grab_frame()

plt.close()
print(f"Done — saved to {OUT_PATH}")
