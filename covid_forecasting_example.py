"""
Gaussian Process Regression — COVID-19 Outbreak Animation
==========================================================
Renders an 8-scene WebM that walks through every step of GP regression:

  1. Wide prior (no data)
  2. Early wave observations arrive
  3. Vaccine rollout (week 16) — posterior bends down
  4. Omicron variant detected (week 34) — surge observed
  5. Post-Omicron decline — full arc in view
  6. Interpolation query (week 25) — tight CI
  7. Extrapolation query (week 50) — wide CI
  8. Final annotated 52-week posterior

Output: gp_covid.webm  (written to the current working directory)

Run via pixi:
    pixi run render

Or directly:
    python gp_covid_anim.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.animation import FFMpegWriter

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these values to customise the animation
# ═══════════════════════════════════════════════════════════════════════════════

# ── Observations ───────────────────────────────────────────────────────────────
# Each row is [week, daily_cases].  Add, remove, or edit rows to change the data.
# The scene list below references observations by their 0-based row index.
OBS = np.array([
    [ 3,   80],   # 0  early seeding
    [ 7,  320],   # 1  first wave climbing
    [12,  590],   # 2  first wave peak
    [20,  310],   # 3  vaccine dip
    [30,  420],   # 4  pre-Omicron uptick
    [38, 1250],   # 5  Omicron peak
    [45,  480],   # 6  post-Omicron decline
], dtype=float)

NOISE    = 25.0   # observation noise std (cases/day) — increase to trust data less
VAC_WEEK = 16     # week of the vaccine rollout annotation line
OMI_WEEK = 34     # week of the Omicron detection annotation line

# ── GP kernel ─────────────────────────────────────────────────────────────────
RBF_LENGTH_SCALE = 7.0    # smoothness: larger → longer correlation length
RBF_SIGNAL_STD   = 350.0  # amplitude: scales the prior variance
PRIOR_MEAN       = 500.0  # prior mean used when no data has been observed yet
PRIOR_STD        = 420.0  # prior std  used when no data has been observed yet

# ── Output ────────────────────────────────────────────────────────────────────
CI_MULTIPLIER = 1.96   # z-score for the confidence band  (1.96 → 95 % CI)
FPS           = 30     # output frame rate
TRANSITION    = 25     # morph frames between scenes
OUT_PATH      = 'gp_covid.webm'
DPI           = 120

# ═══════════════════════════════════════════════════════════════════════════════

# ── Colour palette ─────────────────────────────────────────────────────────────
SURFACE = '#161b22'
OBS_C   = '#f0a500'
MEAN_C  = '#3fb950'
CI_C    = '#3fb950'
VAC_C   = '#a5d6ff'
OMI_C   = '#d2a8ff'
QRY_C   = '#ff7b72'
TXT_C   = '#e6edf3'
MUT_C   = '#8b949e'

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


def remove_errbar(eb):
    """Remove a matplotlib errorbar container and all of its child artists."""
    eb[0].remove()
    for artist in eb[1]: artist.remove()
    for artist in eb[2]: artist.remove()


# ── Scene definitions ──────────────────────────────────────────────────────────
# obs_revealed: row indices from OBS that are visible in this scene (0-based)
# query:        a week number to show a forecast marker at, or None
XMIN, XMAX = 1,  52
YMIN, YMAX = 0, 1500

scenes = [
    {
        'title':        'Step 1 · Prior — Before Any Data',
        'desc':         'GP prior: wide uncertainty, any smooth curve is plausible',
        'obs_revealed': [],
        'show_vac':     False,
        'show_omi':     False,
        'query':        None,
        'hold_frames':  60,
    },
    {
        'title':        'Step 2 · Observe: Early Wave',
        'desc':         'First reports arrive — case counts start climbing',
        'obs_revealed': [0, 1, 2],
        'show_vac':     False,
        'show_omi':     False,
        'query':        None,
        'hold_frames':  55,
    },
    {
        'title':        'Step 3 · Vaccine Rollout (Week 16)',
        'desc':         'Vaccine deployed — GP posterior bends downward',
        'obs_revealed': [0, 1, 2, 3],
        'show_vac':     True,
        'show_omi':     False,
        'query':        None,
        'hold_frames':  55,
    },
    {
        'title':        'Step 4 · Omicron Variant Detected (Week 34)',
        'desc':         'New variant emerges — cases surge far above first wave',
        'obs_revealed': [0, 1, 2, 3, 4, 5],
        'show_vac':     True,
        'show_omi':     True,
        'query':        None,
        'hold_frames':  65,
    },
    {
        'title':        'Step 5 · Post-Omicron Decline',
        'desc':         'Full arc observed — wave crests then falls',
        'obs_revealed': [0, 1, 2, 3, 4, 5, 6],
        'show_vac':     True,
        'show_omi':     True,
        'query':        None,
        'hold_frames':  55,
    },
    {
        'title':        'Step 6 · Predict: Interpolation (Week 25)',
        'desc':         'Between observations — GP is confident, narrow band',
        'obs_revealed': [0, 1, 2, 3, 4, 5, 6],
        'show_vac':     True,
        'show_omi':     True,
        'query':        25,
        'hold_frames':  60,
    },
    {
        'title':        'Step 7 · Predict: Extrapolation (Week 50)',
        'desc':         'Beyond all data — uncertainty grows, band fans out',
        'obs_revealed': [0, 1, 2, 3, 4, 5, 6],
        'show_vac':     True,
        'show_omi':     True,
        'query':        50,
        'hold_frames':  70,
    },
    {
        'title':        'Final Posterior — 52-Week Forecast',
        'desc':         'Confident reconstruction + honest uncertainty beyond last observation',
        'obs_revealed': [0, 1, 2, 3, 4, 5, 6],
        'show_vac':     True,
        'show_omi':     True,
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
fig, ax = plt.subplots(figsize=(11, 6), facecolor='none')
ax.set_facecolor('none')
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
ax.set_xticks([XMIN, 10, VAC_WEEK, 20, 30, OMI_WEEK, 38, 45, XMAX])
ax.set_xticklabels(
    [f'W{w}' for w in [XMIN, 10, VAC_WEEK, 20, 30, OMI_WEEK, 38, 45, XMAX]],
    fontsize=8,
)

# Persistent plot artists
ci_band     = ax.fill_between([], [], [], color=CI_C, alpha=0.15, zorder=2)
mean_line,  = ax.plot([], [], color=MEAN_C, linewidth=2.2, zorder=3)
obs_scatter = ax.scatter([], [], color=OBS_C, s=70, zorder=5,
                         edgecolors='none', linewidths=1.4)
vac_line    = ax.axvline(VAC_WEEK, color=VAC_C, linewidth=1.5, linestyle='--', alpha=0, zorder=4)
omi_line    = ax.axvline(OMI_WEEK, color=OMI_C, linewidth=1.5, linestyle='--', alpha=0, zorder=4)
vac_label   = ax.text(VAC_WEEK + 0.3, YMAX * 0.92, 'Vaccine',  color=VAC_C,
                      fontsize=8.5, fontweight='bold', alpha=0, zorder=6)
omi_label   = ax.text(OMI_WEEK + 0.3, YMAX * 0.92, 'Omicron', color=OMI_C,
                      fontsize=8.5, fontweight='bold', alpha=0, zorder=6)
qry_line    = ax.axvline(0, color=QRY_C, linewidth=1.2, linestyle=':', alpha=0, zorder=4)
qry_errbar  = ax.errorbar([], [], yerr=[], fmt='o', color=QRY_C,
                           capsize=5, capthick=1.5, elinewidth=2, markersize=7, zorder=6)
qry_label   = ax.text(0, 0, '', color=QRY_C, fontsize=9, ha='center', zorder=7,
                      fontfamily='monospace')
fore_shade  = ax.axvspan(obsX.max(), XMAX, alpha=0, color='white', zorder=1)
title_txt   = ax.text(0.5, 1.04, '', transform=ax.transAxes,
                      ha='center', va='bottom', color=TXT_C, fontsize=13, fontweight='bold')
desc_txt    = ax.text(0.5, -0.13, '', transform=ax.transAxes,
                      ha='center', va='top', color=MUT_C, fontsize=9.5, style='italic')

leg_elements = [
    mlines.Line2D([0],[0], color=MEAN_C, linewidth=2, label='GP mean'),
    mpatches.Patch(facecolor=CI_C, alpha=0.35, label='95% CI'),
    mlines.Line2D([0],[0], marker='o', color='w', markerfacecolor=OBS_C,
                  markersize=8, label='Reported cases'),
    mlines.Line2D([0],[0], color=VAC_C, linewidth=1.5, linestyle='--', label='Vaccine rollout'),
    mlines.Line2D([0],[0], color=OMI_C, linewidth=1.5, linestyle='--', label='Omicron detected'),
    mlines.Line2D([0],[0], marker='o', color='w', markerfacecolor=QRY_C,
                  markersize=7, label='Forecast query'),
]
ax.legend(handles=leg_elements, loc='upper left', fontsize=8,
          framealpha=0.3, facecolor=SURFACE, edgecolor='#30363d', labelcolor=TXT_C)

plt.tight_layout(rect=[0, 0.06, 1, 0.97])

# Cache GP predictions per scene to avoid recomputing every frame
xs_full = np.linspace(XMIN, XMAX, 300)
scene_cache = {}
for si, sc in enumerate(scenes):
    mask = np.array([i in sc['obs_revealed'] for i in range(n)])
    scene_cache[si] = gp_predict(xs_full, mask)

def lerp(a, b, t):
    return a * (1 - t) + b * t

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
        show_vac = scene['show_vac'] or (t > 0.5 and nscene['show_vac'])
        show_omi = scene['show_omi'] or (t > 0.5 and nscene['show_omi'])
        obs_mask = set(nscene['obs_revealed'] if t > 0.5 else scene['obs_revealed'])
        query    = nscene['query']  if t > 0.5 else scene['query']
        is_final = nscene.get('final', False) and t > 0.5
        title    = nscene['title'] if t > 0.5 else scene['title']
        desc     = nscene['desc']  if t > 0.5 else scene['desc']
    else:
        mu, sig  = scene_cache[si]
        show_vac = scene['show_vac']
        show_omi = scene['show_omi']
        obs_mask = set(scene['obs_revealed'])
        query    = scene.get('query')
        is_final = scene.get('final', False)
        title    = scene['title']
        desc     = scene['desc']

    ci_band.remove()
    ci_band = ax.fill_between(xs_full,
                               mu - CI_MULTIPLIER * sig,
                               mu + CI_MULTIPLIER * sig,
                               color=CI_C, alpha=0.14, zorder=2)
    mean_line.set_data(xs_full, mu)

    rx = obsX[list(obs_mask)] if obs_mask else np.array([])
    ry = obsY[list(obs_mask)] if obs_mask else np.array([])
    obs_scatter.set_offsets(np.c_[rx, ry] if len(rx) else np.empty((0, 2)))

    vac_alpha = 0.75 if show_vac else 0.0
    omi_alpha = 0.75 if show_omi else 0.0
    vac_line.set_alpha(vac_alpha);  vac_label.set_alpha(vac_alpha)
    omi_line.set_alpha(omi_alpha);  omi_label.set_alpha(omi_alpha)

    fore_shade.set_alpha(0.04 if is_final else 0.0)

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
                                  markersize=8, zorder=6)
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
                                  markersize=7, zorder=6)
        qry_label.set_alpha(0.0)

    title_txt.set_text(title)
    desc_txt.set_text(desc)

# ── Render ─────────────────────────────────────────────────────────────────────
print(f"Rendering {len(frames)} frames at {FPS} fps → {OUT_PATH}")

writer = FFMpegWriter(fps=FPS, bitrate=2500,
                      extra_args=['-vcodec', 'libvpx-vp9', '-pix_fmt', 'yuva420p'])

with writer.saving(fig, OUT_PATH, dpi=DPI):
    for fi, frame_info in enumerate(frames):
        render_frame(frame_info)
        writer.grab_frame()
        if fi % 50 == 0:
            print(f"  {fi}/{len(frames)} frames")

plt.close()
print(f"Done — saved to {OUT_PATH}")
