# GP COVID Animation

Renders an animated MP4 walking through every step of **Gaussian Process regression**
using a COVID-19 outbreak as the running example.

## Scenes

| # | Scene | What you see |
|---|-------|-------------|
| 1 | Prior | Wide uncertainty band — any smooth curve is plausible |
| 2 | Early wave | First 3 observations arrive; posterior snaps in |
| 3 | Vaccine rollout | Blue marker at W16; posterior curves downward |
| 4 | Omicron detected | Purple marker at W34; Omicron peak observation added |
| 5 | Post-Omicron decline | Full 7-observation arc visible |
| 6 | Interpolation (W25) | Query between events → tight confidence interval |
| 7 | Extrapolation (W50) | Query beyond all data → wide confidence interval |
| 8 | Final posterior | Forecast region shaded; full 52-week view |

## Quick start

### Prerequisites
Install [pixi](https://prefix.dev/docs/pixi/overview):
```
curl -fsSL https://pixi.sh/install.sh | bash
```

### Run
```bash
pixi run render
```

This installs all dependencies into an isolated environment and writes
`gp_covid.mp4` into the current directory (~1.5 MB, ~23 s at 30 fps).

### Or run directly (if you have Python + ffmpeg already)
```bash
pip install numpy matplotlib
python gp_covid_anim.py
```

## Project structure

```
gp_covid_pixi/
├── pixi.toml          # environment + task definitions
├── gp_covid_anim.py   # animation script
└── README.md
```

## Key parameters (top of `gp_covid_anim.py`)

| Variable | Default | Effect |
|----------|---------|--------|
| `FPS` | 30 | Output frame rate |
| `TRANSITION` | 25 | Morph frames between scenes |
| `l` (kernel) | 7.0 weeks | Smoothness / correlation length |
| `sf` (kernel) | 350 cases | Prior signal amplitude |
| `NOISE` | 25 cases | Observation noise std |
| `dpi` | 120 | Output resolution |
