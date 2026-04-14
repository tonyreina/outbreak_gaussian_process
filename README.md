# GP COVID Animation

Renders an animated WebM walking through every step of **Gaussian Process regression**
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
`gp_covid.webm` into the current directory (~30 fps, transparent background).

### Or run directly (if you have Python + ffmpeg already)
```bash
pip install numpy matplotlib
python covid_forecasting_example.py
```

## Project structure

```
outbreak_gaussian_process/
├── pixi.toml                    # environment + task definitions
├── covid_forecasting_example.py # animation script
└── README.md
```

## Key parameters (top of `covid_forecasting_example.py`)

All tunable values live in the `CONFIGURATION` block at the top of the script.

| Variable | Default | Effect |
|----------|---------|--------|
| `OBS` | 7-row array | Week + daily-case data points |
| `NOISE` | 25 cases | Observation noise std — increase to trust data less |
| `VAC_WEEK` | 16 | Week of the vaccine rollout annotation line |
| `OMI_WEEK` | 34 | Week of the Omicron detection annotation line |
| `RBF_LENGTH_SCALE` | 7.0 weeks | GP smoothness / correlation length |
| `RBF_SIGNAL_STD` | 350 cases | GP prior amplitude |
| `CI_MULTIPLIER` | 1.96 | Confidence band z-score (1.96 → 95 % CI) |
| `FPS` | 30 | Output frame rate |
| `TRANSITION` | 25 | Morph frames between scenes |
| `DPI` | 120 | Output resolution |
