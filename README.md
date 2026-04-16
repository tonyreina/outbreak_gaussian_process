# MOVID Forecasting Example

Renders an animated MP4 walking through every step of **Gaussian Process regression**
using a stylised COVID-19 outbreak modelled on **Maryland's first year (Mar 2020 – Mar 2021)**.

![Final posterior frame](preview.png)

---

## What is Gaussian Process Regression?

A **Gaussian Process (GP)** is a probability distribution over functions.
Instead of fitting a single curve to data, a GP keeps track of *every* plausible curve
simultaneously and reports how confident it is at each point.

Formally, a GP is specified by:

- **Mean function** `m(x)` — the expected value of the function at input `x` (often set to 0 or a constant baseline).
- **Covariance (kernel) function** `k(x, x')` — encodes how correlated the function values at `x` and `x'` are.
  A large `k(x, x')` means "if the epidemic is high at week `x`, it is likely also high at nearby week `x'`."

This animation uses the **RBF (Radial Basis Function) kernel**, also called the squared-exponential kernel:

```
k(x, x') = σ² · exp( −(x − x')² / (2ℓ²) )
```

where `σ` controls the overall amplitude and `ℓ` is the **length-scale** (how quickly the function
can change — a longer length-scale produces smoother curves).

### The Bayesian view

GP regression is a direct application of **Bayesian inference**:

| Concept | Bayesian term | What you see in the animation |
|---------|--------------|-------------------------------|
| Belief before seeing data | **Prior** | Wide shaded band — all smooth epidemic curves are plausible |
| Data (observed case counts) | **Likelihood** | Orange dots added one scene at a time |
| Updated belief after data | **Posterior** | Narrowed band that passes through the observations |

The update rule is Bayes' theorem:

```
P(function | data)  ∝  P(data | function) · P(function)
         ↑                     ↑                   ↑
      posterior             likelihood            prior
```

Because both the prior and the likelihood are Gaussian, the posterior is also Gaussian —
and it has a **closed-form analytical solution**. This is what makes GP regression tractable
without Monte Carlo sampling.

### Prior → Posterior in closed form

Given `n` training points `X` with noisy observations `y = f(X) + ε`, `ε ~ N(0, σ_n² I)`,
the posterior predictive distribution at new inputs `X*` is:

```
μ*(X*)  = K(X*, X) · [K(X, X) + σ_n² I]⁻¹ · y
Σ*(X*)  = K(X*, X*) − K(X*, X) · [K(X, X) + σ_n² I]⁻¹ · K(X, X*)
```

where `K(A, B)` denotes the matrix of kernel evaluations between rows of `A` and `B`.
The diagonal of `Σ*` gives the **pointwise variance** — the width of the confidence band.
- **Near observations** the variance collapses toward zero (the model is certain).
- **Far from observations** the variance reverts toward the prior (the model is uncertain).

---

## scikit-learn implementation

This project uses **scikit-learn's `GaussianProcessRegressor`** (`sklearn.gaussian_process`).
The kernel is constructed as a product of two components:

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

kernel = (ConstantKernel(constant_value=RBF_SIGNAL_STD**2,
                         constant_value_bounds='fixed') *
          RBF(length_scale=RBF_LENGTH_SCALE,
              length_scale_bounds='fixed'))

gpr = GaussianProcessRegressor(
    kernel=kernel,
    alpha=NOISE**2,      # observation noise variance added to diagonal
    optimizer=None,      # fix hyperparameters — no MLE optimisation
    normalize_y=False,
)
```

Key design choices:

| Choice | Reason |
|--------|--------|
| `optimizer=None` | Hyperparameters are set manually from domain knowledge; no log-marginal-likelihood optimisation is needed for the animation |
| `alpha=NOISE**2` | Adds `σ_n² I` to the training kernel matrix — handles measurement noise and prevents numerical blow-up |
| `return_cov=True` for sampling | Returns the full posterior covariance matrix so we can draw self-consistent sample paths via `multivariate_normal` |
| Eigenvalue clamping | `np.linalg.eigh` + `np.maximum(eigenvalues, 0)` corrects any tiny negative eigenvalues from floating-point round-off before drawing samples |
| Mean centering | Observations are shifted by `PRIOR_MEAN` before fitting and shifted back after prediction, keeping the GP prior mean aligned with the baseline epidemic level |

---

## Historical basis

Case counts and event dates are derived from Maryland's actual COVID-19 record.
Week 1 = March 5, 2020 (first confirmed Maryland case); Week 52 = March 4, 2021.

| Week | Approx. date | Event / observation |
|------|-------------|---------------------|
| W2  | Mar 12 | First confirmed cases (~8/day) |
| W4  | Mar 26 | First wave climbing (~220/day); **Stay-at-home order** (Mar 23) |
| W7  | Apr 16 | First wave peak (~820/day); **Mask mandate issued** (Apr 15) |
| W11 | May 14 | Post-peak decline (~480/day); **Stage 1 reopen** (May 15) |
| W18 | Jul 2  | Summer plateau (~590/day) |
| W25 | Aug 20 | Late summer (~510/day) |
| W27 | Sep 4  | **Stage 3 reopen** (schools authorised to resume) |
| W32 | Oct 8  | Fall surge building (~950/day) |
| W39 | Nov 26 | Thanksgiving surge (~2,200/day) |
| W42 | Dec 17 | **Vaccine rollout begins** (first doses Dec 14) |
| W44 | Dec 31 | Winter peak (~3,800/day) |
| W47 | Jan 28 | Post-peak declining (~2,500/day) |

Hospital capacity lines:
- **ICU beds (1,200):** total licensed ICU beds statewide (MHCC FY2020)
- **Hospital surge threshold (2,500/day):** daily case rate at which Maryland hospitals were severely strained (Dec 2020)

---

## Scenes

| # | Scene | What you see |
|---|-------|-------------|
| 1 | Prior | Wide uncertainty band — any smooth curve is plausible |
| 2 | First cases + WFH | Seeding at W2; stay-at-home order appears at W4 |
| 3 | First wave peak + mask mandate | Sharp rise to W7; Stage 1 reopen at W11 |
| 4 | Summer plateau | Cases settle ~500–600/day; Stage 3 annotation |
| 5 | Fall / Thanksgiving surge | Rapid acceleration toward winter peak |
| 6 | Winter peak + vaccine rollout | ~3,800/day peak; vaccine line at W42 |
| 7 | Interpolation (W15) | Query between waves → narrow CI |
| 8 | Extrapolation (W52) | Query past last data → wide, honest uncertainty |
| 9 | Final posterior | Full year reconstruction; forecast region shaded |

The dashed white line shows the **simulated ground truth** — a smooth three-Gaussian
approximation of Maryland's spring, summer, and winter epidemic waves.

---

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
`outbreak_animation.mp4` into the current directory at 30 fps.

### Or run directly (if you have Python + ffmpeg already)
```bash
pip install numpy matplotlib scikit-learn tqdm
python movid_forecasting_example.py
```

---

## Project structure

```
outbreak_gaussian_process/
├── pixi.toml                     # environment + task definitions
├── movid_forecasting_example.py  # animation script
├── preview.png                   # final-frame preview (for this README)
└── README.md
```

---

## Key parameters (top of `movid_forecasting_example.py`)

All tunable values live in the `CONFIGURATION` block at the top of the script.

| Variable | Default | Effect |
|----------|---------|--------|
| `OBS` | 10-row array | Week + daily-case data points |
| `NOISE` | 150 cases | Observation noise std — increase to trust data less |
| `WFH_WEEK` | 4 | Week of the stay-at-home order annotation line |
| `VAX_WEEK` | 42 | Week of the vaccine rollout annotation line |
| `MASK_WEEK` | 7 | Week of the mask mandate annotation line |
| `SCHOOL_WEEK` | 27 | Week of Stage 3 / school reopening annotation line |
| `HOSPITAL_CAPACITY` | 2500 | Daily-case surge threshold (horizontal reference line) |
| `ICU_CAPACITY` | 1200 | Licensed ICU bed count (horizontal reference line) |
| `RBF_LENGTH_SCALE` | 6.0 weeks | GP smoothness / correlation length |
| `RBF_SIGNAL_STD` | 2000 cases | GP prior amplitude |
| `PRIOR_MEAN` | 1000 cases | Constant mean function baseline |
| `N_SAMPLES` | 5 | Number of posterior sample paths drawn |
| `CI_MULTIPLIER` | 1.96 | Confidence band z-score (1.96 → 95 % CI) |
| `FPS` | 30 | Output frame rate |
| `TRANSITION` | 25 | Morph frames between scenes |
| `DPI` | 120 | Output resolution |

---

## Data sources

Historical case counts and event dates are approximations derived from the following public sources:

- **Maryland Department of Health — COVID-19 Data Dashboard**
  https://health.maryland.gov/covid/Pages/Maryland-COVID-19-Data.aspx

- **COVID-19 pandemic in Maryland — Wikipedia**
  https://en.wikipedia.org/wiki/COVID-19_pandemic_in_Maryland

- **MarylandReporter — largest one-day case spike (Apr 9, 2020)**
  https://marylandreporter.com/2020/04/09/maryland-has-its-largest-one-day-rise-in-covid-19-cases/

- **USAFacts — Maryland COVID-19 overview (Omicron peak ~13,400/day Jan 2022)**
  https://usafacts.org/answers/how-did-covid-19-affect-people-in-the-us/state/maryland/

- **Maryland Health Care Commission — Licensed Acute Care Beds FY2020 (ICU bed count)**
  https://mhcc.maryland.gov/mhcc/pages/hcfs/hcfs_hospital/documents/acute_care/chcf_Licensed_Acute_Care_Beds_by_Hospital_and_Service_%20Maryland_FY2020.pdf

- **Baltimore Sun — Maryland ICU bed capacity at pandemic onset (Mar 2020)**
  https://www.baltimoresun.com/coronavirus/bs-hs-coronavirus-icu-beds-20200313-sg7g7zvkkzhy5ncog5dit53tgm-story.html

- **Washington Post — Maryland hospitals near capacity (Dec 2021)**
  https://www.washingtonpost.com/dc-md-va/2021/12/15/maryland-hospitals-coronavirus-capacity/
