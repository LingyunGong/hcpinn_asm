# hcpinn_asm

Hard-constrained physics-informed neural network (PINN) for plasma etching
profile simulation. The network represents the level-set function of the
evolving etch front; the initial/boundary conditions defined by the layout
(mask opening and frozen mask band) are enforced **analytically** through a
hard-constraint transformation, so no IC/BC penalty weights are needed. The
physics enters through a deterministic, differentiable rate model based on
flux integration (direct + reflected ions, neutral diffusion) instead of
labeled data.

This repository accompanies the manuscript *"A Hard-Constrained
Physics-Informed Neural Network Approach for High-Efficiency Etching Process
Modeling in Advanced IC Manufacturing"*.

## Repository structure

```
├── main.py                    # 3D training entry point (production pipeline)
├── load_model.py              # load the pretrained 3D model and visualize
├── main_2D.py                 # 2D trench etching (self-contained production code)
├── rate_precompute.py         # rate model: flux integration & precomputed flux field
├── ablation_main.py           # hard vs. soft constraint ablation (paper Table 5)
├── ablation_2d.py             # hard vs. soft ablation on the 2D trench case
├── soft_failure_analysis.py   # diagnostic figures for soft-constraint failure modes
├── config/default_config.py   # training / geometry configuration
├── models/                    # network definitions + pretrained checkpoints
│   ├── neural_siren.py        #   SpaceTimeSIREN with the hard-constraint forward pass
│   ├── etching_models.py      #   differentiable etching rate model
│   ├── final_model.pth        #   pretrained 3D hole model (production checkpoint)
│   └── history models/        #   earlier trench/hole checkpoints
├── training/                  # loss functions and trainer
├── utils/geometry.py          # initial interface / mask geometry helpers
└── visualization/             # surface extraction and plotting
```

## Environment

- Python ≥ 3.9, PyTorch ≥ 1.12 (tested with 2.x, CUDA optional — the small
  networks train fast on CPU as well), NumPy, SciPy, Matplotlib, Plotly.
- No dataset is required: the rate model supplies the physics analytically.

## Quick start

**1. Pretrained 3D model (8:1 aspect-ratio hole)**

```bash
python load_model.py          # loads models/final_model.pth and visualizes profiles
```

**2. Train the 3D model from scratch**

```bash
python main.py
```

**3. 2D trench case**

```bash
python main_2D.py
```

**4. Constraint ablation (paper Table 5)**

The ablation trains the hard-constrained variant and soft-constrained
variants (IC/BC penalty weight `lambda = 1, 10, 100`) under an identical
configuration (same architecture, collocation sampling, rate model,
optimizer, and epoch budget; fixed seed) and compares the etched profiles:

```bash
python ablation_main.py --variant hard --outdir out_hard
python ablation_main.py --variant soft --soft-lam 1   --outdir out_soft1
python ablation_main.py --variant soft --soft-lam 10  --outdir out_soft10
python ablation_main.py --variant soft --soft-lam 100 --outdir out_soft100
```

Each run saves the loss history, wall-clock timing, the final model, and
level-set slices (`*.npz`). Diagnose the failure modes of the soft variants
(phi heat maps, |grad phi| fields, zero-crossing topology of spurious
contours, loss curves — the soft run with the *lowest* training loss is the
one that leaks into the mask band):

```bash
python soft_failure_analysis.py out_soft1 <production_or_hard_npz_dir> fig.png summary.json
```

A 2D version of the same ablation, built directly on `main_2D.py`, is
available via `ablation_2d.py`.

## Rate-model parameters (paper Table 1)

The rate model (Eq. (2) in the paper, a Langmuir–Hinshelwood-type synergistic
model) is implemented in `rate_precompute.py`. All quantities are in reduced
model units and map one-to-one onto the paper's Table 1:

| Code | Paper symbol | Nominal (range) | Role |
|---|---|---|---|
| `config.rate` (`rate_0`) | $k_i$, $k_{i_0}$ | 8 (4–16) | ion-channel rate constants |
| `config.sigma` | $\sigma$ | 0.02 (0.01–0.04) | ion angular-distribution width |
| `ratio` | $\rho$ | 0.002 (≤ 0.01) | reflected-ion mixing weight |
| `32` in the saturation term | $k_a$ | 32 (16–64) | neutral adsorption rate constant |
| `R0` in `neutral_etch_rate` | $k_{n_0}$ | 0.5 (0.25–1.0) | spontaneous chemical rate constant |
| `D` in `neutral_etch_rate` | $D$ | 0.15 (0.08–0.30) | neutral diffusivity |
| `ks` in `neutral_etch_rate` | $k$ | 0.16 (0.08–0.32) | first-order loss rate constant |
| `coefficient` | $\gamma$ | 0.8 | unit conversion (volume per reaction) |
| — | $k_d$ | ≈ 0 (neglected) | thermal desorption ($\ll k_a J_n$) |

Geometry (model units): mask opening radius `config.radius = 0.25`, mask
height `mask_h = 2 * radius * config.h = 0.8`; neutral penetration length
`l = sqrt(D * w0 / (2 * k))` with opening width `w0 = 0.5`.
