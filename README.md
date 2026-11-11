# LieSolver experiments in 2D and 3D

Reproducible wave-equation experiments with **two or three spatial dimensions plus time**, comparing LieSolver with classical finite differences. This repository includes code, configurations, models, numerical results, and figures. These are independent experimental extensions, not reproductions of published 2D/3D results from the authors.

## Provenance and acknowledgments

This work **uses the original LieSolver code** by René P. Klausen and collaborators:

- Original repository: https://github.com/oduwancheekee/liesolver
- Paper: *LieSolver: A PDE-constrained solver for IBVPs using Lie symmetries*, https://arxiv.org/abs/2510.25731
- Version used: `178c4142598ea9cd6398a6180773becea48a17a2`.

`upstream/liesolver/` contains the unmodified original Python library, its installation files, and its Clear BSD license. `SOURCE.json` records provenance and file hashes. No additional repository clone is needed. Original notices are preserved, including older anonymous-review text in README/pyproject; the upstream `LICENSE` file contains the redistribution terms. The third-party license does not automatically apply to this repository's experimental code.

## Structure and code

- `configs_2d/`, `configs_3d/`: domains, initial modes, brick families, parameter bounds, seeds, training, classical discretization, and evaluation.
- `pipeline/wave2d.py`, `wave3d.py`: initial and boundary conditions, Halton sampling, Lie transformations, and analytical references.
- `upstream/liesolver/liesolver/model.py`: `Transformation`, `BrickFamily`, `Brick`, candidate generation, selection, and original optimization. The library also retains its original auxiliary modules; no 1D or PINN experiments or results are included.
- `pipeline/solvers.py`: LieSolver adapter, brick addition/refinement loop, and training history.
- `pipeline/classical2d.py`, `classical3d.py`: classical integrators and solution storage.
- `pipeline/artifacts*.py`, `comparison.py`: evaluation, numerical exports, and figure generation from exported data.
- `run.py`: execution, comparison, and plot regeneration.
- `tests/`: symmetry checks, classical convergence, CFL validation, export, and reload tests; also checks that the solvers do not use the exact reference to solve the problem.
- `results/`: eight runs, three comparisons, and the combined LaTeX report. See the [results index](results/README.md).

## Installation

Python >=3.12 is required; the archived runs used Python 3.14.6. Run these commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

`requirements-validated.txt` pins numerical and plotting dependencies from the original environment, excluding unrelated CUDA/PINN packages. To attempt to reproduce that exact environment, use `python -m pip install -r requirements-validated.txt`. Each run's `metadata.json` also preserves the complete environment package versions. Runtimes depend on hardware and software versions; bitwise agreement across platforms is not guaranteed.

## Reproducing the experiments

```bash
python run.py run configs_2d/wave_2d_sine_4pi.yaml --method liesolver
python run.py run configs_2d/wave_2d_sine_4pi.yaml --method classical
python run.py run configs_3d/wave_3d_sine_4pi.yaml --method liesolver
python run.py run configs_3d/wave_3d_sine_4pi.yaml --method classical
```

Each run creates a new folder containing the date, case, method, seed, and unique identifier. To run these four cases sequentially and automatically compare each pair:

```bash
python reproduce.py
```

The preliminary 2D case uses `configs_2d/wave_2d_sine.yaml`. Preliminary 3D runs share the final runs' configuration, but their runtimes were measured while other work was running concurrently; they are excluded from the report. The classical 3D simulation uses 201³ nodes: allow several GB of available RAM.

To regenerate figures or compare two runs with the same dimensionality:

```bash
python run.py plot results/RUN_ID
python run.py compare results/LIESOLVER_RUN_ID results/CLASSICAL_RUN_ID
```

## Physical problem and evaluation

We solve `u_tt = sum_j u_xjxj`, with unit wave speed and dimensionless coordinates, on `[0,1]^d × [0,1]`, where `d=2,3`. The main initial displacement is the product of `sin(4*pi*x_j)`, initial velocity is zero, and every spatial face has a homogeneous Dirichlet condition. The reference solution is that product multiplied by `cos(4*pi*sqrt(d)*t)`.

The classical method uses second-order centered differences, five-/seven-point Laplacians, a Taylor start, and a CFL setting of 0.8: 201 nodes per spatial axis and 354/434 time steps. In 3D, rolling time levels are maintained and sampled fields are exported; the entire fine space-time grid is not stored. Away from evaluation points, the 3D checkpoint interpolates the saved fields and therefore does not recover the entire original fine-grid solution. It retains the fine initial field for initial-condition evaluation.

Both methods are evaluated on the same grid within each dimension: 51²×41 points in 2D and 31³×41 in 3D. The relative L2 error excludes all spatial and temporal boundaries. Reported runtimes measure fitting or integration; they exclude plotting/final evaluation and, for the classical 3D solver, separately timed sampling.

## Bricks and randomness

Families are explicit expressions in the YAML `bricks` field; **symbolic families are not generated randomly**. The starting solutions are `sin(x)*sin(y)*cos(sqrt(2)*t)` in 2D and `sin(x)*sin(y)*sin(z)*cos(sqrt(3)*t)` in 3D. Spatial translations and simultaneous space-time scaling preserve the PDE. Translation bounds are ±pi and scale bounds are 0.5–20 in the main cases.

Each iteration samples 1000 parameter candidates per family and selects the candidate most correlated with the initial/boundary-condition residual. Global refinement follows, with up to 40 evaluations. The runs use NumPy seed 0, an MSE target of 1e-8, and at most 20 bricks. The final models contain 1 brick in 2D and 19 in 3D.

In this upstream version, `add_best_brick` calls `sample_params(pool_size, self.sobol_seed)`: the second argument occupies the `sampler` parameter, so the integer seed selects the NumPy random branch, **not Sobol**. This behavior is preserved to reproduce the results. Sampling follows the uniform/logarithmic type defined by each transformation. Initial/boundary-condition points use Halton sampling: 1000 points per surface, totaling 6000/8000 points.

## Changing the problem

Copy a YAML file and modify `problem.modes` (`mx`, `my`, and `mz` in 3D, with `amplitude`), `data.range_dim`, `seed`, `bricks`, and/or the `classical` grid. The current problem classes support finite sums of rectangular sine modes, zero initial velocity, homogeneous boundaries, initial time zero, and unit wave speed. Other conditions require extending the problem class and reference solution.

The current families have a single common scale: not all modes with different frequencies along each axis are exactly representable. When adding problems, provide base solutions and transformations valid for the PDE; the exact reference is used to measure error, not for training. YAML files are trusted local inputs interpreted with SymPy.

## Results and limitations

The main report is [report.tex](results/report_wave_2d_3d_20261110_2b9eddf1/report.tex), with a table, description, unrounded CSV values, and source records. LieSolver achieved relative errors of 3.997e-4 (2D) and 8.234e-4 (3D); the classical solver achieved 6.121e-4 and 1.404e-3. The classical 3D result exceeds the demonstration threshold of 1e-3, so its `acceptance_passed` is false; the run is complete. These are single-seed results for modes contained in the solution family; runtime differences do not constitute a controlled scalability measurement.

Original NPZ files and numerical values are preserved. Only absolute paths in text files were converted to relative paths for portability. `results/SHA256SUMS` supports integrity checks of the published archive. Packaging removes experimental 1D/PINN branches from the launcher and adapters and reads the original version from `UPSTREAM_COMMIT`, without requiring a nested Git repository. The 2D/3D algorithms are preserved.
