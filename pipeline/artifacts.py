"""Numerical exports are the sole input of all figures."""
import csv
import json
from pathlib import Path
import numpy as np


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plots(out):
    out = Path(out)
    with np.load(out / 'domain.npz') as data:
        dim = data['points'].shape[1]
    if dim == 3:
        from .artifacts2d import plots2d
        return plots2d(out)
    if dim == 4:
        from .artifacts3d import plots3d
        return plots3d(out)
    raise ValueError('Only 2D and 3D wave experiments are supported.')


def export(adapter, problem, out):
    if len(problem.coords) == 3:
        from .artifacts2d import export2d
        return export2d(adapter, problem, out)
    if len(problem.coords) == 4:
        from .artifacts3d import export3d
        return export3d(adapter, problem, out)
    raise ValueError('Only 2D and 3D wave experiments are supported.')
