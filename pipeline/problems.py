"""Shared symbolic evaluation helper used by the 2D and 3D problems."""
import numpy as np
import sympy as sp


def evaluate(expr, coords, points):
    value = sp.lambdify(coords, expr, 'numpy')(*points.T)
    return np.broadcast_to(value, (len(points),)).astype(float).copy()

