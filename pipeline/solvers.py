"""Adapter around the original LieSolver algorithm."""
import os
import time
import numpy as np
from liesolver.model import LieSolver, BrickFamily


def constraint_mse(predict, constraints):
    return float(np.mean(np.concatenate([(predict(c.x, c.deriv_order)-c.y)**2 for c in constraints])))


class LieAdapter:
    def __init__(self, cfg, problem):
        self.cfg, self.problem = cfg, problem
        np.random.seed(cfg["seed"])
        families = [BrickFamily.init_from_str(b, problem.module.trafos, problem.coords) for b in cfg['bricks']]
        self.model = LieSolver(families=families, constraints=problem.train, ridge=0.1, sobol_seed=cfg['seed'])

    def predict(self, x, derivative=None):
        if derivative is None: derivative=(0,)*len(self.problem.coords)
        return self.model.eval_bricks(self.model.bricks, x, derivative) @ self.model.amplitudes

    def fit(self, out):
        cfg = self.cfg['fit']
        history = []
        start = time.perf_counter()
        def record(action):
            history.append(dict(step=len(history), action=action, seconds=time.perf_counter()-start,
                bricks=len(self.model.bricks), parameters=sum(b.params.size+1 for b in self.model.bricks),
                train_mse=float(self.model.mse), test_mse=constraint_mse(self.predict, self.problem.test)))
            print(history[-1], flush=True)
        try:
            for i in range(1, cfg['max_bricks']+1):
                self.model.add_best_brick(pool_size=cfg['pool_size'])
                record('add')
                if i % cfg['batch_size'] == 0:
                    self.model.refine(max_nfev=cfg['nfev_batch'], active_idx=list(range(max(0,i-cfg['batch_size']),i)))
                    record('refine_batch')
                if i % cfg['global_every'] == 0 or i == cfg['max_bricks']:
                    self.model.refine(max_nfev=cfg['nfev_global'])
                    record('refine_all')
                if self.model.mse <= cfg['mse_tol']:
                    break
        finally:
            if self.model.bricks:
                self.model.save(out, filename='model_liesolver.npz')
        return history, time.perf_counter()-start
