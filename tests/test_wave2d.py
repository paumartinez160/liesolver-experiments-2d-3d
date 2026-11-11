import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run
import numpy as np
import sympy as sp
import yaml
from pipeline.wave2d import Wave2DProblem, scale_spacetime, x,y,t
from pipeline.classical2d import Classical2DAdapter
from pipeline.solvers import LieAdapter
from pipeline.artifacts import export,write_csv,plots


class Wave2DTests(unittest.TestCase):
    def config(self):
        return yaml.safe_load((ROOT/'configs_2d/wave_2d_sine.yaml').read_text())

    def test_reference_and_symmetries(self):
        cfg=self.config();p=Wave2DProblem(cfg)
        points=np.array([[.2,.3,.4],[.6,.7,.8]])
        exact=np.sin(np.pi*points[:,0])*np.sin(np.pi*points[:,1])*np.cos(np.sqrt(2)*np.pi*points[:,2])
        np.testing.assert_allclose(p.reference(points),exact,atol=1e-14)
        a=sp.Symbol('a',positive=True)
        mapped=scale_spacetime(x*x+y*y+2*t*t,a)
        self.assertEqual(sp.simplify(sp.diff(mapped,t,2)-sp.diff(mapped,x,2)-sp.diff(mapped,y,2)),0)
        solver=LieAdapter(cfg,p)
        for family in solver.model.families:
            for params in family.sample_params(3,sobol_seed=7):
                r=family.eval(points,params,(0,0,2))-family.eval(points,params,(2,0,0))-family.eval(points,params,(0,2,0))
                np.testing.assert_allclose(r,0,atol=1e-11)
        self.assertEqual(p.train.constraints[0].x.shape[1],3)
        self.assertEqual(len(p.train.constraints),6)
        q=Wave2DProblem(cfg)
        np.testing.assert_array_equal(p.train.constraints[0].x,q.train.constraints[0].x)

    def test_classical_convergence_and_cfl(self):
        points=np.random.default_rng(42).uniform([.05,.05,.05],[.95,.95,.95],size=(300,3))
        errors=[]
        for n in [21,41]:
            cfg=self.config();cfg['classical'].update(nx=n,ny=n)
            p=Wave2DProblem(cfg);a=Classical2DAdapter(cfg,p)
            with tempfile.TemporaryDirectory() as tmp:
                with patch.object(p,'reference',side_effect=AssertionError('Reference used in solver')):
                    a.fit(Path(tmp))
                errors.append(np.linalg.norm(a.predict(points)-p.reference(points)))
                b=Classical2DAdapter(cfg,p);b.restore(Path(tmp))
                np.testing.assert_array_equal(a.predict(points),b.predict(points))
                np.testing.assert_array_equal(a.u[:,0,:],0)
                np.testing.assert_array_equal(a.u[:,:,-1],0)
        self.assertLess(errors[1],errors[0]/3)
        cfg['classical']['dt']=1.
        with self.assertRaisesRegex(ValueError,'Unstable'): Classical2DAdapter(cfg,p)

    def test_liesolver_fit_reload_export_without_reference_training(self):
        cfg=self.config();cfg['evaluation']['grid']=[21,21,11]
        p=Wave2DProblem(cfg);a=LieAdapter(cfg,p)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            with patch.object(p,'reference',side_effect=AssertionError('Reference used in training')):
                history,_=a.fit(out)
            expected=a.predict(p.test.constraints[0].x)
            a.model.load_from_npz(out,p.module.trafos,p.coords)
            np.testing.assert_array_equal(a.predict(p.test.constraints[0].x),expected)
            metrics=export(a,p,out)
            self.assertTrue(metrics['acceptance_passed'])
            write_csv(out/'history.csv',history);plots(out)
            with np.load(out/'domain.npz') as data:
                self.assertEqual(tuple(data['shape']),(11,21,21))
                np.testing.assert_array_equal(data['error'],data['prediction']-data['reference'])
            self.assertTrue((out/'domain.png').exists())

    def test_rectangular_mode_reference(self):
        cfg=self.config();cfg['data']['range_dim']['y']=[0,2]
        cfg['problem']['modes']=[dict(mx=2,my=1,amplitude=.7)]
        p=Wave2DProblem(cfg)
        residual=sp.diff(p.exact,t,2)-sp.diff(p.exact,x,2)-sp.diff(p.exact,y,2)
        self.assertEqual(sp.simplify(residual),0)
        cfg['classical'].update(nx=61,ny=41)
        a=Classical2DAdapter(cfg,p)
        with tempfile.TemporaryDirectory() as tmp: a.fit(Path(tmp))
        points=np.array([[.2,.4,.1],[.6,1.3,.7],[.8,1.8,.9]])
        np.testing.assert_allclose(a.predict(points),p.reference(points),atol=.003)

if __name__=='__main__': unittest.main()
