import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import run
import numpy as np
import sympy as sp
import yaml
from pipeline.wave3d import Wave3DProblem,x,y,z,t
from pipeline.classical3d import Classical3DAdapter
from pipeline.solvers import LieAdapter
from pipeline.artifacts import export,plots,write_csv


class Wave3DTests(unittest.TestCase):
    def cfg(self):
        c=yaml.safe_load((ROOT/'configs_3d/wave_3d_sine_4pi.yaml').read_text())
        c['problem']['modes'][0].update(mx=1,my=1,mz=1)
        c['classical'].update(nx=21,ny=21,nz=21)
        c['evaluation']['grid']=[9,9,9,9]
        return c

    def test_exact_reference_and_bricks(self):
        c=self.cfg();p=Wave3DProblem(c);a=LieAdapter(c,p)
        residual=sp.diff(p.exact,t,2)-sum(sp.diff(p.exact,v,2) for v in [x,y,z])
        self.assertEqual(sp.simplify(residual),0)
        points=np.random.default_rng(5).uniform(.05,.95,(30,4))
        np.testing.assert_allclose(p.reference(points),np.prod(np.sin(np.pi*points[:,:3]),axis=1)*np.cos(np.sqrt(3)*np.pi*points[:,3]),atol=1e-14)
        for family in a.model.families:
            params=family.sample_params(1)[0]
            r=family.eval(points,params,(0,0,0,2))-sum(family.eval(points,params,tuple(2 if j==i else 0 for j in range(4))) for i in range(3))
            np.testing.assert_allclose(r,0,atol=1e-10)
        self.assertEqual(len(p.train.constraints),8)
        np.testing.assert_array_equal(p.train.constraints[0].x,Wave3DProblem(c).train.constraints[0].x)

    def test_rolling_solver_convergence_reload_and_no_oracle(self):
        c=self.cfg();p=Wave3DProblem(c)
        grid=np.linspace(0,1,9)
        T,Z,Y,X=np.meshgrid(grid,grid,grid,grid,indexing='ij')
        points=np.column_stack([X.ravel(),Y.ravel(),Z.ravel(),T.ravel()])
        errors=[]
        for n in [21,41]:
            c['classical'].update(nx=n,ny=n,nz=n);a=Classical3DAdapter(c,p)
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)
                with patch.object(p,'reference',side_effect=AssertionError('Oracle used')): history,_=a.fit(out)
                errors.append(np.linalg.norm(a.predict(points)-p.reference(points)))
                b=Classical3DAdapter(c,p);b.restore(out)
                np.testing.assert_array_equal(a.predict(points),b.predict(points))
                self.assertEqual(a.samples.shape,(9,9,9,9))
                np.testing.assert_allclose(a.samples[:,0],0,atol=1e-14)
                export(a,p,out);write_csv(out/'history.csv',history);plots(out)
                self.assertTrue((out/'domain.png').exists())
        self.assertLess(errors[1],errors[0]/3)
        c['classical']['dt']=.5
        with self.assertRaisesRegex(ValueError,'CFL'): Classical3DAdapter(c,p)

    def test_liesolver_fit_without_oracle(self):
        c=self.cfg();c['fit'].update(mse_tol=1e-6,max_bricks=3)
        p=Wave3DProblem(c);a=LieAdapter(c,p)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(p,'reference',side_effect=AssertionError('Oracle used')): a.fit(Path(tmp))
            probe=np.random.default_rng(4).uniform(0,1,(100,4));before=a.predict(probe)
            a.model.load_from_npz(Path(tmp),p.module.trafos,p.coords)
            np.testing.assert_array_equal(before,a.predict(probe))
            self.assertLess(np.linalg.norm(a.predict(probe)-p.reference(probe))/np.linalg.norm(p.reference(probe)),.002)

if __name__=='__main__': unittest.main()
