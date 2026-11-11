"""Five-point spatial stencil and centered time stepping for u_tt=u_xx+u_yy."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .problems import evaluate


class Classical2DAdapter:
    def __init__(self,cfg,problem):
        self.cfg,self.problem=cfg,problem
        spec=cfg.get('classical',{})
        self.nx,self.ny=spec.get('nx',101),spec.get('ny',101)
        if any(isinstance(n,bool) or not isinstance(n,int) or n<4 for n in [self.nx,self.ny]):
            raise ValueError('nx and ny must be integers >=4.')
        self.dx=np.diff(problem.bounds[0])[0]/(self.nx-1)
        self.dy=np.diff(problem.bounds[1])[0]/(self.ny-1)
        cfl=float(spec.get('cfl',.8))
        if not 0<cfl<=1: raise ValueError('cfl must be in (0,1].')
        limit=1/np.sqrt(self.dx**-2+self.dy**-2)
        requested=float(spec.get('dt',cfl*limit))
        if not np.isfinite(requested) or requested<=0: raise ValueError('dt must be positive and finite.')
        duration=np.diff(problem.bounds[2])[0]
        self.steps=max(2,int(np.ceil(duration/requested)))
        self.dt=duration/self.steps
        if self.dt>limit*(1+1e-12): raise ValueError('Unstable 2D wave grid: dt²(1/dx²+1/dy²)>1.')
        cells=(self.steps+1)*self.nx*self.ny
        if cells>50_000_000: raise ValueError('Grid exceeds 50 million stored values.')
        self.info=dict(scheme='centered-wave-2d',nx=self.nx,ny=self.ny,time_steps=self.steps,
                       dx=float(self.dx),dy=float(self.dy),dt=float(self.dt),
                       cfl_2d=float(self.dt/limit),stored_values=cells,interpolation='trilinear')

    def fit(self,out):
        start=time.perf_counter(); p=self.problem
        self.x=np.linspace(*p.bounds[0],self.nx);self.y=np.linspace(*p.bounds[1],self.ny)
        self.t=np.linspace(*p.bounds[2],self.steps+1)
        self.u=np.zeros((self.steps+1,self.ny,self.nx))
        X,Y=np.meshgrid(self.x,self.y,indexing='xy')
        points=np.column_stack([X.ravel(),Y.ravel(),np.zeros(X.size)])
        self.u[0]=evaluate(p.u0,p.coords,points).reshape(self.ny,self.nx)
        self.u[0,[0,-1],:]=0;self.u[0,:,[0,-1]]=0
        def lap(a):
            return ((a[1:-1,2:]-2*a[1:-1,1:-1]+a[1:-1,:-2])/self.dx**2
                    +(a[2:,1:-1]-2*a[1:-1,1:-1]+a[:-2,1:-1])/self.dy**2)
        # Taylor start, with zero initial velocity imposed by the problem definition.
        self.u[1,1:-1,1:-1]=self.u[0,1:-1,1:-1]+.5*self.dt**2*lap(self.u[0])
        for n in range(1,self.steps):
            self.u[n+1,1:-1,1:-1]=2*self.u[n,1:-1,1:-1]-self.u[n-1,1:-1,1:-1]+self.dt**2*lap(self.u[n])
        if not np.isfinite(self.u).all(): raise FloatingPointError('Non-finite classical solution.')
        seconds=time.perf_counter()-start
        history=[dict(step=i,time=float(self.t[i]),max_abs_u=float(np.max(np.abs(self.u[i]))))
                 for i in range(self.steps+1)]
        out=Path(out)
        np.savez_compressed(out/'model_classical.npz',x=self.x,y=self.y,t=self.t,u=self.u)
        (out/'classical_grid.json').write_text(json.dumps(self.info,indent=2))
        self._setup_interpolator()
        return history,seconds

    def _setup_interpolator(self):
        self.interpolator=RegularGridInterpolator((self.t,self.y,self.x),self.u,bounds_error=True)

    def predict(self,points,derivative=None):
        if derivative is None or derivative==(0,0,0):
            return self.interpolator(points[:,[2,1,0]])
        if derivative==(0,0,1) and np.all(points[:,2]==0):
            self.interpolator(points[:,[2,1,0]])  # Validate the domain before returning imposed velocity.
            return np.zeros(len(points))
        raise ValueError('Only values and initial velocity are supported by the 2D classical adapter.')

    def restore(self,out):
        out=Path(out)
        with np.load(out/'model_classical.npz') as data:
            self.x,self.y,self.t,self.u=(data[k].copy() for k in ['x','y','t','u'])
        self.info=json.loads((out/'classical_grid.json').read_text())
        self._setup_interpolator()
