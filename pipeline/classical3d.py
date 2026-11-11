"""Seven-point wave stencil with rolling time levels and sampled dense output."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .problems import evaluate


class Classical3DAdapter:
    def __init__(self,cfg,problem):
        self.cfg,self.problem=cfg,problem
        spec=cfg.get('classical',{});self.sizes=[spec.get(key,201) for key in ['nx','ny','nz']]
        if any(isinstance(n,bool) or not isinstance(n,int) or n<4 for n in self.sizes):
            raise ValueError('nx,ny,nz must be integers >=4.')
        self.spacing=np.diff(problem.bounds[:3],axis=1).ravel()/(np.array(self.sizes)-1)
        limit=1/np.sqrt(np.sum(self.spacing**-2));cfl=float(spec.get('cfl',.8))
        if not 0<cfl<=1: raise ValueError('cfl must be in (0,1].')
        requested=float(spec.get('dt',cfl*limit))
        if not np.isfinite(requested) or requested<=0: raise ValueError('Positive finite dt required.')
        self.steps=max(2,int(np.ceil(np.diff(problem.bounds[3])[0]/requested)))
        self.dt=np.diff(problem.bounds[3])[0]/self.steps
        if self.dt>limit*(1+1e-12): raise ValueError('Unstable 3D CFL: dt² sum(1/dx_i²)>1.')
        if np.prod(self.sizes)>12_000_000: raise ValueError('Spatial grid exceeds 12 million nodes.')
        self.eval_sizes=cfg['evaluation']['grid']
        if len(self.eval_sizes)!=4 or any(not isinstance(n,int) or n<3 for n in self.eval_sizes):
            raise ValueError('Evaluation grid must contain four integers >=3.')
        if np.prod(self.eval_sizes)>10_000_000: raise ValueError('Evaluation grid exceeds 10 million points.')
        self.info=dict(scheme='centered-wave-3d',nx=self.sizes[0],ny=self.sizes[1],nz=self.sizes[2],
                       time_steps=self.steps,dx=float(self.spacing[0]),dy=float(self.spacing[1]),dz=float(self.spacing[2]),
                       dt=float(self.dt),cfl_3d=float(self.dt/limit),storage='rolling levels plus sampled evaluation fields',
                       interpolation='trilinear in space, linear between native time levels; saved-grid interpolation on reload')

    def fit(self,out):
        start=time.perf_counter();sampling_seconds=0.;p=self.problem
        self.axes=[np.linspace(*p.bounds[i],self.sizes[i]) for i in range(3)]
        self.eval_axes=[np.linspace(*p.bounds[i],self.eval_sizes[i]) for i in range(4)]
        ex,ey,ez,et=self.eval_axes
        Z,Y,X=np.meshgrid(ez,ey,ex,indexing='ij')
        query=np.column_stack([Z.ravel(),Y.ravel(),X.ravel()])
        # Broadcast coordinate arrays: no full four-column input grid is allocated.
        import sympy as sp
        f=sp.lambdify(p.coords,p.u0,'numpy')
        xx,yy,zz=self.axes
        old=np.broadcast_to(f(xx[None,None,:],yy[None,:,None],zz[:,None,None],0.),(len(zz),len(yy),len(xx))).copy()
        def boundary(a):
            a[[0,-1],:,:]=0;a[:,[0,-1],:]=0;a[:,:,[0,-1]]=0
        boundary(old);self.initial=old.copy()
        current=np.zeros_like(old);new=np.zeros_like(old)
        core=(slice(1,-1),)*3
        dx,dy,dz=self.spacing
        def lap(a):
            return ((a[1:-1,1:-1,2:]-2*a[core]+a[1:-1,1:-1,:-2])/dx**2
                    +(a[1:-1,2:,1:-1]-2*a[core]+a[1:-1,:-2,1:-1])/dy**2
                    +(a[2:,1:-1,1:-1]-2*a[core]+a[:-2,1:-1,1:-1])/dz**2)
        self.samples=np.empty((len(et),len(ez),len(ey),len(ex)))
        def spatial_sample(a):
            return RegularGridInterpolator((zz,yy,xx),a,bounds_error=True)(query).reshape(len(ez),len(ey),len(ex))
        tic=time.perf_counter();self.samples[0]=spatial_sample(old);sampling_seconds+=time.perf_counter()-tic
        sample_index=1;history=[dict(step=0,time=0.,max_abs_u=float(np.max(np.abs(self.samples[0]))))]
        for n in range(self.steps):
            if n==0: current[core]=old[core]+.5*self.dt**2*lap(old)
            else:
                new[core]=2*current[core]-old[core]+self.dt**2*lap(current)
                old,current,new=current,new,old
            while sample_index<len(et) and et[sample_index]<=(n+1)*self.dt+1e-13:
                tic=time.perf_counter()
                weight=np.clip((et[sample_index]-n*self.dt)/self.dt,0,1)
                self.samples[sample_index]=(1-weight)*spatial_sample(old)+weight*spatial_sample(current)
                history.append(dict(step=n+1,time=float(et[sample_index]),max_abs_u=float(np.max(np.abs(self.samples[sample_index])))))
                sample_index+=1;sampling_seconds+=time.perf_counter()-tic
        if sample_index!=len(et) or not np.isfinite(self.samples).all(): raise FloatingPointError('Incomplete/non-finite samples.')
        seconds=time.perf_counter()-start-sampling_seconds
        self.info.update(sampling_seconds=sampling_seconds,saved_field_values=int(self.samples.size),
                         spatial_nodes=int(np.prod(self.sizes)))
        out=Path(out)
        np.savez_compressed(out/'model_classical.npz',x=xx,y=yy,z=zz,initial=self.initial,
                            eval_x=ex,eval_y=ey,eval_z=ez,eval_t=et,u_samples=self.samples)
        (out/'classical_grid.json').write_text(json.dumps(self.info,indent=2))
        self._interpolators()
        return history,seconds

    def _interpolators(self):
        xx,yy,zz=self.axes;ex,ey,ez,et=self.eval_axes
        self.initial_interp=RegularGridInterpolator((zz,yy,xx),self.initial,bounds_error=True)
        self.field_interp=RegularGridInterpolator((et,ez,ey,ex),self.samples,bounds_error=True)

    def predict(self,points,derivative=None):
        p=np.asarray(points,float)
        if not np.isfinite(p).all() or np.any(p<self.problem.bounds[:,0]) or np.any(p>self.problem.bounds[:,1]):
            raise ValueError('Points outside the domain.')
        if derivative==(0,0,0,1) and np.all(p[:,3]==0): return np.zeros(len(p))
        if derivative is not None and derivative!=(0,0,0,0): raise ValueError('Only values and initial velocity supported.')
        result=np.zeros(len(p));initial=p[:,3]==0
        result[initial]=self.initial_interp(p[initial][:,[2,1,0]])
        boundary=np.any((p[:,:3]==self.problem.bounds[:3,0])|(p[:,:3]==self.problem.bounds[:3,1]),axis=1)
        rest=~initial & ~boundary
        result[rest]=self.field_interp(p[rest][:,[3,2,1,0]])
        result[boundary]=0
        return result

    def restore(self,out):
        out=Path(out)
        with np.load(out/'model_classical.npz') as d:
            self.axes=[d[k].copy() for k in ['x','y','z']]
            self.eval_axes=[d[k].copy() for k in ['eval_x','eval_y','eval_z','eval_t']]
            self.initial=d['initial'].copy();self.samples=d['u_samples'].copy()
        self.info=json.loads((out/'classical_grid.json').read_text());self._interpolators()
