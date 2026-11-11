"""3 spatial dimensions plus time: rectangular wave eigenmode benchmarks."""
import sys
import numpy as np
import sympy as sp
from scipy.stats import qmc
from liesolver.model import Transformation
from liesolver.constraints import Constraint, ConstraintSet
from .problems import evaluate

x,y,z,t=sp.symbols('x y z t',real=True)
coords_sym=[x,y,z,t]


def shift_x(expr,a): return expr.subs(x,x-a)
def shift_y(expr,a): return expr.subs(y,y-a)
def shift_z(expr,a): return expr.subs(z,z-a)
def scale_spacetime(expr,s):
    return expr.subs({v:s*v for v in coords_sym},simultaneous=True)


trafos=[Transformation(shift_x,[-np.pi,np.pi],1),
        Transformation(shift_y,[-np.pi,np.pi],2),
        Transformation(shift_z,[-np.pi,np.pi],3),
        Transformation(scale_spacetime,[.5,20],4)]


class Wave3DProblem:
    def __init__(self,cfg):
        self.cfg=cfg; self.module=sys.modules[__name__];self.coords=coords_sym
        self.bounds=np.asarray([cfg['data']['range_dim'][str(c)] for c in coords_sym],float)
        if self.bounds.shape!=(4,2) or not np.isfinite(self.bounds).all() or not np.all(np.diff(self.bounds,axis=1)>0):
            raise ValueError('Increasing finite x,y,z,t intervals required.')
        if self.bounds[3,0]!=0 or cfg['data'].get('phys'): raise ValueError('Requires t0=0 and c=1.')
        self.u0=sp.Integer(0);self.exact=sp.Integer(0)
        if not cfg['problem']['modes']: raise ValueError('At least one mode is required.')
        for mode in cfg['problem']['modes']:
            modes=[mode[k] for k in ['mx','my','mz']]
            if any(isinstance(m,bool) or not isinstance(m,int) or m<1 for m in modes):
                raise ValueError('Mode indices must be positive integers.')
            amplitude=float(mode['amplitude'])
            if not np.isfinite(amplitude): raise ValueError('Finite amplitude required.')
            spatial=sp.Rational(str(amplitude));ks=[]
            for i,m in enumerate(modes):
                k=m*sp.pi/sp.Rational(str(self.bounds[i,1]-self.bounds[i,0]));ks.append(k)
                spatial*=sp.sin(k*(coords_sym[i]-sp.Rational(str(self.bounds[i,0]))))
            self.u0+=spatial;self.exact+=spatial*sp.cos(sp.sqrt(sum(k*k for k in ks))*t)
        self.train=self.sample(True);self.test=self.sample(False)

    def sample(self,training):
        result=ConstraintSet()
        faces=[(3,0,'IC_u0',self.u0,(0,0,0,0)),(3,0,'IC_ut0',sp.Integer(0),(0,0,0,1))]
        faces += [(i,side,f'BC_{coords_sym[i]}_{side}',sp.Integer(0),(0,0,0,0)) for i in range(3) for side in [0,1]]
        count=self.cfg['data']['num_face_points']
        for index,(axis,side,name,expr,derivative) in enumerate(faces):
            if training:
                uv=qmc.Halton(3,scramble=True,seed=self.cfg['seed']+index).random(count)
            else:
                n=int(np.ceil(np.cbrt(count)));grid=np.linspace(0,1,n+2)[1:-1]
                uv=np.column_stack([a.ravel() for a in np.meshgrid(grid,grid,grid,indexing='ij')])
            points=np.tile(self.bounds[:,0],(len(uv),1));free=[i for i in range(4) if i!=axis]
            points[:,free]+=uv*(self.bounds[free,1]-self.bounds[free,0])
            points[:,axis]=self.bounds[axis,side]
            result.add(Constraint(x=points,y=evaluate(expr,coords_sym,points),deriv_order=derivative,weight=1.,name=name))
        return result

    def reference(self,points,modes=None): return evaluate(self.exact,coords_sym,points)
