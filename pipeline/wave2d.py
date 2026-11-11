"""2+1D wave benchmark and valid Lie symmetries; upstream optimizer is unchanged."""
import sys
import numpy as np
import sympy as sp
from scipy.stats import qmc
from liesolver.model import Transformation
from liesolver.constraints import Constraint, ConstraintSet
from .problems import evaluate

x, y, t = sp.symbols('x y t', real=True)
coords_sym = [x, y, t]


def translate_x(expr, amount):
    return expr.subs(x, x-amount)


def translate_y(expr, amount):
    return expr.subs(y, y-amount)


def scale_spacetime(expr, scale):
    return expr.subs({x:scale*x, y:scale*y, t:scale*t}, simultaneous=True)


trafos = [Transformation(translate_x, [-3.141592653589793,3.141592653589793], 1),
          Transformation(translate_y, [-3.141592653589793,3.141592653589793], 2),
          Transformation(scale_spacetime, [.5,10], 4)]


class Wave2DProblem:
    def __init__(self,cfg):
        self.cfg=cfg
        self.module=sys.modules[__name__]
        self.coords=coords_sym
        self.bounds=np.asarray([cfg['data']['range_dim'][key] for key in ['x','y','t']],float)
        if self.bounds.shape!=(3,2) or not np.isfinite(self.bounds).all() or not np.all(np.diff(self.bounds,axis=1)>0):
            raise ValueError('Expected increasing finite x,y,t intervals.')
        if self.bounds[2,0]!=0 or cfg['data'].get('phys'):
            raise ValueError('Wave 2D currently requires t0=0 and c=1.')
        self.u0=sp.Integer(0); self.exact=sp.Integer(0)
        modes=cfg['problem']['modes']
        if not modes: raise ValueError('At least one spatial mode is required.')
        for mode in modes:
            mx,my=mode['mx'],mode['my']
            if any(isinstance(m,bool) or not isinstance(m,int) or m<1 for m in [mx,my]):
                raise ValueError('mx and my must be positive integers.')
            a=float(mode['amplitude'])
            if not np.isfinite(a): raise ValueError('Finite mode amplitudes required.')
            kx=mx*sp.pi/sp.Rational(str(self.bounds[0,1]-self.bounds[0,0]))
            ky=my*sp.pi/sp.Rational(str(self.bounds[1,1]-self.bounds[1,0]))
            spatial=sp.Rational(str(a))*sp.sin(kx*(x-sp.Rational(str(self.bounds[0,0]))))*sp.sin(ky*(y-sp.Rational(str(self.bounds[1,0]))))
            self.u0+=spatial
            self.exact+=spatial*sp.cos(sp.sqrt(kx**2+ky**2)*t)
        self.corner_mismatch=[0.]*4
        self.train=self.sample(True)
        self.test=self.sample(False)

    def sample(self,training):
        constraints=ConstraintSet()
        # Initial displacement, initial velocity, four spatial faces.
        faces=[(2,0,'IC_u0',self.u0,(0,0,0)),(2,0,'IC_ut0',sp.Integer(0),(0,0,1))]
        faces += [(axis,side,f'BC_{name}_{side}',sp.Integer(0),(0,0,0))
                  for axis,name in enumerate(['x','y']) for side in [0,1]]
        for index,(axis,side,name,expr,derivative) in enumerate(faces):
            count=self.cfg['data']['num_face_points']
            if training:
                uv=qmc.Halton(2,scramble=True,seed=self.cfg['seed']+index).random(count)
            else:
                n=int(np.ceil(np.sqrt(count)))
                a,b=np.meshgrid(np.linspace(0,1,n+2)[1:-1],np.linspace(0,1,n+2)[1:-1],indexing='ij')
                uv=np.column_stack([a.ravel(),b.ravel()])
            points=np.tile(self.bounds[:,0],(len(uv),1))
            free=[i for i in range(3) if i!=axis]
            points[:,free]+=uv*(self.bounds[free,1]-self.bounds[free,0])
            points[:,axis]=self.bounds[axis,side]
            constraints.add(Constraint(x=points,y=evaluate(expr,self.coords,points),deriv_order=derivative,weight=1.,name=name))
        return constraints

    def reference(self,points,modes=None):
        # Closed form, evaluated only by the evaluation/export path.
        return evaluate(self.exact,self.coords,points)
