"""2D spatial snapshots and full x,y,t numerical exports."""
import csv
from pathlib import Path
import numpy as np
from .artifacts import write_csv


def export2d(adapter,problem,out):
    nx,ny,nt=problem.cfg['evaluation']['grid']
    if min(nx,ny,nt)<3: raise ValueError('Evaluation grid needs >=3 points on each axis.')
    x=np.linspace(*problem.bounds[0],nx); y=np.linspace(*problem.bounds[1],ny); t=np.linspace(*problem.bounds[2],nt)
    T,Y,X=np.meshgrid(t,y,x,indexing='ij')
    points=np.column_stack([X.ravel(),Y.ravel(),T.ravel()])
    ref=problem.reference(points)
    pred=np.concatenate([adapter.predict(chunk) for chunk in np.array_split(points,max(1,int(np.ceil(len(points)/4096))))])
    if not np.isfinite(pred).all(): raise ValueError('Non-finite predictions.')
    err=pred-ref
    np.savez_compressed(out/'domain.npz',points=points,reference=ref,prediction=pred,error=err,
                        x=x,y=y,t=t,shape=np.array([nt,ny,nx]))
    residuals={};rows=[]
    for split,constraints in [('train',problem.train),('test',problem.test)]:
        squared=[]
        for c in constraints:
            values=adapter.predict(c.x,c.deriv_order)
            squared.extend((values-c.y)**2)
            for p,target,v in zip(c.x,c.y,values):
                rows.append(dict(split=split,name=c.name,x=p[0],y=p[1],t=p[2],dx=c.deriv_order[0],dy=c.deriv_order[1],dt=c.deriv_order[2],target=target,prediction=v,error=v-target))
        residuals[split+'_ibc_mse']=float(np.mean(squared))
    write_csv(out/'constraints.csv',rows)
    inside=np.ones(len(points),bool)
    for axis in range(3): inside&=(points[:,axis]>problem.bounds[axis,0])&(points[:,axis]<problem.bounds[axis,1])
    norm=np.linalg.norm(ref[inside]);relative=float(np.linalg.norm(err[inside])/norm) if norm else None
    metric=dict(**residuals,domain_mse=float(np.mean(err[inside]**2)),domain_l2_relative=relative,
                domain_max_error=float(np.max(np.abs(err[inside]))),reference_type='exact finite sum of rectangular wave eigenmodes',
                evaluation_points=len(points),acceptance_passed=bool(residuals['train_ibc_mse']<=1e-6 and relative is not None and relative<=1e-3))
    temporal=[]
    for i,tt in enumerate(t):
        e=err.reshape(nt,ny,nx)[i,1:-1,1:-1];r=ref.reshape(nt,ny,nx)[i,1:-1,1:-1]
        temporal.append(dict(t=tt,rmse=float(np.sqrt(np.mean(e**2))),reference_rms=float(np.sqrt(np.mean(r**2))),max_error=float(np.max(np.abs(e)))))
    write_csv(out/'error_by_time.csv',temporal)
    selected=np.unique(np.linspace(0,nt-1,4,dtype=int));center=ny//2
    slices=[]
    for ti in selected:
        for xi in range(nx):
            j=ti*ny*nx+center*nx+xi
            slices.append(dict(x=x[xi],y=y[center],t=t[ti],reference=ref[j],prediction=pred[j],error=err[j]))
    write_csv(out/'slices.csv',slices)
    return metric


def plots2d(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out)
    with np.load(out/'domain.npz') as z: data={k:z[k].copy() for k in z.files}
    nt,ny,nx=data['shape'];selected=np.unique(np.linspace(0,nt-1,4,dtype=int))
    fig,axes=plt.subplots(3,len(selected),figsize=(4*len(selected),9),squeeze=False,layout='constrained')
    for row,(key,title) in enumerate([('reference','Reference'),('prediction','Prediction'),('error','Error')]):
        field=data[key].reshape(nt,ny,nx)
        scale=float(np.max(np.abs(data['error'] if key=='error' else data['reference']))) or 1e-15
        for col,ti in enumerate(selected):
            ax=axes[row,col]
            im=ax.pcolormesh(data['x'],data['y'],field[ti],shading='auto',cmap='coolwarm',vmin=-scale,vmax=scale)
            ax.set(title=f"{title}, t={data['t'][ti]:.3g}",xlabel='x',ylabel='y',aspect='equal')
            fig.colorbar(im,ax=ax)
    fig.savefig(out/'domain.png',dpi=140);plt.close(fig)
    with (out/'error_by_time.csv').open() as f: rows=list(csv.DictReader(f))
    fig,ax=plt.subplots(layout='constrained')
    ax.semilogy([float(r['t']) for r in rows],np.maximum([float(r['rmse']) for r in rows],1e-30),label='Interior RMSE')
    ax.semilogy([float(r['t']) for r in rows],np.maximum([float(r['max_error']) for r in rows],1e-30),label='Interior max error')
    ax.set(xlabel='t',ylabel='Error');ax.legend();fig.savefig(out/'error_by_time.png',dpi=160);plt.close(fig)
    with (out/'history.csv').open() as f: rows=list(csv.DictReader(f))
    fig,ax=plt.subplots(layout='constrained')
    keys=[k for k in ['train_mse','test_mse','max_abs_u'] if k in rows[0]]
    for key in keys: ax.semilogy([float(r['step']) for r in rows],np.maximum([float(r[key]) for r in rows],1e-30),label=key)
    ax.set(xlabel='Time step' if 'max_abs_u' in keys else 'Optimization step',ylabel='Value');ax.legend()
    fig.savefig(out/'history.png',dpi=160);plt.close(fig)
    with (out/'slices.csv').open() as f: rows=list(csv.DictReader(f))
    fig,ax=plt.subplots(layout='constrained')
    for tt in sorted({float(r['t']) for r in rows}):
        group=[r for r in rows if float(r['t'])==tt]
        line,=ax.plot([float(r['x']) for r in group],[float(r['reference']) for r in group],label=f't={tt:.3g}')
        ax.plot([float(r['x']) for r in group],[float(r['prediction']) for r in group],'--',color=line.get_color())
    ax.set(xlabel='x',ylabel='u',title='Mid-y cut: reference (solid), prediction (dashed)');ax.legend()
    fig.savefig(out/'slices.png',dpi=160);plt.close(fig)
