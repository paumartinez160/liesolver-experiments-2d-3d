"""4D evaluation arrays and non-nodal spatial slices for the 3D wave experiment."""
import csv
from pathlib import Path
import numpy as np
from .artifacts import write_csv


def export3d(adapter,problem,out):
    sizes=problem.cfg['evaluation']['grid']
    if len(sizes)!=4 or any(not isinstance(n,int) or n<3 for n in sizes) or np.prod(sizes)>10_000_000:
        raise ValueError('Invalid or excessively large evaluation grid.')
    axes=[np.linspace(*problem.bounds[i],sizes[i]) for i in range(4)]
    x,y,z,t=axes
    T,Z,Y,X=np.meshgrid(t,z,y,x,indexing='ij')
    points=np.column_stack([X.ravel(),Y.ravel(),Z.ravel(),T.ravel()]);del X,Y,Z,T
    ref=np.empty(len(points));pred=np.empty(len(points))
    for start in range(0,len(points),4096):
        chunk=points[start:start+4096]
        ref[start:start+len(chunk)]=problem.reference(chunk)
        pred[start:start+len(chunk)]=adapter.predict(chunk)
    err=pred-ref
    if not np.isfinite(pred).all(): raise ValueError('Non-finite predictions.')
    shape=(len(t),len(z),len(y),len(x))
    np.savez_compressed(out/'domain.npz',points=points,reference=ref,prediction=pred,error=err,
                        x=x,y=y,z=z,t=t,shape=np.array(shape))
    inside=np.all((points>problem.bounds[:,0])&(points<problem.bounds[:,1]),axis=1)
    norm=np.linalg.norm(ref[inside]);relative=float(np.linalg.norm(err[inside])/norm) if norm else None
    metrics=dict(domain_mse=float(np.mean(err[inside]**2)),domain_l2_relative=relative,
                 domain_max_error=float(np.max(np.abs(err[inside]))),reference_type='exact wave eigenmode sum',evaluation_points=len(points))
    rows=[]
    for split,constraints in [('train',problem.train),('test',problem.test)]:
        squared=[]
        for c in constraints:
            values=adapter.predict(c.x,c.deriv_order);squared.extend((values-c.y)**2)
            for p,target,v in zip(c.x,c.y,values):
                rows.append(dict(split=split,name=c.name,x=p[0],y=p[1],z=p[2],t=p[3],
                                 dx=c.deriv_order[0],dy=c.deriv_order[1],dz=c.deriv_order[2],dt=c.deriv_order[3],target=target,prediction=v,error=v-target))
        metrics[split+'_ibc_mse']=float(np.mean(squared))
    write_csv(out/'constraints.csv',rows)
    metrics['acceptance_passed']=bool(metrics['train_ibc_mse']<=1e-6 and relative is not None and relative<=1e-3)
    fields=[a.reshape(shape) for a in [ref,pred,err]]
    zi=int(np.argmax(np.sum(fields[0][0]**2,axis=(1,2))))
    yi=int(np.argmax(np.sum(fields[0][0,zi]**2,axis=1)))
    slices=[];temporal=[]
    for ti,tt in enumerate(t):
        a=fields[2][ti,1:-1,1:-1,1:-1]
        temporal.append(dict(t=tt,rmse=float(np.sqrt(np.mean(a*a))),max_error=float(np.max(np.abs(a)))))
        if ti in np.unique(np.linspace(0,len(t)-1,4,dtype=int)):
            for xi,xx in enumerate(x):
                slices.append(dict(x=xx,y=y[yi],z=z[zi],t=tt,reference=fields[0][ti,zi,yi,xi],prediction=fields[1][ti,zi,yi,xi],error=fields[2][ti,zi,yi,xi]))
    write_csv(out/'slices.csv',slices);write_csv(out/'error_by_time.csv',temporal)
    return metrics


def field_figures(data,out):
    import matplotlib.pyplot as plt
    shape=tuple(data['shape']);ref=data['reference'].reshape(shape)
    zi=int(np.argmax(np.sum(ref[0]**2,axis=(1,2))))
    times=np.unique(np.linspace(0,shape[0]-1,4,dtype=int))
    fig,axes=plt.subplots(3,len(times),figsize=(4*len(times),9),squeeze=False,layout='constrained')
    for row,key in enumerate(['reference','prediction','error']):
        values=data[key].reshape(shape)
        scale=float(np.max(np.abs(data['error'] if key=='error' else data['reference']))) or 1e-15
        for col,ti in enumerate(times):
            ax=axes[row,col]
            im=ax.pcolormesh(data['x'],data['y'],values[ti,zi],shading='auto',cmap='coolwarm',vmin=-scale,vmax=scale)
            ax.set(xlabel='x',ylabel='y',title=f"{key}, t={data['t'][ti]:.3g}, z={data['z'][zi]:.3g}",aspect='equal')
            fig.colorbar(im,ax=ax)
    fig.savefig(out/'domain.png',dpi=140);plt.close(fig)


def plots3d(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out)
    with np.load(out/'domain.npz') as d: data={k:d[k] for k in d.files if k!='points'}
    field_figures(data,out)
    for name,xkey,keys,ylabel in [('history','step',['train_mse','test_mse','max_abs_u'],'Loss / norm'),
                                  ('error_by_time','t',['rmse','max_error'],'Interior error')]:
        with (out/f'{name}.csv').open() as f: rows=list(csv.DictReader(f))
        fig,ax=plt.subplots(layout='constrained')
        for key in keys:
            if key in rows[0]: ax.semilogy([float(r[xkey]) for r in rows],np.maximum([float(r[key]) for r in rows],1e-30),label=key)
        ax.set(xlabel=xkey,ylabel=ylabel);ax.legend();fig.savefig(out/f'{name}.png',dpi=160);plt.close(fig)
    with (out/'slices.csv').open() as f: rows=list(csv.DictReader(f))
    fig,ax=plt.subplots(layout='constrained')
    for tt in sorted({float(r['t']) for r in rows}):
        group=[r for r in rows if float(r['t'])==tt]
        line,=ax.plot([float(r['x']) for r in group],[float(r['reference']) for r in group],label=f't={tt:.3g}')
        ax.plot([float(r['x']) for r in group],[float(r['prediction']) for r in group],'--',color=line.get_color())
    ax.set(xlabel='x',ylabel='u',title=f"Cut at y={float(rows[0]['y']):.3g}, z={float(rows[0]['z']):.3g}")
    ax.legend();fig.savefig(out/'slices.png',dpi=160);plt.close(fig)
