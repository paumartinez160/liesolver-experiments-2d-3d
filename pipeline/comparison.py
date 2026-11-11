"""Compare completed runs on exactly the same evaluation grid and reference."""
import json
import uuid
from pathlib import Path
import numpy as np
from .artifacts import write_csv


def compare(folders, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folders=[Path(f) for f in folders]
    if len(folders)<2: raise ValueError('Provide at least two completed runs.')
    arrays=[]; rows=[]; labels=[]
    for folder in folders:
        metadata=json.loads((folder/'metadata.json').read_text())
        if metadata['status']!='completed': raise ValueError(f'Incomplete run: {folder}')
        with np.load(folder/'domain.npz') as d:
            arrays.append({k:d[k].copy() for k in d.files})
        if len(arrays)>1 and (not np.array_equal(arrays[0]['points'],arrays[-1]['points']) or
                             not np.allclose(arrays[0]['reference'],arrays[-1]['reference'],rtol=1e-12,atol=1e-14)):
            raise ValueError('Runs must have the same evaluation points and reference. Re-run with a common grid.')
        label=f"{metadata['method']}_s{metadata['seed']}_{len(rows)+1}"
        labels.append(label)
        m=json.loads((folder/'metrics.json').read_text())
        rows.append(dict(label=label,method=metadata['method'],folder=str(folder.resolve()),
                         relative_l2=m['domain_l2_relative'],ibc_mse=m['test_ibc_mse'],
                         solve_seconds=m.get('solve_seconds',m['training_seconds']),
                         total_seconds=metadata['total_seconds'],acceptance_passed=m['acceptance_passed']))
    out=Path(destination)/('comparison_'+uuid.uuid4().hex[:12]);out.mkdir(parents=True)
    write_csv(out/'comparison.csv',rows)
    fields={'points':arrays[0]['points'],'reference':arrays[0]['reference']}
    for label,a in zip(labels,arrays):
        fields[label+'_prediction']=a['prediction'];fields[label+'_error']=a['error']
    np.savez_compressed(out/'comparison_data.npz',**fields)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,key,title in zip(axes,['relative_l2','solve_seconds'],['Interior relative L2 error','Solve / training time [s]']):
        ax.bar(labels,[r[key] for r in rows]);ax.set_yscale('log');ax.set_title(title)
        ax.tick_params(axis='x',labelrotation=20)
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    points=arrays[0]['points'];time_axis=points.shape[1]-1;times=np.unique(points[:,time_axis])
    if time_axis==3:
        xs,ys,zs=[np.unique(points[:,i]) for i in range(3)]
        shape=(len(times),len(zs),len(ys),len(xs))
        initial=arrays[0]['reference'].reshape(shape)[0]
        zi=int(np.argmax(np.sum(initial**2,axis=(1,2))))
        yi=int(np.argmax(np.sum(initial[zi]**2,axis=1)))
    chosen=times[np.unique(np.linspace(0,len(times)-1,4,dtype=int))]
    fig,axes=plt.subplots(1,len(chosen),figsize=(4*len(chosen),3.5),layout='constrained')
    for ax,t in zip(np.atleast_1d(axes),chosen):
        mask=points[:,time_axis]==t
        if time_axis==2:
            ys=np.unique(points[:,1]);mask &= points[:,1]==ys[len(ys)//2]
        if time_axis==3: mask &= (points[:,2]==zs[zi]) & (points[:,1]==ys[yi])
        ax.plot(points[mask,0],arrays[0]['reference'][mask],color='black',label='Reference')
        for label,a in zip(labels,arrays): ax.plot(points[mask,0],a['prediction'][mask],'--',label=label)
        ax.set(title=f't={t:.4g}',xlabel='x',ylabel='u');ax.legend(fontsize=7)
    fig.savefig(out/'comparison_slices.png',dpi=160);plt.close(fig)
    if time_axis in (2,3):
        xs=np.unique(points[:,0]);ys=np.unique(points[:,1])
        for suffix,keys,names in [('fields',['reference']+['prediction']*len(arrays),['Reference']+labels),
                                  ('errors',['error']*len(arrays),labels)]:
            sources=[arrays[0]]+arrays if suffix=='fields' else arrays
            limit=max(float(np.max(np.abs(a[k]))) for a,k in zip(sources,keys)) or 1e-15
            fig,axes=plt.subplots(len(sources),len(chosen),figsize=(4*len(chosen),3.2*len(sources)),squeeze=False,layout='constrained')
            for row,(source,key,name) in enumerate(zip(sources,keys,names)):
                for col,tt in enumerate(chosen):
                    field_mask=points[:,time_axis]==tt
                    if time_axis==3: field_mask &= points[:,2]==zs[zi]
                    field=source[key][field_mask].reshape(len(ys),len(xs))
                    ax=axes[row,col]
                    im=ax.pcolormesh(xs,ys,field,shading='auto',cmap='coolwarm',vmin=-limit,vmax=limit)
                    ax.set(title=f'{name}, t={tt:.3g}'+(f', z={zs[zi]:.3g}' if time_axis==3 else ''),xlabel='x',ylabel='y',aspect='equal')
                    fig.colorbar(im,ax=ax)
            fig.savefig(out/f'comparison_{suffix}.png',dpi=140);plt.close(fig)
    return out
