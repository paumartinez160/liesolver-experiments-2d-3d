#!/usr/bin/env python3
"""Run one case, a benchmark suite, or regenerate figures from saved numbers."""
import argparse
import contextlib
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import uuid

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
sys.path.insert(0,str(ROOT/'upstream/liesolver'))
import numpy as np
import yaml
from pipeline.solvers import LieAdapter
ADAPTERS = dict(liesolver=LieAdapter, classical=None)
from pipeline.artifacts import export,plots,write_csv


def run(config,method,seed=None):
    cfg=yaml.safe_load(Path(config).read_text())
    if seed is not None: cfg['seed']=seed
    if cfg['pde'] not in ('wave_2d', 'wave_3d'):
        raise ValueError('Only wave_2d and wave_3d are supported.')
    problem_class=None
    adapter_class=ADAPTERS[method]
    if cfg['pde']=='wave_2d':
        if method=='pinn': raise ValueError('This 2D experiment supports LieSolver and classical only.')
        from pipeline.wave2d import Wave2DProblem
        from pipeline.classical2d import Classical2DAdapter
        problem_class=Wave2DProblem
        adapter_class=LieAdapter if method=='liesolver' else Classical2DAdapter
    if cfg['pde']=='wave_3d':
        if method=='pinn': raise ValueError('3D supports LieSolver and classical only.')
        from pipeline.wave3d import Wave3DProblem
        from pipeline.classical3d import Classical3DAdapter
        problem_class=Wave3DProblem
        adapter_class=LieAdapter if method=='liesolver' else Classical3DAdapter
    if method=='liesolver' and cfg['fit']['max_bricks']<1: raise ValueError('max_bricks must be positive')
    if method=='classical':
        if cfg['pde']=='wave_3d': cfg['classical']={'nx':201,'ny':201,'nz':201,'cfl':.8,**cfg.get('classical',{})}
        cfg['classical']={**({'nx':101,'ny':101,'cfl':.8} if cfg['pde']=='wave_2d' else {'nx':401,'cfl':.9}),**cfg.get('classical',{})}
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=ROOT/'results'/f"{stamp}_{Path(config).stem}_{method}_s{cfg['seed']}_{uuid.uuid4().hex[:8]}"
    out.mkdir(parents=True,exist_ok=False)
    (out/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    meta=dict(status='running',method=method,seed=cfg['seed'],python=sys.version,platform=platform.platform(),
              cpu_count=os.cpu_count(),threads=os.environ['OPENBLAS_NUM_THREADS'],
              upstream_commit=(ROOT/'upstream/liesolver/UPSTREAM_COMMIT').read_text().strip(),
              versions={d.metadata['Name']:d.version for d in importlib.metadata.distributions()})
    def save_meta(): (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    save_meta(); print('Run:',out,flush=True)
    started=time.perf_counter()
    try:
        with (out/'run.log').open('w',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            np.random.seed(cfg['seed'])
            problem=problem_class(cfg)
            adapter=adapter_class(cfg,problem)
            meta['setup_seconds']=time.perf_counter()-started
            history,seconds=adapter.fit(out)
            write_csv(out/'history.csv',history)
            metrics=export(adapter,problem,out)
            metrics['training_seconds']=seconds  # Legacy CSV compatibility; classical has no training.
            metrics['solve_seconds']=seconds
            if method=='classical': metrics.update(adapter.info)
            if method=='liesolver':
                model=adapter.model
                before=adapter.predict(problem.test.constraints[0].x)
                model.load_from_npz(out,problem.module.trafos,problem.coords)
                metrics['reload_max_delta']=float(np.max(np.abs(before-adapter.predict(problem.test.constraints[0].x))))
                if cfg['pde']=='wave_3d':
                    probe=np.random.default_rng(2026).uniform(problem.bounds[:,0],problem.bounds[:,1],size=(256,4))
                    residual=adapter.predict(probe,(0,0,0,2))-sum(adapter.predict(probe,tuple(2 if j==i else 0 for j in range(4))) for i in range(3))
                elif cfg['pde']=='wave_2d':
                    probe=np.random.default_rng(2026).uniform(problem.bounds[:,0],problem.bounds[:,1],size=(256,3))
                    residual=adapter.predict(probe,(0,0,2))-adapter.predict(probe,(2,0,0))-adapter.predict(probe,(0,2,0))
                else:
                    residual=adapter.predict(problem.test.constraints[0].x, (0,1) if cfg['pde']=='heat_1d' else (0,2))-adapter.predict(problem.test.constraints[0].x,(2,0))
                metrics['pde_residual_max_at_ic_test' if cfg['pde'] not in ('wave_2d','wave_3d') else 'pde_residual_max_interior']=float(np.max(np.abs(residual)))
                metrics['fit_mse_target_met']=bool(model.mse<=cfg['fit']['mse_tol'])
                metrics['fit_termination']='mse_target' if metrics['fit_mse_target_met'] else 'max_bricks'
                metrics['bricks']=len(model.bricks)
                metrics['parameters']=sum(b.params.size+1 for b in model.bricks)
            else:
                before=adapter.predict(problem.test.constraints[0].x)
                restored=adapter_class(cfg,problem)
                restored.restore(out)
                metrics['reload_max_delta']=float(np.max(np.abs(before-restored.predict(problem.test.constraints[0].x))))
            (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
            plots(out)
            meta['status']='completed'
    except BaseException:
        meta['status']='interrupted' if sys.exc_info()[0] is KeyboardInterrupt else 'failed'
        (out/'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        meta['total_seconds']=time.perf_counter()-started
        save_meta()
    print(json.dumps(metrics,indent=2)); return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    one=sub.add_parser('run'); one.add_argument('config',type=Path)
    one.add_argument('--method',choices=list(ADAPTERS),default='liesolver'); one.add_argument('--seed',type=int)
    plot=sub.add_parser('plot'); plot.add_argument('folder',type=Path)
    comp=sub.add_parser('compare'); comp.add_argument('folders',nargs='+',type=Path)
    suite=sub.add_parser('suite'); suite.add_argument('--methods',nargs='+',choices=list(ADAPTERS),default=list(ADAPTERS))
    suite.add_argument('--seeds',nargs='+',type=int,default=list(range(5)))
    suite.add_argument('--cases',nargs='+',default=None)
    args=parser.parse_args()
    if args.command=='run': run(args.config,args.method,args.seed)
    elif args.command=='plot': plots(args.folder)
    elif args.command=='compare':
        from pipeline.comparison import compare
        print(compare(args.folders,ROOT/'results'))
    else:
        rows=[]
        for cfg in sorted([*(ROOT/'configs_2d').glob('*.yaml'), *(ROOT/'configs_3d').glob('*.yaml')]):
            if args.cases and cfg.stem not in args.cases: continue
            for method in args.methods:
                for seed in args.seeds:
                    try:
                        folder=run(cfg,method,seed)
                        metrics=json.loads((folder/'metrics.json').read_text())
                        rows.append(dict(case=cfg.stem,method=method,seed=seed,status='completed',folder=str(folder),
                                         l2=metrics['domain_l2_relative'],mse=metrics['test_ibc_mse'],seconds=metrics['training_seconds']))
                    except Exception as exc:
                        print(f'Failed {cfg.stem}/{method}/{seed}: {exc}',file=sys.stderr)
                        rows.append(dict(case=cfg.stem,method=method,seed=seed,status='failed',folder='',l2=None,mse=None,seconds=None))
        batch=ROOT/'results'/('suite_'+uuid.uuid4().hex[:12]); batch.mkdir(parents=True)
        write_csv(batch/'runs.csv',rows)
        summary=[]
        for case,method in sorted({(r['case'],r['method']) for r in rows}):
            group=[r for r in rows if (r['case'],r['method'])==(case,method) and r['status']=='completed']
            row=dict(case=case,method=method,successful=len(group),requested=len(args.seeds))
            for key in ['l2','mse','seconds']:
                values=[r[key] for r in group if r[key] is not None]
                row[key+'_median']=float(np.median(values)) if values else None
                row[key+'_std']=float(np.std(values)) if values else None
            summary.append(row)
        write_csv(batch/'summary.csv',summary); print(batch)
        if any(r['status']=='failed' for r in rows): sys.exit(1)

if __name__=='__main__': main()
