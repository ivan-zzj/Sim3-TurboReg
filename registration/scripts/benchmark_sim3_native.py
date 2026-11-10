#!/usr/bin/env python3
"""Paired, resumable CPU comparison of official TEASER++ and native TurboReg."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import teaser_sim3_benchmark as b
from tune_teaser_sim3 import stats

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--repeats',type=int,default=40)
    p.add_argument('--split',choices=['development','selection','holdout','retrospective'],required=True)
    p.add_argument('--original-inputs',type=Path)
    args=p.parse_args()
    config=json.loads((b.REPO/'configs/teaser_sim3_protocol.json').read_text())
    config.update(test_seed=args.seed,repeats=args.repeats,device='cpu')
    b.configure_runtime(config)
    import turboreg_native as native
    np=b.np
    candidates=json.loads(args.candidates.read_text())
    assert len({c['id'] for c in candidates})==len(candidates)
    for c in candidates:
        assert c['method'] in ['Ours','TEASER++'] and c['device']=='cpu'
    meta=dict(config=config,candidates=candidates,split=args.split,
              source_inputs=None if args.original_inputs is None else str(args.original_inputs.resolve()),
              environment=b.environment(),native_build=json.loads((b.REPO/'build/sim3_native_build.json').read_text()),
              native_binary_sha256=b.digest(Path(native.__file__)),
              script_sha256=b.digest(Path(__file__)),base_script_sha256=b.digest(Path(b.__file__)))
    assert meta['native_build']['binary_sha256']==meta['native_binary_sha256']
    fingerprint=hashlib.sha256(b.canonical(meta).encode()).hexdigest()
    with b.exclusive(args.output):
        run=args.output/'run.json'
        if run.exists(): assert json.loads(run.read_text())['fingerprint']==fingerprint,'Resume contract changed'
        else: b.atomic_json(run,dict(fingerprint=fingerprint,**meta))
        models,manifest=b.load_models(Path('/path/to/data/sim3_benchmarks'),config)
        b.atomic_json(args.output/'data_manifest.json',manifest)
        (args.output/'trials').mkdir(exist_ok=True)
        trials=[]
        for oi,obj in enumerate(config['objects']):
            for pct in config['outlier_percent']:
                for rep in range(args.repeats):
                    tid=f'{obj}-o{pct:02d}-r{rep:03d}'
                    if args.original_inputs:
                        with np.load(args.original_inputs/'trials'/(tid+'.npz')) as data: trial={k:data[k] for k in data.files}
                    else: trial=b.make_trial(models[obj],config,oi,pct,rep)
                    trial=b.saved_trial(args.output/'trials'/(tid+'.npz'),trial)
                    trials.append((dict(trial_id=tid,object=obj,outlier_percent=pct,repeat=rep,
                                        seed=trial['seed'].tolist(),gt=trial['gt'].tolist()),trial))
        legacy_options=json.loads((b.REPO/'configs/teaser_sim3_ours_final.json').read_text())
        def solve(c,trial):
            if c['method']=='TEASER++': return b.solve('TEASER++',trial,config,legacy_options)
            src=np.array(trial['src'],dtype=np.float64,order='C');dst=np.array(trial['dst'],dtype=np.float64,order='C')
            ih=b.input_hash(src,dst)
            row=dict(method='Ours',input_sha256=ih,status='failed',transformation=None,
                     scale_error=None,rotation_error_deg=None,translation_error=None)
            start=time.perf_counter()
            try:
                result=native.register_sim3(src,dst,c['options'])
                row['runtime_s']=time.perf_counter()-start
                T=result['transformation']
                if np.isfinite(T).all(): row['transformation']=T.tolist()
                row.update(b.errors(T,trial['gt']))
                row['status']='ok'
            except Exception as exc:
                row.setdefault('runtime_s',time.perf_counter()-start)
                row['exception']=f'{type(exc).__name__}: {exc}'
            assert ih==b.input_hash(src,dst)==b.input_hash(trial['src'],trial['dst'])
            return row
        warm=b.make_trial(models['bunny'],config,0,0,100000,'validation')
        pending=[];records={};saved={}
        for c in candidates:
            path=args.output/(c['id']+'.json')
            if path.exists():
                saved[c['id']]=json.loads(path.read_text()); assert saved[c['id']]['fingerprint']==fingerprint
            else:
                for _ in range(10): solve(c,warm)
                pending.append(c);records[c['id']]=[]
        # Rotate candidate order between trials to reduce systematic order effects.
        for ti,(metadata,trial) in enumerate(trials):
            if not pending: break
            shift=ti%len(pending)
            for c in pending[shift:]+pending[:shift]:
                row=solve(c,trial);row.update(metadata,candidate_id=c['id'],device='cpu')
                records[c['id']].append(row)
            if (ti+1)%160==0: print(f'{ti+1}/{len(trials)} paired inputs; {len(pending)} methods/configurations',flush=True)
        summaries=[]
        for c in candidates:
            if c['id'] in saved: result=saved[c['id']]
            else:
                rows=records[c['id']]
                summary=dict(id=c['id'],device='cpu',method=c['method'],**stats(rows))
                conditions=[dict(object=obj,outlier_percent=pct,**stats([r for r in rows if r['object']==obj and r['outlier_percent']==pct]))
                            for obj in config['objects'] for pct in config['outlier_percent']]
                result=dict(fingerprint=fingerprint,candidate=c,summary=summary,conditions=conditions,rows=rows)
                b.atomic_json(args.output/(c['id']+'.json'),result)
            summaries.append(result['summary'])
            s=result['summary'];print(f'{c["id"]}: fail={s["failures"]}/{s["n"]}, RE={s["rotation_mean_failure_180"]:.6f}, '
                                     f'ms={s["runtime_s_median_finite"]*1000:.6f}, loss={s["selection_loss"]:.8f}',flush=True)
        b.atomic_json(args.output/'summary.json',dict(complete=True,summaries=summaries))
        print('COMPLETE',args.output,flush=True)

if __name__=='__main__': main()
