#!/usr/bin/env python3
"""Paired-input Ours-only GPU latency and accuracy regression; no baseline solver."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extension-dir', type=Path, default=ROOT/'bindings')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--limit', type=int)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--check-contract', action='store_true')
    p.add_argument('--inputs', nargs='+',
                   help='Directories of frozen .npz inputs; default: tuning confirmation trials')
    p.add_argument('--time-transfer', action='store_true',
                   help='Also record end-to-end time including H2D upload and D2H result retrieval')
    args = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
        os.environ[key] = '1'
    import numpy as np
    import torch
    sys.path.insert(0, str(args.extension_dir.resolve()))
    import turboreg_gpu as backend
    import teaser_sim3_benchmark as metrics
    metrics.np = np
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.init()
    torch.cuda.set_device(0)
    if args.check_contract:
        import unittest
        sys.path.insert(0,str(ROOT/'tests'))
        from test_ours_gpu import OursGpuContract
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OursGpuContract))
        if not result.wasSuccessful():raise RuntimeError('GPU contract regression failed')
    options = json.loads((ROOT/'configs/teaser_sim3_ours_final.json').read_text())
    opt = backend.Sim3Options()
    for key,value in options.items():setattr(opt,key,value)
    if args.inputs:
        paths = []
        for spec in args.inputs:
            base = Path(spec)
            if not base.is_absolute():
                base = ROOT / base
            paths += sorted(base.glob('*.npz'))
        assert paths, f'no .npz inputs matched: {args.inputs}'
    else:
        paths = sorted((ROOT/'results/teaser_tuning_final/confirmation_1/trials').glob('*.npz'))
        paths += sorted((ROOT/'results/teaser_tuning_final/confirmation_2/trials').glob('*.npz'))
        assert len(paths)==1280
    if args.limit:paths=paths[:args.limit]
    if args.output.exists():raise ValueError('Use a new output directory; historical measurements are immutable.')
    args.output.mkdir(parents=True)
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    meta=dict(gpu=torch.cuda.get_device_name(),physical_gpu=args.gpu,torch=torch.__version__,
              extension=str(backend.__file__),extension_sha256=digest(backend.__file__),options=options,
              repeats=args.repeats,inputs=len(paths),
              input_dirs=[str(p) for p in (args.inputs or ['results/teaser_tuning_final/confirmation_1/trials',
                                                           'results/teaser_tuning_final/confirmation_2/trials'])],
              timing='Synchronized single-pair API wall time; inputs already on GPU, output retrieval excluded.',
              timing_e2e=('runtime_e2e_ms additionally includes numpy->GPU upload and result retrieval'
                          if args.time_transfer else None))
    (args.output/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
    with np.load(paths[0]) as d:
        a=torch.from_numpy(d['src'].copy()).cuda();b=torch.from_numpy(d['dst'].copy()).cuda()
    for _ in range(30):backend.register_sim3(a,b,opt)
    torch.cuda.synchronize()
    rows=[]
    with (args.output/'results.jsonl').open('w') as log:
        for i,path in enumerate(paths):
            with np.load(path) as d:
                src=d['src'].copy();dst=d['dst'].copy();gt=d['gt'].copy()
            a=torch.from_numpy(src).cuda();b=torch.from_numpy(dst).cuda()
            times=[];error=None;output=None
            for _ in range(args.repeats):
                torch.cuda.synchronize();start=time.perf_counter()
                try:output=backend.register_sim3(a,b,opt)
                except RuntimeError as e:error=str(e)
                torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
            row=dict(input=str(path.relative_to(ROOT)),input_sha256=metrics.input_hash(src,dst),
                     runtime_ms=times,status='ok' if error is None else 'failed')
            if args.time_transfer:
                e2e=[]
                for _ in range(args.repeats):
                    start=time.perf_counter()
                    ta=torch.from_numpy(src).cuda();tb=torch.from_numpy(dst).cuda()
                    try:
                        out2=backend.register_sim3(ta,tb,opt)
                        out2.transformation.cpu().numpy()
                    except RuntimeError:
                        pass
                    torch.cuda.synchronize();e2e.append((time.perf_counter()-start)*1000)
                row['runtime_e2e_ms']=e2e
            if error:row['error']=error
            else:
                transform=output.transformation.cpu().numpy()
                row.update(transformation=transform.tolist(),**metrics.errors(transform,gt))
            rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
            if (i+1)%160==0:print(f'{i+1}/{len(paths)} inputs',flush=True)
    summary=dict(inputs=len(rows),failures=sum(r['status']!='ok' for r in rows),
                 median_ms=float(np.median([t for r in rows for t in r['runtime_ms']])),
                 p95_ms=float(np.quantile([t for r in rows for t in r['runtime_ms']],.95)))
    if args.time_transfer:
        e2e_all=[t for r in rows for t in r.get('runtime_e2e_ms',[])]
        if e2e_all:
            summary['median_e2e_ms']=float(np.median(e2e_all))
            summary['p95_e2e_ms']=float(np.quantile(e2e_all,.95))
    for key in metrics.METRICS[:3]:summary[key]=float(np.mean([r[key] for r in rows if r['status']=='ok']))
    if args.profile:
        # Four recorded inputs covering the four outlier rates, after timing is complete.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            for pct in [60,70,80,90]:
                path=ROOT/f'results/teaser_tuning_final/confirmation_1/trials/bunny-o{pct}-r000.npz'
                with np.load(path) as d:
                    a=torch.from_numpy(d['src'].copy()).cuda();b=torch.from_numpy(d['dst'].copy()).cuda()
                torch.cuda.synchronize()
                with torch.profiler.record_function(f'registration_{pct}'):
                    backend.register_sim3(a,b,opt);torch.cuda.synchronize()
        prof.export_chrome_trace(str(args.output/'trace.json'))
        events=json.loads((args.output/'trace.json').read_text())['traceEvents']
        regions=[e for e in events if e.get('cat')=='user_annotation' and e.get('name','').startswith('registration_')]
        calls=[]
        from collections import Counter
        for region in regions:
            inside=[e for e in events if region['ts']<=e['ts']<region['ts']+region['dur']]
            calls.append(dict(name=region['name'],kernels=sum(e.get('cat')=='kernel' for e in inside),
                              runtime_calls=dict(Counter(e['name'] for e in inside if e.get('cat')=='cuda_runtime')),
                              copies=dict(Counter(e['name'] for e in inside if e.get('cat')=='gpu_memcpy'))))
        summary['profile']=calls
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
