#!/usr/bin/env python3
"""Freeze, replay, and retrain the final MNIST experiment without changing shared code."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mnist-final-matplotlib')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

PACKAGES = ['torch', 'torchvision', 'numpy', 'pandas', 'matplotlib', 'scipy', 'Pillow', 'seaborn', 'ipython']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str) + '\n')


def environment():
    import torch
    return dict(python=sys.version, executable=sys.executable,
                packages={p: importlib.metadata.version(p) for p in PACKAGES},
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                device=torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu')


def freeze(args):
    root = args.root.resolve()
    repo = args.repo.resolve()
    if root.exists():
        raise RuntimeError('Freeze destination already exists; refusing to overwrite')
    root.mkdir(parents=True)
    records = {}

    def copy(src, dst):
        src = Path(src)
        if src.is_dir():
            for f in sorted(src.rglob('*')):
                if f.is_file() and '__pycache__' not in f.parts:
                    copy(f, Path(dst) / f.relative_to(src))
            return
        if not src.is_file() or src.is_symlink():
            raise RuntimeError(f'Missing or symlink input: {src}')
        target = root / dst
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        records[str(dst)] = dict(original=str(src.resolve()), sha256=digest(target), bytes=target.stat().st_size)

    base = Path('/cwork/yx306/RDR')
    copy(base/'mnist-generator-trainval-testall', 'inputs/generator')
    copy(base/'mnist-vae-dcgan-comparison', 'inputs/comparison')
    copy(repo/'experiments/results/MNIST_label_perturbation', 'inputs/perturbation')
    copy(repo/'experiments/results/MNIST_two_halves', 'inputs/null')
    copy(repo/'experiments/JRSSB/MNIST_jrssb.md', 'report_template.md')
    for name in ['MNIST_generator_strict_split.py', 'MNIST_generator_strict_split.slurm',
                 'MNIST_label_perturbation.py', 'MNIST_two_halves.py']:
        copy(repo/'experiments'/name, 'source/experiments/'+name)
    # Freeze the complete shared dependency closure and dataset-owned helpers.
    for source in sorted((repo/'utils').glob('*.py')):
        copy(source, 'source/utils/'+source.name)
    for name in ['__init__.py', 'helpers.py', 'sampling.py']:
        copy(repo/'experiments/MNIST'/name, 'source/experiments/MNIST/'+name)
    if (repo/'experiments/__init__.py').exists():
        copy(repo/'experiments/__init__.py', 'source/experiments/__init__.py')
    data = Path('/hpc/group/mastatlab/yx306/MNIST')
    copy(data/'MNIST/raw', 'assets/MNIST/raw')
    copy(data/'mnist_vae/vae.py', 'assets/mnist_vae/vae.py')
    copy(data/'mnist_vae/vae_epoch_25.pth', 'assets/mnist_vae/vae_epoch_25.pth')
    copy(data/'mnist_dcgan/netG_epoch_99.pth', 'assets/mnist_dcgan/netG_epoch_99.pth')
    for job in ['53679306', '53679669']:
        for ext in ['out', 'err']:
            copy(base/f'out/mnist-generator-strict-{job}.{ext}', f'logs/mnist-generator-strict-{job}.{ext}')
    copy(Path(__file__), 'workflow.py')
    dump(root/'environment.json', environment())
    versions = environment()['packages']
    (root/'requirements-replay.txt').write_text('\n'.join(f'{k}=={v}' for k,v in versions.items())+'\n')
    dump(root/'git_state.json', dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
         status=subprocess.check_output(['git','status','--short'],cwd=repo,text=True),
         note='Source snapshots are audit-time copies, not authenticated historical training sources.'))
    dump(root/'inputs_manifest.json', records)
    (root/'README.md').write_text('''# Final MNIST reproduction package

Run from this extracted directory, with Python 3.9 and requirements-replay.txt:

```bash
python workflow.py replay --root . --output /cwork/yx306/RDR/mnist-replay-new
python workflow.py train --root . --output /cwork/yx306/RDR/mnist-retrain-new --experiment all --numerics historical
python workflow.py replay --root . --from-training /cwork/yx306/RDR/mnist-retrain-new --output /cwork/yx306/RDR/mnist-report-from-training
```

Output directories must be new. Replay verifies all frozen input hashes and
reconstructs tables, the report and figure from saved final evidence without
training or external data access. All required MNIST data, generator weights,
ratio checkpoints and source files are included. Original image mosaics are
retained because original generated image/latent tensors were not saved.

Training runs all three original protocols (paired generators, perturbation,
validation null) with frozen source code and records new fitted model states,
loss histories and input-scale checks. It never replaces the historical result.
The default historical profile uses cuDNN TF32, with deterministic-algorithm
forcing disabled, matching the original script defaults. The optional strict
profile changes arithmetic and is not the final-result reproduction profile.
The pretrained VAE/DCGAN are inputs, not trained by this workflow. Reproducing
their pretraining is outside the RDR experiment. Historical perturbation/null
weights are absent. Recovery of the final scores must pass exact comparisons before any cleanup.
Training is validated separately, including comparison with historical scores
and repeat-run agreement. A changed historical result is
not silently accepted. See training_audit.json when available.

The environment records the audit/replay runtime, not the historical GPU
runtime. GPU retraining additionally requires a compatible CUDA PyTorch build.
Requirements installation and a clean GPU environment must be validated on
the target system. No repository-wide requirements or shared files are edited.
''')
    print(root, flush=True)


def validate_inputs(root):
    records = json.loads((root/'inputs_manifest.json').read_text())
    for name, rec in records.items():
        if digest(root/name) != rec['sha256']:
            raise RuntimeError(f'Frozen input changed: {name}')
    return records


def load(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=False)


def stats(scores):
    import numpy as np
    x = scores.double().numpy()
    return dict(n=len(x), mean=float(x.mean()), std=float(x.std(ddof=1)),
                q05=float(np.quantile(x,.05)), q95=float(np.quantile(x,.95)),
                rmse=float(np.sqrt(np.mean((x-1)**2))))


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def replay(args):
    import numpy as np
    import pandas as pd
    import torch
    root, out = args.root.resolve(), args.output.resolve()
    results = args.from_training.resolve() if args.from_training else root/'inputs'
    validate_inputs(root)
    if out.exists():
        raise RuntimeError('Replay output exists')
    out.mkdir(parents=True)
    if args.from_training:
        for exp in ['generator','perturbation','null']:
            if exp == 'generator':
                for name in ['vae','dcgan']:
                    new=load(results/f'generator/{name}_rdr_checkpoint.pt')
                    old=load(root/f'inputs/generator/{name}_rdr_checkpoint.pt')
                    assert all(torch.equal(new['model_state'][k],v) for k,v in old['model_state'].items())
                    assert all(torch.equal(new[f'test_{g}_scores'],old[f'test_{g}_scores']) for g in ['p','q'])
            else:
                new=load(results/exp/'rdr_scores.pt');old=load(root/'inputs'/exp/'rdr_scores.pt')
                for role in ['train','test' if exp=='perturbation' else 'validation']:
                    for g in ['p','q']:
                        for field in ['scores','labels','original_indices']:
                            assert torch.equal(new[role][g][field],old[role][g][field]),(exp,role,g,field)
        from PIL import Image
        assert np.array_equal(np.array(Image.open(results/'generator/mnist_generator_strict_split_figure.png')),
                              np.array(Image.open(root/'inputs/generator/mnist_generator_strict_split_figure.png')))
    checks, metrics, tables = {}, {}, {}
    csv = pd.read_csv(results/'generator/rdr_scores.csv')
    panel = pd.read_csv(root/'inputs/comparison/panel_selection.csv')
    # IDX parsing avoids dataset download and leaves input directories unchanged.
    labels = np.frombuffer((root/'assets/MNIST/raw/t10k-labels-idx1-ubyte').read_bytes(),dtype=np.uint8,offset=8)
    genrows = []
    genindices = None
    for name in ['vae','dcgan']:
        b = load(results/f'generator/{name}_rdr_checkpoint.pt')
        s = b['split_indices']
        assert len(s['train_indices']) == 55000 and len(s['validation_loss_indices']) == 5000
        assert set(s['train_indices'].tolist()).isdisjoint(s['validation_loss_indices'].tolist())
        assert set(s['train_indices'].tolist()) | set(s['validation_loss_indices'].tolist()) == set(range(60000))
        assert s['test_indices'].tolist() == list(range(10000))
        assert np.array_equal(labels, b['test_p_labels'].numpy())
        if genindices is not None:
            assert all(torch.equal(genindices[k],v) for k,v in s.items())
        genindices=s
        p=b['test_p_scores'].double().numpy(); q=b['test_q_scores'].double().numpy()
        for role, x in [('p',p),('q',q)]:
            rows=csv[(csv.generator==name)&(csv.group==f'test_{role}_'+('real' if role=='p' else 'generated'))]
            assert len(x)==10000 and np.isfinite(x).all() and (x>=0).all() and (x<=2).all()
            assert np.allclose(rows.rdr.to_numpy(),x,rtol=0,atol=1e-14)
            ps=panel[(panel.generator==name)&(panel.role==role)]
            assert len(ps)==120 and np.allclose(ps.rdr,x[ps.test_position],rtol=0,atol=1e-14)
        pp=np.maximum(p,1e-6); qq=np.maximum(q,1e-6)
        var=float(1-.5*np.mean(pp**-.5)-.25*np.mean(pp**.5)-.25*np.mean(qq**.5))
        aff=float(1-.5*(np.sqrt(np.maximum(p,1e-8)).mean()+np.sqrt(np.maximum(q,1e-8)).mean()))
        metrics[name]=dict(p_mean=float(p.mean()),q_mean=float(q.mean()),variational=var,affinity=aff,best_epoch=b['best_epoch'])
        expected={'vae':(.276175,.300565),'dcgan':(.096444,.123618)}[name]
        assert abs(var-expected[0])<.00000051 and abs(aff-expected[1])<.00000051
        genrows.append([f'Real vs {name.upper()}', '10,000 / 10,000',f'{p.mean():.6f}',f'{q.mean():.6f}',f'{var:.6f}',f'{aff:.6f}'])
    tables['generator']=table(['Contrast','Test P / Q','Mean RDR on P','Mean RDR on Q','Variational H²','Affinity plug-in H²'],genrows)
    b=load(results/'perturbation/rdr_scores.pt')
    s=b['split_indices']; v=set(s['validation_loss_p_indices'].tolist()); t=set(s['test_p_indices'].tolist())
    assert len(v)==len(t)==5000 and v.isdisjoint(t) and v|t==set(range(10000))
    for role,pool,n in [('train',set(range(60000)),60000),('test',t,5000)]:
        assert len(b[role]['p']['scores'])==n and len(b[role]['q']['scores'])==n
        assert set(b[role]['q']['sampled_indices'].tolist()) <= pool
    assert len(b['validation_loss']['q_sampled_indices'])==5000
    assert set(b['validation_loss']['q_sampled_indices'].tolist())<=v
    p=b['test']['p']['scores'].double().numpy(); y=b['test']['p']['labels'].numpy()
    q=b['test']['q']['scores'].double().numpy(); yq=b['test']['q']['labels'].numpy()
    pi=np.bincount(y,minlength=10)/5000; w=b['q_label_probs'].numpy(); nominal=.2/(.1+w); target=2*pi/(pi+w)
    assert np.array_equal(y,labels[b['test']['p']['original_indices'].numpy()])
    drows=[]
    for d in range(10):
        drows.append([d,f'{w[d]:.3f}',f'{nominal[d]:.4f}',f'{target[d]:.4f}',int((y==d).sum()),f'{p[y==d].mean():.4f}',f'{q[yq==d].mean():.4f}' if (yq==d).any() else '—'])
    tables['digits']=table(['Digit','Q probability','Nominal target','Test-pool target','Test P n','Mean fitted RDR on test P','Mean fitted RDR on test Q'],drows)
    met={}
    for key,tar in [('nominal',nominal),('test_pool',target)]:
        err=p-tar[y];met[key]=dict(mae=float(np.abs(err).mean()),rmse=float(np.sqrt(np.mean(err**2))))
    met['affinity']=float(1-.5*(np.sqrt(p).mean()+np.sqrt(q).mean()))
    met['group_means']=[float(p[np.isin(y,d)].mean()) for d in [[0,1],[2,3],[4,5,6,7],[8,9]]]
    met['unique_q']=[len(set(b['train']['q']['sampled_indices'].tolist())),len(set(b['validation_loss']['q_sampled_indices'].tolist())),len(set(b['test']['q']['sampled_indices'].tolist()))]
    assert met['unique_q']==[28734,2414,2346]
    assert round(met['nominal']['mae'],4)==.1755 and round(met['nominal']['rmse'],4)==.2315
    assert round(met['test_pool']['mae'],4)==.1770 and round(met['test_pool']['rmse'],4)==.2327
    assert round(met['affinity'],6)==-.000406
    metrics['perturbation']=met
    b=load(results/'null/rdr_scores.pt'); nullrows=[]; ns={}
    assert 'test' not in b
    for role in ['train','validation']:
        assert set(b[role]['p']['original_indices'].tolist()).isdisjoint(b[role]['q']['original_indices'].tolist())
        assert set(b[role]['p']['original_indices'].tolist()) | set(b[role]['q']['original_indices'].tolist()) == set(range(60000 if role=='train' else 10000))
        for group in ['p','q']:
            z=stats(b[role][group]['scores']);assert z['n']==(30000 if role=='train' else 5000)
            ns[role+'_'+group]=z
            nullrows.append([('Training' if role=='train' else 'Validation')+' '+group.upper(),f"{z['n']:,}",f"{z['mean']:.6f}",f"{z['std']:.6f}",f"[{z['q05']:.6f}, {z['q95']:.6f}]",f"{z['rmse']:.6f}"])
    tables['null']=table(['Evaluated sample','n','Mean RDR','SD','5th–95th percentile','RMSE against r=1'],nullrows)
    metrics['null']=ns
    text=(root/'report_template.md').read_text()
    for start,key in [('| Contrast |','generator'),('| Digit |','digits'),('| Evaluated sample |','null')]:
        a=text.index(start);e=text.find('\n\n',a)
        assert text[a:e].splitlines()[2:] == tables[key].splitlines()[2:], 'Published table mismatch: '+key
        text=text[:a]+tables[key]+text[e:]
    original_root=Path('/hpc/home/yx306/RDR/experiments/JRSSB')
    records=json.loads((root/'inputs_manifest.json').read_text())
    mapping={v['original']:root/k for k,v in records.items()}
    if args.from_training:
        for k,v in records.items():
            if any(k.startswith('inputs/'+exp+'/') for exp in ['generator','perturbation','null']):
                mapping[v['original']]=results/Path(k).relative_to('inputs')
    def link(m):
        lab,tar=m.groups()
        if tar=='agent3.md':return '`agent3.md` (separate CelebA report)'
        old=str((original_root/tar).resolve())
        if old not in mapping:raise RuntimeError('Unpackaged report link: '+old)
        dest=mapping[old]
        if '/mnist-vae-dcgan-comparison/' in old:
            dest=out/'figure'/Path(old).name
        return lab+'('+os.path.relpath(dest,out)+')'
    text=re.sub(r'(!?\[[^\]]*\])\(([^)]+)\)',link,text)
    text=text.replace('seed 42 partitions the official','global seed 42 (split seed 52) partitions the official')
    subprocess.run([sys.executable,str(root/'inputs/comparison/make_mnist_vae_dcgan_comparison.py'),
                    '--source',str(results/'generator'),'--output',str(out/'figure')],check=True,cwd=out)
    (out/'MNIST_jrssb.md').write_text(text)
    for _,tar in re.findall(r'(!?\[[^\]]*\])\(([^)]+)\)',text):assert (out/tar).is_file(),tar
    for k,v in tables.items():(out/f'{k}_table.md').write_text(v+'\n')
    dump(out/'metrics.json',metrics)
    checks=dict(input_hashes=True,generator_csv_scores=True,panel_scores=True,split_membership=True,
                table_values_at_report_precision=True,all_report_links=True,environment=environment(),
                historical_training_recovered=bool(args.from_training),
                limitation='Recovered auxiliary weights reproduce archived scores; original auxiliary parameter identity cannot be checked because those original weights were not saved.')
    dump(out/'replay_audit.json',checks)
    print(json.dumps(checks),flush=True)


def train(args):
    root,out=args.root.resolve(),args.output.resolve()
    validate_inputs(root)
    if out.exists():raise RuntimeError('Training output already exists')
    out.mkdir(parents=True)
    if args.experiment=='all':
        for exp in ['generator','perturbation','null']:
            subprocess.run([sys.executable,str(root/'workflow.py'),'train','--root',str(root),'--output',str(out/exp),'--experiment',exp,'--numerics',args.numerics],check=True,cwd=out)
        return
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    torch.set_num_threads(4)
    strict = args.numerics == 'strict'
    torch.use_deterministic_algorithms(strict)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=strict
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=not strict
    sys.path.insert(0,str(root/'source'))
    import utils.DRE_batch as dre
    assert Path(dre.__file__).resolve().is_relative_to(root/'source')
    import utils.DRE_func as dre_func
    assert Path(dre_func.__file__).resolve().is_relative_to(root/'source')
    scale={'train_min':1.,'train_max':-1.,'validation_min':1.,'validation_max':-1.,'calls':0}
    original=dre.run_DRE_fdiv_cnn_minibatch
    fitted=[]
    def run_checked(*pos,**kw):
        def checked_sampler(fn,key):
            def sample(*a,**k):
                x=fn(*a,**k);assert torch.isfinite(x).all() and x.min()>=-1.000001 and x.max()<=1.000001
                scale[key+'_min']=min(scale[key+'_min'],float(x.min()));scale[key+'_max']=max(scale[key+'_max'],float(x.max()));return x
            return sample
        kw['q_sampler']=checked_sampler(kw['q_sampler'],'train')
        kw['val_q_sampler']=checked_sampler(kw['val_q_sampler'],'validation')
        # Check every numerator image while keeping original loader iteration and RNG behavior.
        class CheckLoader:
            def __init__(self,loader,key):self.loader,self.key=loader,key
            def __len__(self):return len(self.loader)
            def __iter__(self):
                for batch in self.loader:
                    x=batch[0];assert torch.isfinite(x).all() and x.min()>=-1.000001 and x.max()<=1.000001
                    scale[self.key+'_min']=min(scale[self.key+'_min'],float(x.min()));scale[self.key+'_max']=max(scale[self.key+'_max'],float(x.max()))
                    yield batch
        kw['p_loader']=CheckLoader(kw['p_loader'],'train');kw['val_loader']=CheckLoader(kw['val_loader'],'validation')
        result=original(*pos,**kw);fitted.append(result);scale['calls']+=1;return result
    dre.run_DRE_fdiv_cnn_minibatch=run_checked
    if args.experiment=='generator':
        from experiments import MNIST_generator_strict_split as mod
        assert Path(mod.__file__).resolve().is_relative_to(root/'source')
        old_save=mod.save_outputs
        def save(results,*a,**k):
            for name,b in results.items():
                torch.save({key:value for key,value in b.items() if key in ['test_p_images','test_q_images','test_p_labels','test_p_scores','test_q_scores']},out/f'{name}_test_images.pt')
            return old_save(results,*a,**k)
        mod.save_outputs=save
        argv=['generator','--data-root',str(root/'assets'),'--output-dir',str(out),
              '--generators','vae','dcgan','--protocol','trainval_testall','--eval-input-scale','model',
              '--seed','42','--batch-size','512','--num-workers','0','--num-epochs','20','--vae-epochs','3',
              '--patience','5','--validation-batches','10','--validation-size','5000','--eval-n','0','--output-alpha','0.5']
        sys.argv=argv;mod.main()
    else:
        name='MNIST_label_perturbation.py' if args.experiment=='perturbation' else 'MNIST_two_halves.py'
        script=root/'source/experiments'/name
        code=script.read_text()
        # Only redirect filesystem destinations; keep statistical/training source unchanged.
        assert code.count('DATA_ROOT = REPO_ROOT / "data"')==1
        code=code.replace('DATA_ROOT = REPO_ROOT / "data"','DATA_ROOT = Path('+repr(str(root/'assets'))+')')
        code=code.replace('DOWNLOAD = True','DOWNLOAD = False')
        var='MNIST_LABEL_PERTURB_OUTPUT_DIR' if args.experiment=='perturbation' else 'MNIST_TWO_HALVES_OUTPUT_DIR'
        os.environ[var]=str(out)
        plt.show=lambda *a,**k:plt.close('all')
        g={'__name__':'__main__','__file__':str(script)}
        exec(compile(code,str(script),'exec'),g)
        torch.save(dict(model_state=g['model'].state_dict(),losses=g['losses'],val_losses=g['val_losses'],
                        best_epoch=g['best_epoch'],output_alpha=.5,seed=g['SEED'],input_scale='[-1,1]'),out/'rdr_checkpoint.pt')
    env=environment()
    env['numerical_profile']=args.numerics
    env['deterministic_algorithms']=torch.are_deterministic_algorithms_enabled()
    env['cudnn_allow_tf32']=torch.backends.cudnn.allow_tf32
    dump(out/'training_environment.json',env)
    dump(out/'scale_audit.json',scale)
    dump(out/'source_adaptations.json',dict(source='frozen audit-time source',
         changes=['redirect paths','disable downloads','capture missing checkpoints and generated display tensors',
                  'numerical profile '+args.numerics,'validate numerator and midpoint pixel ranges'],
         historical_identity_claim=False))
    print('TRAINING COMPLETE',args.experiment,flush=True)


def compare(args):
    import numpy as np
    root=args.root.resolve();a=args.output.resolve();b=args.repeat.resolve()
    result={}
    for exp in ['generator','perturbation','null']:
        comparisons=[]
        if exp=='generator':
            names=['vae_rdr_checkpoint.pt','dcgan_rdr_checkpoint.pt']
            pairs=[(load(a/exp/n),load(b/exp/n),load(root/'inputs/generator'/n)) for n in names]
        else:
            pairs=[(load(a/exp/'rdr_scores.pt'),load(b/exp/'rdr_scores.pt'),load(root/f'inputs/{exp}/rdr_scores.pt'))]
        for x,y,h in pairs:
            def vectors(d):
                if 'test_p_scores' in d:return [d['test_p_scores'].numpy(),d['test_q_scores'].numpy()]
                return [d[role][g]['scores'].numpy() for role in ['train', 'test' if exp=='perturbation' else 'validation'] for g in ['p','q']]
            vx,vy,vh=vectors(x),vectors(y),vectors(h)
            comparisons.append(dict(repeat_bitwise_equal=all(np.array_equal(i,j) for i,j in zip(vx,vy)),
                repeat_max_abs=max(float(np.max(np.abs(i-j))) for i,j in zip(vx,vy)),
                historical_mae=[float(np.mean(np.abs(i-j))) for i,j in zip(vx,vh)],
                historical_max_abs=[float(np.max(np.abs(i-j))) for i,j in zip(vx,vh)],
                historical_best_epoch=h.get('best_epoch'),new_best_epoch=x.get('best_epoch')))
        result[exp]=comparisons
    result['new_training_repeat_pass']=all(v['repeat_bitwise_equal'] for k,vs in result.items() if isinstance(vs,list) for v in vs)
    result['historical_training_recovered']=all(max(v['historical_max_abs'])<1e-6 for k,vs in result.items() if isinstance(vs,list) for v in vs)
    dump(a.parent/'training_audit.json',result)
    print(json.dumps(result,indent=2),flush=True)
    if not result['new_training_repeat_pass']:raise RuntimeError('Training repeat differs')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['freeze','replay','train','compare'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,default=Path.cwd())
    p.add_argument('--from-training',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--repeat',type=Path)
    p.add_argument('--numerics',choices=['strict','historical'],default='historical')
    p.add_argument('--experiment',choices=['all','generator','perturbation','null'],default='all')
    args=p.parse_args()
    if args.command!='freeze' and args.output is None:p.error('--output is required')
    if args.command=='compare' and args.repeat is None:p.error('--repeat is required')
    globals()[args.command](args)

if __name__=='__main__':main()
