#!/usr/bin/env python3
"""Exact-path MNIST cleanup, gated on verified end-to-end reproduction."""
from pathlib import Path
import argparse, hashlib, json, zipfile

REPO=Path('/hpc/home/yx306/RDR')
BASE=Path('/cwork/yx306/RDR')
PACKAGE=BASE/'MNIST_jrssb_final'
DIRECTORIES=[BASE/n for n in [
 'mnist-generator-strict-split', 'mnist-generator-legacy-testval-alpha1-dcgan',
 'mnist-generator-legacy-testval-alpha1-dcgan-rescore-eval01',
 'mnist-generator-trainval-testall-alpha1-dcgan-eval01',
 'mnist-generator-trainval-testall-alpha2-dcgan']]
FILES=[REPO/'experiments'/n for n in [
 'MNIST_generator_alpha1_dcgan_eval01.slurm','MNIST_generator_alpha1_dcgan_legacy.slurm',
 'MNIST_generator_alpha2_dcgan.slurm','MNIST_generator_rescore_eval_scale.py',
 'MNIST_inception_embeddings.py','MNIST_two_halves_inception_embeddings.py',
 'MNIST_generator_fid_features.py','MNIST_generator_fid_features.slurm','MNIST_generator_fid_features.md']]
FILES += [BASE/'out'/(stem+'.'+ext) for stem in [
 'mnist-generator-strict-53679140','mnist-dcgan-alpha1-legacy-53687824',
 'mnist-dcgan-alpha1-eval01-53689365','mnist-dcgan-alpha2-53687229'] for ext in ['out','err']]
CACHES=['MNIST_two_halves','MNIST_generator_rescore_eval_scale','MNIST_label_perturbation',
        'MNIST_inception_embeddings','MNIST_generator_strict_split','MNIST_generator_fid_features',
        'MNIST_two_halves_inception_embeddings']

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def candidates():
 files=list(FILES)
 for d in DIRECTORIES:
  assert d.is_dir() and not d.is_symlink(),d
  files.extend(p for p in d.rglob('*') if p.is_file())
 for stem in CACHES:
  files.extend((REPO/'experiments/__pycache__').glob(stem+'.*.pyc'))
 files.extend((REPO/'utils/__pycache__').glob('MNIST_help.*.pyc'))
 for p in files:
  assert p.is_file() and not p.is_symlink(),p
  assert p.resolve()==p,p
 return sorted(set(files))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','apply']);a=p.parse_args()
 ledger=PACKAGE/'cleanup_manifest.json'
 if a.action=='prepare':
  rows=[{'path':str(f),'sha256':sha(f),'bytes':f.stat().st_size} for f in candidates()]
  ledger.write_text(json.dumps({'action':'dry-run','files':rows,'bytes':sum(r['bytes'] for r in rows)},indent=2)+'\n')
  print('Prepared',len(rows),'exact files; no deletions');return
 audit=json.loads((PACKAGE/'end_to_end_audit.json').read_text())
 for key in ['historical_scores_exact','generator_weights_exact','historical_repeat_exact',
             'regenerated_source_png_exact','regenerated_final_png_exact','fresh_extraction_replay_pass']:
  assert audit[key] is True,(key,audit.get(key))
 manifest=json.loads(ledger.read_text());rows=manifest['files']
 assert {str(p) for p in candidates()}=={r['path'] for r in rows}
 for r in rows:assert sha(Path(r['path']))==r['sha256'],r['path']
 archive=BASE/'MNIST_retired_pre_final.zip'
 if archive.exists():raise RuntimeError('Retirement archive exists; refusing overwrite')
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for r in rows:z.write(r['path'],r['path'].lstrip('/'))
  z.writestr('cleanup_manifest.json',json.dumps(manifest,indent=2))
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  for r in rows:assert hashlib.sha256(z.read(r['path'].lstrip('/'))).hexdigest()==r['sha256']
 for r in rows:
  f=Path(r['path']);assert sha(f)==r['sha256'];f.unlink()
 for d in DIRECTORIES:
  for child in sorted([p for p in d.rglob('*') if p.is_dir()],key=lambda x:len(x.parts),reverse=True):child.rmdir()
  d.rmdir()
 manifest.update(action='completed',archive=str(archive),archive_sha256=sha(archive),
  protected=['shared utilities','all non-MNIST files','primary MNIST artifacts','raw data','external embedding cache',
             'README-referenced MNIST_batch.py and MNIST_batch.ipynb'])
 (PACKAGE/'cleanup_receipt.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('Retired',len(rows),'files;',manifest['bytes'],'bytes; verified rollback archive',archive)

if __name__=='__main__':main()
