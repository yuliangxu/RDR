#!/usr/bin/env python3
"""Verify recovery of all archived MNIST scores and a fresh package replay."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def same_image(a,b):return np.array_equal(np.array(Image.open(a)),np.array(Image.open(b)))
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--extraction',type=Path,required=True)
p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2],
 help='Checkout matching the frozen source; use the historical checkout for old packages')
a=p.parse_args();root=a.root.resolve();ex=a.extraction.resolve()
first=root/'training_historical';repeat=root/'training_historical_repeat'
records=[];n_scores=0
for name in ['vae','dcgan']:
 original=load(root/f'inputs/generator/{name}_rdr_checkpoint.pt')
 one=load(first/f'generator/{name}_rdr_checkpoint.pt');two=load(repeat/f'generator/{name}_rdr_checkpoint.pt')
 assert all(torch.equal(v,one['model_state'][k]) and torch.equal(v,two['model_state'][k]) for k,v in original['model_state'].items())
 for role in ['p','q']:
  v=original[f'test_{role}_scores'];assert torch.equal(v,one[f'test_{role}_scores']) and torch.equal(v,two[f'test_{role}_scores'])
  n_scores+=len(v)
 assert original['best_epoch']==one['best_epoch']==two['best_epoch']
 assert all(torch.equal(v,one['split_indices'][k]) and torch.equal(v,two['split_indices'][k]) for k,v in original['split_indices'].items())
 records.append({'experiment':name,'weights_equal_original':True,'all_archived_scores_equal':True,'repeat_equal':True,'best_epoch':one['best_epoch']})
recovered=root/'recovered_models';recovered.mkdir(exist_ok=True)
for exp in ['perturbation','null']:
 original=load(root/'inputs'/exp/'rdr_scores.pt');one=load(first/exp/'rdr_scores.pt');two=load(repeat/exp/'rdr_scores.pt')
 for role in ['train','test' if exp=='perturbation' else 'validation']:
  for group in ['p','q']:
   for key in ['scores','labels','original_indices']:
    v=original[role][group][key]
    assert torch.equal(v,one[role][group][key]) and torch.equal(v,two[role][group][key]),(exp,role,group,key)
   n_scores+=len(original[role][group]['scores'])
 assert original['best_epoch']==one['best_epoch']==two['best_epoch']
 s1=load(first/exp/'rdr_checkpoint.pt')['model_state'];s2=load(repeat/exp/'rdr_checkpoint.pt')['model_state']
 assert all(torch.equal(v,s2[k]) for k,v in s1.items())
 shutil.copyfile(first/exp/'rdr_checkpoint.pt',recovered/f'{exp}_rdr_checkpoint.pt')
 records.append({'experiment':exp,'all_archived_scores_equal':True,'repeat_equal':True,'repeat_weights_equal':True,'original_weights_available':False,'best_epoch':one['best_epoch']})
assert n_scores==240000
source=root/'inputs/generator/mnist_generator_strict_split_figure.png'
assert same_image(source,first/'generator/mnist_generator_strict_split_figure.png')
assert same_image(source,repeat/'generator/mnist_generator_strict_split_figure.png')
final=root/'inputs/comparison/mnist_vae_dcgan_comparison.png'
assert same_image(final,root/'replay_from_training/figure/mnist_vae_dcgan_comparison.png')
assert same_image(final,ex/'replayed/figure/mnist_vae_dcgan_comparison.png')
x=json.loads((ex/'replayed/replay_audit.json').read_text())
for k in ['input_hashes','generator_csv_scores','panel_scores','split_membership','table_values_at_report_precision','all_report_links']:assert x[k] is True
# Check every member of the extracted archive, not just the direct scientific inputs.
manifest=json.loads((ex/'MNIST_jrssb_final/package_manifest.json').read_text())
for k,v in manifest.items():assert sha(ex/'MNIST_jrssb_final'/k)==v['sha256']
for exp in ['generator','perturbation','null']:
 for run in [first,repeat]:
  scale=json.loads((run/exp/'scale_audit.json').read_text())
  assert scale['train_min']==scale['validation_min']==-1 and scale['train_max']==scale['validation_max']==1
  env=json.loads((run/exp/'training_environment.json').read_text())
  assert env['numerical_profile']=='historical' and env['cudnn_allow_tf32'] is True
# Keep the source-equality gate strict across old and reorganized packages.
inputs=json.loads((root/'inputs_manifest.json').read_text())
source_files=[name for name in inputs if name.startswith(('source/utils/','source/experiments/MNIST/'))]
assert source_files, 'No shared source files recorded in the frozen input manifest'
for name in source_files:
 frozen=root/name;current=a.repo.resolve()/name[len('source/'):]
 assert sha(frozen)==inputs[name]['sha256'], f'Frozen source changed: {name}'
 if not current.is_file() or sha(current)!=sha(frozen):
  raise RuntimeError(f'Source checkout differs: {name}; set --repo to the recorded frozen-source checkout')
audit={'historical_scores_exact':True,'archived_score_count':n_scores,'generator_weights_exact':True,
 'historical_repeat_exact':True,'regenerated_source_png_exact':True,'regenerated_final_png_exact':True,
 'fresh_extraction_replay_pass':True,'shared_utilities_unchanged':True,'train_validation_scale':'[-1,1]',
 'experiments':records,'numerical_profile':json.loads((first/'generator/training_environment.json').read_text()),
 'training_jobs':[55338090,55343216],
 'scope':'RDR training from the supplied pretrained VAE/DCGAN, not their pretraining.',
 'auxiliary_weight_caveat':'Recovered auxiliary weights reproduce all archived scores exactly; original auxiliary parameter identity cannot be checked because original weights were absent.',
 'extraction_path':str(ex),
 'recovered_model_hashes':{p.name:sha(p) for p in recovered.iterdir() if p.is_file()}}
(root/'end_to_end_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
