#!/usr/bin/env python3
"""Build and optionally extract a portable final-MNIST reproduction archive."""
from pathlib import Path
import argparse,hashlib,json,re,zipfile
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True);p.add_argument('--zip',type=Path,required=True)
p.add_argument('--extract',type=Path)
a=p.parse_args();root=a.root.resolve();files={}
if a.extract and a.extract.exists():raise RuntimeError('Extraction directory exists')
for folder in ['assets','source','inputs','logs','replay','recovered_models','tools','audits']:
 if not (root/folder).exists():continue
 for f in sorted((root/folder).rglob('*')):
  if f.is_file() and '__pycache__' not in f.parts:files[f.relative_to(root).as_posix()]=f.read_bytes()
for name in ['workflow.py','README.md','requirements-replay.txt','environment.json','git_state.json','inputs_manifest.json',
             'report_template.md','workflow_revision.json','end_to_end_audit.json','cleanup_manifest.json','cleanup_receipt.json',
             'REPRODUCTION_AUDIT.md','MNIST_jrssb_verify_training.slurm']:
 if (root/name).is_file():files[name]=(root/name).read_bytes()
text=(root/'replay/MNIST_jrssb.md').read_text()
def rewrite(m):
 label,target=m.groups();dest=(root/'replay'/target).resolve().relative_to(root).as_posix()
 return label+'('+dest+')'
text=re.sub(r'(!?\[[^\]]*\])\(([^)]+)\)',rewrite,text)
files['MNIST_jrssb.md']=text.encode()
(root/'MNIST_jrssb.md').write_text(text)
for _,target in re.findall(r'(!?\[[^\]]*\])\(([^)]+)\)',text):assert target in files,target
manifest={k:{'sha256':hashlib.sha256(v).hexdigest(),'bytes':len(v)} for k,v in files.items()}
files['package_manifest.json']=json.dumps(manifest,indent=2).encode()
with zipfile.ZipFile(a.zip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for name,data in files.items():z.writestr('MNIST_jrssb_final/'+name,data)
with zipfile.ZipFile(a.zip) as z:
 assert z.testzip() is None
 for name,meta in manifest.items():assert hashlib.sha256(z.read('MNIST_jrssb_final/'+name)).hexdigest()==meta['sha256']
 if a.extract:z.extractall(a.extract)
if a.extract:
 for name,meta in manifest.items():assert hashlib.sha256((a.extract/'MNIST_jrssb_final'/name).read_bytes()).hexdigest()==meta['sha256']
print(json.dumps({'zip':str(a.zip),'bytes':a.zip.stat().st_size,'files':len(files),'extracted_to':str(a.extract)}))
