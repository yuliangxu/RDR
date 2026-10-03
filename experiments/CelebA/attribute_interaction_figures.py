#!/usr/bin/env python3
"""Publication layout for the sealed EBM analysis; no model fitting."""
import argparse
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import pandas as pd
from experiments.CelebA.attribute_interactions import sealed,read,sha,require,seal,plots


def main(args):
    run=args.analysis.resolve();output=args.output.resolve()
    require(not output.exists(),'Use a new figure output directory')
    receipt=sealed(run/'COMPLETE.json')
    require(receipt['tasks']==20 and receipt['ablations']==940,'Incomplete analysis')
    for name,digest in receipt['files'].items():require(sha(run/name)==digest,'Changed analysis output')
    metrics=pd.read_csv(run/'metrics_per_fit.csv');importance=pd.read_csv(run/'importance_per_fit.csv')
    output.mkdir(parents=True);plots(output,metrics,importance)
    sources={}
    for name in ('attribute_interactions.py','attribute_interaction_figures.py'):
        path=output/'source/experiments/CelebA'/name;path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(Path(__file__).with_name(name),path);sources[str(path.relative_to(output/'source'))]=sha(path)
    files={p.name:sha(p) for p in output.iterdir() if p.is_file()}
    seal(output/'COMPLETE.json',dict(complete=True,figure_pairs=3,parent=str(run),parent_completion_sha256=sha(run/'COMPLETE.json'),files=files,source_files=sources,changes='Sparse scientific-notation ticks avoid label overlap; readable legends; numerical results unchanged'))
    print('Complete: three publication figure pairs')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',type=Path,default=Path('/cwork/yx306/RDR/CelebA/attribute_interactions_20261003'))
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
