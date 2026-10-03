#!/usr/bin/env python3
"""Presentation-only update: three-digit image ranges and learned-null histograms."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR','/tmp/celeba-publication-mpl')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from experiments.CelebA.merged_test import (
    read, sha, sealed, require, write, verify_run, load_main_data, merge_scores,
    compact_panels, ROLES, SOURCE_FILES,
)

FAMILIES=(('feature','lower'),('feature','upper'),('pixel','lower'),('pixel','upper'),('pixel','real'))


def balanced_quantiles(a,b,probabilities=(.025,.975)):
    """Inverse empirical CDF of the equally weighted A/B score distributions."""
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    require(a.ndim==b.ndim==1 and len(a)>0 and len(b)>0,'Empty or non-vector scores')
    values=np.concatenate((a,b))
    require(np.isfinite(values).all() and np.all((values>=0)&(values<=2)),'Invalid RDR values')
    probabilities=np.asarray(probabilities,dtype=float)
    require(np.all((probabilities>=0)&(probabilities<=1)),'Invalid quantile probability')
    weights=np.concatenate((np.full(len(a),.5/len(a)),np.full(len(b),.5/len(b))))
    order=np.argsort(values,kind='stable');cdf=np.cumsum(weights[order]);cdf[-1]=1.
    indices=np.minimum(np.searchsorted(cdf,probabilities,side='left'),len(values)-1)
    return values[order[indices]]


def legend_handles():
    return [Line2D([],[],color='#1764bd',lw=1.4,label='Fold A'),
            Line2D([],[],color='#d62755',lw=1.4,label='Fold B'),
            Line2D([],[],color='.2',ls='--',lw=1.3,label=r'Null target $r=1$'),
            Patch(facecolor='#eedba8',alpha=.55,label='Central 95% of scores')]


def draw_histogram(ax,record,font_size):
    edges=record['edges']
    ax.axvspan(record['lower'],record['upper'],color='#eedba8',alpha=.55,zorder=0)
    ax.stairs(record['density_a'],edges,color='#1764bd',lw=1.25)
    ax.stairs(record['density_b'],edges,color='#d62755',lw=1.25)
    ax.axvline(1,color='.2',ls='--',lw=1.1)
    ax.set(xlim=(0,2),ylim=(0,None),xticks=[0,1,2])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3,min_n_ticks=2))
    ax.tick_params(labelsize=font_size,length=3)
    ax.grid(axis='y',alpha=.16,lw=.6)
    ax.spines[['top','right']].set_visible(False)


def save(fig,output,stem):
    for extension in ('png','pdf'):
        fig.savefig(output/f'{stem}.{extension}',dpi=220,bbox_inches='tight',pad_inches=.05,
                    metadata={'CreationDate':None,'ModDate':None} if extension=='pdf' else None)
    plt.close(fig)


def family_label(level,source):
    name=r'$P$ vs $P$' if source=='real' else rf'$Q_{"l" if source=="lower" else "u"}$ vs $Q_{"l" if source=="lower" else "u"}$'
    return name+'\n'+('Features' if level=='feature' else 'Pixels')


def render_histograms(output,records):
    fig,axes=plt.subplots(5,5,figsize=(18,17))
    fig.subplots_adjust(left=.105,right=.995,bottom=.055,top=.855,wspace=.36,hspace=.64)
    fig.suptitle(r'Learned same-source RDR: null target $r=1$',fontsize=27,y=.995,fontweight='bold')
    fig.legend(handles=legend_handles(),loc='upper center',bbox_to_anchor=(.55,.968),
               ncol=4,frameon=False,fontsize=20,handlelength=1.6,columnspacing=1.0)
    for row,(level,source) in enumerate(FAMILIES):
        for repeat in range(5):
            record=records[level,source,repeat];ax=axes[row,repeat]
            draw_histogram(ax,record,19)
            ax.set_title(f"RMSE {record['rmse']:.3g}\n95%: [{record['lower']:.3g}, {record['upper']:.3g}]",fontsize=19,pad=7)
        box=axes[row,0].get_position()
        fig.text(.026,(box.y0+box.y1)/2,family_label(level,source),rotation=90,
                 ha='center',va='center',fontsize=21,fontweight='bold')
    for repeat,ax in enumerate(axes[0]):
        box=ax.get_position();fig.text((box.x0+box.x1)/2,.917,f'Repeat {repeat+1}',ha='center',fontsize=22,fontweight='bold')
    fig.supxlabel(r'Fitted RDR $\widehat r$',fontsize=23,y=.012)
    fig.text(.072,.5,'Density',rotation=90,va='center',ha='center',fontsize=22)
    save(fig,output,'null_rdr_histograms')
    # Each family fits a two-column paper width without shrinking its labels.
    stems=['null_rdr_histograms']
    for level,source in FAMILIES:
        fig,axes=plt.subplots(2,3,figsize=(7.2,5.4))
        fig.subplots_adjust(left=.08,right=.985,bottom=.12,top=.80,wspace=.34,hspace=.66)
        for repeat,ax in enumerate(list(axes.flat)[:5]):
            record=records[level,source,repeat]
            draw_histogram(ax,record,10)
            ax.set_title(f"Repeat {repeat+1}: RMSE {record['rmse']:.3g}\n95%: [{record['lower']:.3g}, {record['upper']:.3g}]",fontsize=10,pad=5)
        axes[-1,-1].axis('off');axes[-1,-1].legend(handles=legend_handles(),loc='center',frameon=False,fontsize=10,handlelength=1.5)
        label=family_label(level,source).replace('\n','; ')
        fig.suptitle(f'{label}: learned-null RDR\nMerged held-out test; selected JS configuration',fontsize=13,y=.98)
        fig.supxlabel(r'Fitted RDR $\widehat r$',fontsize=12,y=.01)
        fig.supylabel('Density',fontsize=12,x=.005)
        stem=f'null_rdr_histograms_{level}_{source}';save(fig,output,stem);stems.append(stem)
    return stems


def main(args):
    merged=args.merged.resolve();output=args.output.resolve()
    require(not output.exists(),'Use a new presentation output directory')
    completion=sealed(merged/'COMPLETE.json');spec=read(merged/'protocol.json')
    require(completion['main_evaluations']==20 and completion['null_evaluations']==25,'Incomplete merged study')
    for name,digest in completion['files'].items():require(sha(merged/name)==digest,'Merged artifact changed')
    run=Path(spec['main_parent']);null=Path(spec['null_parent'])
    require(sha(run/'COMPLETE.json')==spec['main_parent_sha256'],'Main parent changed')
    require(sha(null/'COMPLETE.json')==spec['null_parent_sha256'],'Null parent changed')
    protocol,metadata,results,input_hashes=verify_run(run)
    images,manifests=load_main_data(run,metadata,input_hashes)
    output.mkdir(parents=True)
    selected=[];ranges=[];stems=[]
    for i,(level,branch) in enumerate(FAMILIES[:4]):
        counts={role:{side:metadata['roles'][role][src]['count'] for side,src in (('p','real'),('q',branch))} for role in ROLES}
        pooled=merge_scores(results[level,branch,0]['predictions'],counts)
        stem,rows,groups=compact_panels(output,level,branch,pooled,images,manifests,spec['seed']+i*1000,digits=3)
        stems.append(stem);selected+=rows;ranges+=groups
    previous=pd.read_csv(merged/'selected_images.csv',float_precision='round_trip')
    previous=previous[previous.panel=='ranked']
    current=pd.DataFrame(selected)
    identity=['representation','branch','source','target','display_rank','test_row','source_id','rdr']
    for frame in (previous,current):frame.sort_values(identity[:5],inplace=True)
    pd.testing.assert_frame_equal(previous[identity].reset_index(drop=True),current[identity].reset_index(drop=True),check_dtype=False,check_exact=True)
    current.to_csv(output/'selected_images.csv',index=False);pd.DataFrame(ranges).to_csv(output/'image_ranges.csv',index=False)
    records={};summary=[];histogram_rows=[];edges=np.linspace(0,2,401)
    null_protocol=sealed(null/'protocol.json')
    for task in null_protocol['tasks']:
        level,source,repeat=task['representation'],task['source'],task['repeat']
        directory=null/'evaluation'/level/source/f'repeat_{repeat:02d}'
        receipt=read(directory/'COMPLETE.json')
        require(receipt['protocol_sha256']==sha(null/'protocol.json'),'Null task protocol changed')
        require(receipt['task_identity']==[task[k] for k in ('representation','branch','candidate','repeat')],'Null task identity changed')
        path=directory/'predictions.npz';require(sha(path)==receipt['files']['predictions.npz'],'Null scores changed')
        input_hashes[str(path)]=receipt['files']['predictions.npz']
        with np.load(path,allow_pickle=False) as values:
            a,b=[np.concatenate([values[f'{role}_{side}'] for role in ROLES]).astype(float) for side in ('p','q')]
        metric=read(merged/'null'/level/source/f'repeat_{repeat:02d}'/'metrics.json')
        require((len(a),len(b))==(metric['n_test_p'],metric['n_test_q']),'Null test count changed')
        rmse=float(np.sqrt(.5*np.mean((a-1)**2)+.5*np.mean((b-1)**2)))
        require(np.isclose(rmse,metric['rdr_rmse_one'],rtol=1e-12,atol=0),'Null RMSE changed')
        low,high=balanced_quantiles(a,b)
        counts_a=np.histogram(a,bins=edges)[0];counts_b=np.histogram(b,bins=edges)[0]
        da=counts_a/len(a)/np.diff(edges);db=counts_b/len(b)/np.diff(edges)
        records[level,source,repeat]=dict(edges=edges,density_a=da,density_b=db,lower=float(low),upper=float(high),rmse=rmse)
        row=dict(representation=level,source=source,repeat=repeat,display_repeat=repeat+1,n_a=len(a),n_b=len(b),
                 central95_lower=float(low),central95_upper=float(high),rdr_rmse_one=rmse)
        summary.append(row)
        for j in range(400):histogram_rows.append({**row,'bin':j,'left':float(edges[j]),'right':float(edges[j+1]),
                                                   'count_a':int(counts_a[j]),'count_b':int(counts_b[j]),'density_a':float(da[j]),'density_b':float(db[j])})
    require(len(records)==25,'Expected all 25 learned null fits')
    stems+=render_histograms(output,records)
    pd.DataFrame(summary).to_csv(output/'null_histogram_summary.csv',index=False)
    pd.DataFrame(histogram_rows).to_csv(output/'null_histogram_bins.csv',index=False)
    lines=['# CelebA publication figures: three-digit ranges and null histograms','',
           '**Complete: four unchanged image selections with shorter labels; histograms for all 25 learned null fits.**','',
           'Image endpoints use three significant digits, retaining scientific notation for small scores. '
           'The image identities, ordering, scores, and numerical experiment results are unchanged. '
           'Underlying CSVs retain full precision.','',
           '## Null RDR histograms','',
           'The overview has five families in rows and five fitted repetitions in columns. '
           'Each panel overlays density histograms for disjoint source folds A and B and marks the known null RDR 1. '
           'Both original final roles are merged within each fold. Repeats 1–5 correspond to saved repeats 00–04. '
           'All score histograms share 400 equal bins over [0,2]; each fold histogram integrates to one.','',
           'The shaded central 95% is the [0.025,0.975] inverse empirical CDF interval of the equally weighted '
           'A/B score distributions. It describes fitted-score dispersion, not a confidence interval or a '
           'cell-calibration coverage claim. Panel titles report balanced RDR RMSE against the known truth 1. '
           'There is no score clipping, exact-one substitution, refitting, or best-repeat selection.','',
           '![All learned-null RDR histograms](null_rdr_histograms.png)','',
           '[Overview PDF](null_rdr_histograms.pdf). For publication, the five family PDFs below are '
           '7.2 inches wide with 10-point ticks/panel titles and 12-point axis labels; they avoid shrinking '
           'a 25-panel overview to an unreadable size. The overview uses 19-point ticks and panel titles.','']
    for level,source in FAMILIES:
        stem=f'null_rdr_histograms_{level}_{source}'
        lines += [f'- {level} / {source}: [PDF]({stem}.pdf) · [PNG]({stem}.png)']
    lines += ['', '[Histogram summaries](null_histogram_summary.csv) · [All bin counts and densities](null_histogram_bins.csv)','',
              '## Compact image samples','']
    for level,branch in FAMILIES[:4]:
        stem=f'images_{level}_{branch}_ranked';lines += [f'![{level} {branch}]({stem}.png)',f'[PDF]({stem}.pdf)','']
    lines += ['[Image identities and scores](selected_images.csv) · [Full-precision actual ranges](image_ranges.csv)','',
              'The unchanged merged-test numerical report remains in `merged_test_20261003/RESULTS.md`. '
              'This separate presentation supplement preserves the earlier sealed figures and records its own sources and hashes.','']
    (output/'FIGURES.md').write_text('\n'.join(lines))
    sources={}
    for name in (*SOURCE_FILES,'experiments/CelebA/publication_figures.py'):
        dest=output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest);sources[name]=sha(dest)
    write(output/'protocol.json',dict(merged_parent=str(merged),merged_completion_sha256=sha(merged/'COMPLETE.json'),
          significant_digits=3,main_image_repeat=0,seed=spec['seed'],null_fits=25,histogram_bins=400,
          histogram_range=[0,2],quantile_rule='inverse empirical CDF of 0.5*A+0.5*B',interval_scope='central score range, not a CI',
          title_metric='balanced RDR RMSE against 1',training_performed=False,metrics_changed=False,
          image_selection_changed=False,input_hashes=input_hashes,source_files=sources))
    files={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and 'source' not in p.relative_to(output).parts}
    write(output/'COMPLETE.json',dict(complete=True,null_fits=25,figure_pairs=len(stems),ranked_images=len(selected),
          files=files,source_files=sources,protocol_sha256=sha(output/'protocol.json')))
    (output/'COMPLETE.sha256').write_text(sha(output/'COMPLETE.json')+'\n')
    print(json.dumps(dict(complete=True,figure_pairs=len(stems),null_fits=25,ranked_images=len(selected))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--merged',type=Path,default=Path('/cwork/yx306/RDR/CelebA/merged_test_20261003'))
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
