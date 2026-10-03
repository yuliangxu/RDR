#!/usr/bin/env python3
"""Pool saved final halves, assess frozen CelebA models, and render paper panels."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/celeba-merged-test-mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.CelebA.final_figures import read, sha, sealed, verify_run
from experiments.CelebA.null_report import _plots as plot_null_summary
from utils.model_selection import balanced_brier, calibration_diagnostics
from utils.calibration import bin_scores

ROLES = ('test_calibration', 'test_evaluation')
METRICS = ('brier', 'local_gap', 'supported_mass', 'c2_distance', 'c2_width',
           'middle_mass', 'middle_local_gap', 'rdr_mse_one', 'rdr_rmse_one',
           'rdr_mae_one', 'rdr_mean_p', 'rdr_mean_q')
SOURCE_FILES = ('experiments/CelebA/merged_test.py', 'experiments/CelebA/final_figures.py',
                'experiments/CelebA/null_report.py', 'utils/model_selection.py',
                'utils/calibration.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def merge_scores(predictions, counts):
    """Concatenate in the exact same role order as the row/image manifests."""
    pooled = {}
    for side in ('p', 'q'):
        pieces = []
        for role in ROLES:
            values = np.asarray(predictions[f'{role}_{side}'], dtype=float)
            require(values.ndim == 1 and len(values) == counts[role][side], 'Score/count mismatch')
            require(np.isfinite(values).all() and np.all((values >= 0) & (values <= 2)), 'Invalid scores')
            pieces.append(values)
        pooled[side] = np.concatenate(pieces)
    return pooled


def diagnostics(pooled, alpha=.05, bins=20, null=False):
    p, q = pooled['p'], pooled['q']
    summary, cells = calibration_diagnostics(p, q, p, q, alpha=alpha, bins=bins)
    for key in ('n_cal_p', 'n_cal_q', 'n_eval_p', 'n_eval_q'):
        summary.pop(key)
    summary.update(brier=balanced_brier(p, q), n_test_p=len(p), n_test_q=len(q),
                   assessment_scope='same_merged_heldout_test_for_counts_and_neural_summaries')
    for cell in cells:
        for side in ('p', 'q'):
            require(cell[f'cal_count_{side}'] == cell[f'eval_count_{side}'], 'Unequal shared-pool counts')
            cell[f'test_count_{side}'] = cell.pop(f'cal_count_{side}')
            cell.pop(f'eval_count_{side}')
        cell['test_mass'] = cell.pop('eval_mass')
        cell.pop('calibration_mass')
    require(np.isclose(summary['supported_mass'], 1), 'Shared-pool support must be one')
    if null:
        mse = float(.5*np.mean((p-1)**2) + .5*np.mean((q-1)**2))
        summary.update(rdr_mse_one=mse, rdr_rmse_one=float(np.sqrt(mse)),
                       rdr_mae_one=float(.5*np.abs(p-1).mean()+.5*np.abs(q-1).mean()),
                       rdr_mean_p=float(p.mean()), rdr_mean_q=float(q.mean()),
                       excess_brier_true=mse/4, empirical_brier_excess=summary['brier']-.25)
    return summary, cells


def ranked_indices(scores, target, maximum=40, seed=2026100301):
    values = np.asarray(scores, dtype=float)
    require(values.ndim == 1 and np.isfinite(values).all() and np.all((values >= 0) & (values <= 2)),
            'Invalid ranking scores')
    require(target in (0, 1, 2), 'Unknown ranking target')
    # Shuffle only to break exact ties; no prior score-bin restrictions.
    order = np.random.default_rng(seed).permutation(len(values))
    return order[np.argsort(np.abs(values[order]-target), kind='stable')[:maximum]]


class JoinedImages:
    def __init__(self, pieces):
        self.pieces = pieces
        self.offsets = np.cumsum([0] + [len(p) for p in pieces])

    def __getitem__(self, index):
        require(0 <= index < self.offsets[-1], 'Image index out of bounds')
        part = int(np.searchsorted(self.offsets, index, side='right') - 1)
        return self.pieces[part][index-self.offsets[part]]


def montage(images, indices, columns=10, capacity=40):
    rows = (capacity+columns-1)//columns
    canvas = np.full((rows*65-1, columns*65-1, 3), 255, dtype=np.uint8)
    for position, index in enumerate(indices):
        row, column = divmod(position, columns)
        canvas[row*65:row*65+64, column*65:column*65+64] = images[int(index)].transpose(1, 2, 0)
    return canvas


def save(fig, output, stem):
    for extension in ('png', 'pdf'):
        fig.savefig(output/f'{stem}.{extension}', dpi=180, bbox_inches='tight', pad_inches=.025,
                    metadata={'CreationDate': None, 'ModDate': None} if extension == 'pdf' else None)
    plt.close(fig)
    return stem


def draw_cells(ax, cells, title, null=False):
    edges = np.array([r['left'] for r in cells]+[cells[-1]['right']])
    x = (edges[:-1]+edges[1:])/2
    lower, upper = [np.array([r[k] for r in cells]) for k in ('c2_lower', 'c2_upper')]
    ax.fill_between(edges, np.r_[lower,lower[-1]], np.r_[upper,upper[-1]], step='post',
                    color='#8cb9d5', alpha=.6, label='C.2: simultaneous across cells')
    available = np.array([not r['c1_unavailable'] for r in cells])
    estimate = np.array([np.nan if r['calibrated_rdr'] is None else r['calibrated_rdr'] for r in cells])
    low, high = [np.array([np.nan if r[k] is None else r[k] for r in cells]) for k in ('c1_lower','c1_upper')]
    ax.errorbar(x[available], estimate[available],
                yerr=[estimate[available]-low[available], high[available]-estimate[available]],
                fmt='none', ecolor='#b75d23', capsize=2, label='C.1: marginal')
    ax.stairs(estimate, edges, baseline=None, color='#152b3c', label='Test cell-ratio estimate')
    neural = [np.nan if r['neural_mean'] is None else r['neural_mean'] for r in cells]
    ax.scatter(x, neural, s=19, facecolors='none', edgecolors='#b42363', label='Test neural cell mean')
    if np.any(~available):
        ax.scatter(x[~available], np.full(np.sum(~available),-.04), marker='x', color='#b75d23',
                   label='C.1 unavailable', clip_on=False)
    ax.axhline(1, color='.4', ls='--', lw=.8, label='Null truth = 1' if null else None)
    ax.set(title=title, xlim=(0,2), ylim=(-.08,2.05), ylabel='Cell-average RDR')
    ax.grid(alpha=.15)


def draw_counts(ax, cells):
    x = np.array([(r['left']+r['right'])/2 for r in cells])
    for offset, side, color, label in ((-.018,'p','#367c98','P / A'),(.018,'q','#c48a39','Q / B')):
        ax.bar(x+offset, [r[f'test_count_{side}'] for r in cells], width=.034, color=color, label=label)
    ax.set_yscale('symlog', linthresh=1)
    ax.set(ylabel='Test count', xlabel='Fixed neural-score bin')
    ax.legend(fontsize=8, ncol=2, frameon=False)


def ci_main(output, results, level, repeat):
    fig, axes = plt.subplots(2,2,figsize=(11,5.7),sharex=True, layout='constrained',
                             gridspec_kw={'height_ratios':[3,1]})
    for col, branch in enumerate(('lower','upper')):
        cells = results[level,branch,repeat]['cells']
        draw_cells(axes[0,col],cells,f'$P$ vs $Q_{"l" if branch == "lower" else "u"}$')
        draw_counts(axes[1,col],cells)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False,fontsize=8)
    fig.suptitle(f'CelebA {level}: merged held-out test, repeat {repeat:02d}\n'
                 'Nominal 95% cell-average intervals; same test pool for both estimates',fontsize=11)
    return save(fig,output,f'ci_{level}_repeat_{repeat:02d}')


def compact_panels(output, level, branch, pooled, images, manifests, seed, digits=3):
    letter = 'l' if branch == 'lower' else 'u'
    fig = plt.figure(figsize=(15,6.1))
    fig.suptitle(f'{level.capitalize()}-level RDR: $P$ vs $Q_{letter}$',y=.99,
                 fontsize=19,fontweight='bold',fontfamily='DejaVu Serif')
    fig.text(.5,.915,'$P$: real images',ha='center',fontsize=17,fontweight='bold')
    fig.text(.5,.445,f'$Q_{letter}$: generated images',ha='center',fontsize=17,fontweight='bold')
    records, ranges = [], []
    for side_no,(side,source) in enumerate((('p','real'),('q',branch))):
        values = pooled[side]
        for target,label in enumerate(('Smallest RDR','Closest to 1','Largest RDR')):
            tie_seed = seed+side_no*17+target
            selected = ranked_indices(values,target,40,tie_seed)
            lo,hi = float(values[selected].min()),float(values[selected].max())
            ax = fig.add_axes([.008+target*.331,.55 if side_no==0 else .08,.322,.30])
            ax.imshow(montage(images[source],selected),interpolation='nearest')
            ax.set_title(f'{label}\n$n={len(selected)}$, range [{lo:.{digits}g}, {hi:.{digits}g}]',fontsize=10,pad=4)
            ax.axis('off')
            base=dict(representation=level,branch=branch,repeat=0,panel='ranked',source=source,
                      group=label,target=target,sampling_seed=tie_seed,shown_count=len(selected),
                      available_count=len(values),score_min=lo,score_max=hi)
            ranges.append(base)
            for rank,index in enumerate(selected):
                row=manifests[source].iloc[int(index)]
                records.append({**base,'display_rank':rank,'test_row':int(index),'source_id':str(row.source_id),
                                'original_final_role':row.original_final_role,'rdr':float(values[index])})
    return save(fig,output,f'images_{level}_{branch}_ranked'),records,ranges


def bin_panels(output, level, branch, pooled, images, manifests, seed):
    edges=np.linspace(0,2,21)
    fig,axes=plt.subplots(10,4,figsize=(12,12),layout='constrained')
    records,ranges=[],[]
    for side_no,(side,source) in enumerate((('p','real'),('q',branch))):
        assigned=bin_scores(pooled[side],edges)
        for j in range(20):
            pool=np.flatnonzero(assigned==j)
            selected=np.sort(np.random.default_rng(seed+j*17+side_no).choice(pool,min(4,len(pool)),replace=False))
            ax=axes[j%10,2*(j//10)+side_no]
            ax.imshow(montage(images[source],selected,4,4),interpolation='nearest');ax.axis('off')
            label='P' if side=='p' else f'Q_{"l" if branch=="lower" else "u"}'
            ax.set_title(f'{label}: [{edges[j]:.1f}, {edges[j+1]:.1f}{"]" if j==19 else ")"}; '
                         f'{len(selected)}/{len(pool)} shown',fontsize=8,pad=2)
            if not len(pool):ax.text(.5,.5,'Empty test bin',transform=ax.transAxes,ha='center',fontsize=8)
            base=dict(representation=level,branch=branch,repeat=0,panel='all_bins',source=source,bin=j,
                      left=float(edges[j]),right=float(edges[j+1]),shown_count=len(selected),available_count=len(pool),
                      sampling_seed=seed+j*17+side_no)
            ranges.append(base)
            for rank,index in enumerate(selected):
                row=manifests[source].iloc[int(index)]
                records.append({**base,'display_rank':rank,'test_row':int(index),'source_id':str(row.source_id),
                                'original_final_role':row.original_final_role,'rdr':float(pooled[side][index])})
    fig.suptitle(f'CelebA {level}: P vs Q_{"l" if branch=="lower" else "u"}; merged test, repeat 00\n'
                 'Up to four uniformly sampled images per source/bin; empty bins retained',fontsize=11)
    return save(fig,output,f'images_{level}_{branch}_all_bins'),records,ranges


def aggregate(rows, source_key):
    result=[]
    for level,source in dict.fromkeys((r['representation'],r[source_key]) for r in rows):
        group=[r for r in rows if (r['representation'],r[source_key])==(level,source)]
        require(sorted(r['repeat'] for r in group)==list(range(5)), 'Missing/duplicate repeat')
        item=dict(representation=level,source=source,branch=source,repeats=5,expected_repeats=5,
                  complete=True,failed_repeats=0,pending_repeats=0,n_test_p=group[0]['n_test_p'],n_test_q=group[0]['n_test_q'])
        for key in METRICS:
            values=[r.get(key) for r in group if r.get(key) is not None]
            item[key+'_mean']=float(np.mean(values)) if len(values)==5 else None
            item[key+'_sd']=float(np.std(values,ddof=1)) if len(values)==5 else None
            item[key+'_available_repeats']=len(values)
        result.append(item)
    return result


def table(summaries, metrics):
    lines=['| Comparison | '+' | '.join(metrics)+' |','| --- | '+' | '.join(['---']*len(metrics))+' |']
    for r in summaries:
        values=[f"{r[k+'_mean']:.6f} ({r[k+'_sd']:.6f})" if r[k+'_mean'] is not None else 'Unavailable (not all repeats)' for k in metrics]
        lines.append(f"| {r['representation']} / {r['source']} | "+' | '.join(values)+' |')
    return lines


def load_main_data(run, metadata, hashes):
    images,manifests={},{}
    for source in ('real','lower','upper'):
        all_roles={}
        for role,item in metadata['roles'].items():
            record=item[source];path=run/record['path']
            require(sha(path)==record['receipt']['sha256'], 'Manifest changed')
            hashes[str(path)]=sha(path)
            frame=pd.read_csv(path)
            require(len(frame)==record['count'] and frame.source_id.is_unique,'Manifest row count/duplicates')
            all_roles[role]=frame
        for i,role in enumerate(all_roles):
            for other in list(all_roles)[i+1:]:
                require(not set(all_roles[role].source_id)&set(all_roles[other].source_id),'Source rows overlap roles')
                if source=='real':
                    require(not set(all_roles[role].identity)&set(all_roles[other].identity),'Real identities overlap roles')
        parts=[];frames=[]
        for role in ROLES:
            receipt=metadata['arrays']['pixel'][role][source];path=run/receipt['path']
            require(sha(path)==receipt['sha256'],'Pixel array changed')
            require(receipt['manifest_sha256']==metadata['roles'][role][source]['receipt']['sha256'],'Image ordering changed')
            array=np.load(path,mmap_mode='r',allow_pickle=False)
            require(array.dtype==np.uint8 and list(array.shape)==receipt['shape'],'Pixel array shape/dtype')
            parts.append(array);hashes[str(path)]=receipt['sha256']
            frame=all_roles[role].copy();frame['original_final_role']=role;frames.append(frame)
        images[source]=JoinedImages(parts);manifests[source]=pd.concat(frames,ignore_index=True)
    return images,manifests


def main(args):
    output=args.output.resolve()
    require(not output.exists(),'Use a new output directory; prior results are immutable')
    run=args.main_run.resolve();null=args.null_run.resolve()
    protocol,metadata,old_results,hashes=verify_run(run)
    require(protocol['config']['bins']==20 and protocol['config']['ci_alpha']==.05,'Unexpected CI configuration')
    images,manifests=load_main_data(run,metadata,hashes)
    output.mkdir(parents=True)
    figures=output/'figures';figures.mkdir()
    for source,frame in manifests.items():
        frame[['source_id','original_final_role','identity']].to_csv(output/f'test_manifest_{source}.csv',index=False)
    rows=[];results={};allcells=[]
    for key,old in old_results.items():
        level,branch,repeat=key
        counts={role:{side:metadata['roles'][role][src]['count'] for side,src in (('p','real'),('q',branch))} for role in ROLES}
        pooled=merge_scores(old['predictions'],counts)
        require((len(pooled['p']),len(pooled['q']))==(39829,40000),'Unexpected merged main counts')
        summary,cells=diagnostics(pooled,protocol['config']['ci_alpha'],protocol['config']['bins'])
        task=old['task'];row={**task,**summary};rows.append(row)
        results[key]=dict(pooled=pooled,cells=cells)
        target=output/'main'/level/branch/f'repeat_{repeat:02d}'
        write(target/'metrics.json',row);write(target/'cells.json',cells)
        allcells += [{**c,'study':'main','representation':level,'source':branch,'repeat':repeat} for c in cells]
    require(len(rows)==20,'Expected 20 main evaluations')
    summaries=aggregate(rows,'branch')
    pd.DataFrame(rows).to_csv(output/'main_per_fit.csv',index=False)
    pd.DataFrame(summaries).to_csv(output/'main_per_comparison.csv',index=False)
    stems=[];selected=[];ranges=[]
    for level in ('feature','pixel'):
        for repeat in range(5):stems.append(ci_main(figures,results,level,repeat))
    for i,(level,branch) in enumerate((('feature','lower'),('feature','upper'),('pixel','lower'),('pixel','upper'))):
        for renderer in (compact_panels,bin_panels):
            stem,records,counts=renderer(figures,level,branch,results[level,branch,0]['pooled'],images,manifests,args.seed+i*1000)
            stems.append(stem);selected+=records;ranges+=counts
    pd.DataFrame(selected).to_csv(output/'selected_images.csv',index=False)
    pd.DataFrame(ranges).to_csv(output/'image_groups.csv',index=False)
    # The null model halves are source A/B, and remain distinct after merging roles.
    null_completion=sealed(null/'COMPLETE.json');null_protocol=sealed(null/'protocol.json')
    require(null_completion['evaluations']==25 and null_completion['protocol_sha256']==sha(null/'protocol.json'),'Incomplete null parent')
    require(null_protocol['parent_completion_sha256']==sha(run/'COMPLETE.json'),'Null parent differs')
    for name,digest in null_protocol['files'].items():require(sha(null/name)==digest,'Null frozen source/data changed')
    null_meta=read(null/'data/null_data.json');null_rows=[];null_cells={}
    require(null_protocol['config']['bins']==20 and null_protocol['config']['ci_alpha']==.05,'Unexpected null CI configuration')
    for source in ('real','lower','upper'):
        halves={}
        for side in ('p','q'):
            frames=[]
            for role in ROLES:
                receipt=null_meta['roles'][role][source]['sides'][side]['manifest_receipt']
                path=null/receipt['path'];require(sha(path)==receipt['sha256'],'Null manifest changed')
                hashes[str(path)]=receipt['sha256']
                frame=pd.read_csv(path);frame['original_final_role']=role;frames.append(frame)
            halves[side]=pd.concat(frames,ignore_index=True)
            require(halves[side].source_id.is_unique,'Repeated null source ID')
            halves[side][['source_id','original_final_role','identity']].to_csv(output/f'null_test_manifest_{source}_{side}.csv',index=False)
        require(not set(halves['p'].source_id)&set(halves['q'].source_id),'Null A/B image overlap')
        require(set(halves['p'].source_id)|set(halves['q'].source_id)==set(manifests[source].source_id),'Null final union differs from main pool')
        if source=='real':require(not set(halves['p'].identity)&set(halves['q'].identity),'Null A/B identity overlap')
    for task in null_protocol['tasks']:
        level,source,repeat=task['representation'],task['source'],task['repeat']
        directory=null/'evaluation'/level/source/f'repeat_{repeat:02d}'
        receipt=read(directory/'COMPLETE.json')
        require(receipt['protocol_sha256']==sha(null/'protocol.json'),'Null task protocol changed')
        require(receipt['task_identity']==[task[k] for k in ('representation','branch','candidate','repeat')],'Null task identity changed')
        training=null/'fits'/level/source/f'repeat_{repeat:02d}'/'COMPLETE.json'
        require(receipt['training_receipt_sha256']==sha(training),'Null training lineage changed')
        for name,digest in receipt['files'].items():
            require(sha(directory/name)==digest,'Null assessment changed');hashes[str(directory/name)]=digest
        counts={role:{side:null_meta['roles'][role][source]['sides'][side]['count'] for side in ('p','q')} for role in ROLES}
        with np.load(directory/'predictions.npz',allow_pickle=False) as predictions:pooled=merge_scores(predictions,counts)
        require((len(pooled['p']),len(pooled['q']))==((19911,19918) if source=='real' else (20000,20000)), 'Unexpected merged null counts')
        summary,cells=diagnostics(pooled,null_protocol['config']['ci_alpha'],null_protocol['config']['bins'],null=True)
        row={**task,**summary};null_rows.append(row);null_cells[level,source,repeat]=cells
        target=output/'null'/level/source/f'repeat_{repeat:02d}'
        write(target/'metrics.json',row);write(target/'cells.json',cells)
        allcells += [{**c,'study':'null','representation':level,'source':source,'repeat':repeat} for c in cells]
    require(len(null_rows)==25,'Expected 25 null evaluations')
    null_summaries=aggregate(null_rows,'source')
    pd.DataFrame(null_rows).to_csv(output/'null_per_fit.csv',index=False)
    pd.DataFrame(null_summaries).to_csv(output/'null_per_comparison.csv',index=False)
    pd.DataFrame(allcells).to_csv(output/'cell_intervals.csv',index=False)
    plot_null_summary(figures,null_summaries,null_rows);stems.append('null_summary')
    fig,axes=plt.subplots(3,2,figsize=(11,9),layout='constrained')
    for ax,item in zip(axes.flat,null_summaries):
        level,source=item['representation'],item['source']
        draw_cells(ax,null_cells[level,source,0],f'{level} / {source}: A vs B',null=True)
        ax.set_xlabel('Fixed neural-score bin')
    handles,labels=axes[0,0].get_legend_handles_labels();axes[-1,-1].axis('off')
    axes[-1,-1].legend(handles,labels,loc='center',frameon=False,fontsize=9)
    fig.suptitle('CelebA nulls: merged held-out test, repeat 00\nNominal 95% cell-average intervals; same test pool for both estimates')
    stems.append(save(fig,figures,'null_ci_repeat_00'))
    lines=['# CelebA: one merged held-out test pool','',
           '**Complete: 20 selected-model and 25 null assessments; no retraining or model selection.**','',
           'Former final-calibration rows followed by final-evaluation rows form one test pool. '
           'The identical observations supply cell-count C.1/C.2 intervals, neural means, Brier, Gap and example images. '
           'Main counts: 39,829 P and 40,000 per Q; generated null counts: 20,000 per A/B; real null counts: 19,911 A and 19,918 B.','',
           'Test rows are disjoint from training and development rows. Historically inspected design/test observations '
           'still make this a retrospective held-out analysis, not a new untouched prospective study. '
           'Networks, configurations and the 20 score bins remain fixed.','',
           '## Main comparisons','', 'Mean (SD) over five training repetitions on the same merged test pool.','',
           *table(summaries,('brier','local_gap','supported_mass','middle_mass','middle_local_gap')),'',
           '## Learned null controls','',*table(null_summaries,('brier','local_gap','rdr_mse_one','rdr_rmse_one')),'',
           'Supported mass is 1 by construction: every occupied evaluation cell uses those same observations '
           'for its cell-count estimate. This is not separate evidence of calibration. Local Gap uses two '
           'correlated estimates on the same pool; the cell CIs are not CIs for their difference. '
           'C.1/C.2 target cell-average population RDR, not an individual image or the estimated neural mean. '
           'Intervals remain nominal image-level diagnostics, without within-identity dependence adjustment; '
           'C.2 simultaneity is within one comparison. Repetition SD measures training variation, not data-resampling uncertainty.','',
           '## Publication image panels','',
           'Repeat 00 is fixed, not selected for appearance or performance. For each source separately, '
           'choose 40 globally smallest RDRs, 40 nearest to 1, and 40 largest RDRs, with seeded tie breaking. '
           'There are no preset broad score intervals; panel labels give the actual displayed minimum and maximum '
           '(three significant digits). Groups are selected independently, so overlap is possible. '
           'No images or scores are generated or altered. Examples do not represent prevalence.','']
    for level,branch in (('feature','lower'),('feature','upper'),('pixel','lower'),('pixel','upper')):
        stem=f'images_{level}_{branch}_ranked'
        lines += [f'![{level} {branch} ranked examples](figures/{stem}.png)',f'[PDF](figures/{stem}.pdf)','']
    lines += ['## All figures','']+[f'- [{stem} PNG](figures/{stem}.png) · [PDF](figures/{stem}.pdf)' for stem in stems]
    lines += ['', '[Main per-fit metrics](main_per_fit.csv) · [Main aggregates](main_per_comparison.csv) · '
              '[Null per-fit metrics](null_per_fit.csv) · [Null aggregates](null_per_comparison.csv)',
              '[All 900 cell records](cell_intervals.csv) · [Image IDs and scores](selected_images.csv) · '
              '[Group ranges and counts](image_groups.csv) · [Protocol](protocol.json)','']
    (output/'RESULTS.md').write_text('\n'.join(lines))
    sources={}
    for name in SOURCE_FILES:
        path=output/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,path);sources[name]=sha(path)
    record=dict(schema=1,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                main_parent=str(run),null_parent=str(null),main_parent_sha256=sha(run/'COMPLETE.json'),
                null_parent_sha256=sha(null/'COMPLETE.json'),roles=list(ROLES),same_test_pool=True,
                counts={'main':{'p':39829,'q':40000},'generated_null':{'p':20000,'q':20000},'real_null':{'p':19911,'q':19918}},
                training_performed=False,selection_performed=False,retrospective=True,seed=args.seed,
                bins=20,ci_alpha=.05,illustration_repeat=0,ranking_rule='global nearest 0/1/2, seeded exact ties',
                source_files=sources,input_hashes=hashes)
    write(output/'protocol.json',record)
    artifacts={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and 'source' not in p.relative_to(output).parts}
    completion=dict(complete=True,main_evaluations=20,null_evaluations=25,figure_pairs=len(stems),files=artifacts,
                    protocol_sha256=sha(output/'protocol.json'),source_files=sources)
    write(output/'COMPLETE.json',completion)
    (output/'COMPLETE.sha256').write_text(sha(output/'COMPLETE.json')+'\n')
    print(json.dumps({k:completion[k] for k in ('complete','main_evaluations','null_evaluations','figure_pairs')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-run',type=Path,default=Path('/cwork/yx306/RDR/CelebA/final_evaluation_expanded_20261003'))
    parser.add_argument('--null-run',type=Path,default=Path('/cwork/yx306/RDR/CelebA/null_expanded_20261003'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=2026100301)
    main(parser.parse_args())
