#!/usr/bin/env python3
"""Joint CelebA attribute regressions of stable log(2-RDR), on real test images."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/celeba-attribute-mpl')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import torch
from experiments.CelebA.final_figures import read, sealed, sha, verify_run
from experiments.CelebA.selection_training import make_model, _batch

ROLES = ('test_calibration', 'test_evaluation')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stable_log_complement(scaled_logits):
    values = np.asarray(scaled_logits, dtype=np.float64)
    require(np.isfinite(values).all(), 'Nonfinite logits')
    return np.log(2.) - np.logaddexp(0., values)


def align_attributes(path, manifest):
    """Join annotations by filename, never by incidental annotation row order."""
    with path.open() as stream:
        count = int(stream.readline()); names = stream.readline().split()
    table = pd.read_csv(path, sep=r'\s+', skiprows=2, names=['source_id'] + names)
    require(len(names) == 40 and len(table) == count, 'Unexpected annotation dimensions')
    require(table.source_id.is_unique and manifest.source_id.is_unique, 'Duplicate image IDs')
    table = table.set_index('source_id')
    require(set(manifest.source_id).issubset(table.index), 'Missing test annotations')
    attributes = table.loc[manifest.source_id, names].to_numpy()
    require(np.isin(attributes, [-1, 1]).all(), 'Invalid attribute encoding')
    return (attributes == 1).astype(float), names


def regress(response, attributes, names, identities):
    """OLS point estimates as in DDIM; uncertainty clustered by real identity."""
    y = np.asarray(response, dtype=float)
    x = np.column_stack((np.ones(len(y)), attributes))
    require(np.isfinite(y).all() and len(identities) == len(y), 'Invalid regression rows')
    require(np.linalg.matrix_rank(x) == x.shape[1], 'Rank-deficient attribute design')
    fit = sm.OLS(y, x).fit(cov_type='cluster', cov_kwds={'groups': identities, 'use_correction': True}, use_t=True)
    ci = fit.conf_int()[1:]
    # Drop-one partial R^2 from the full-model residual sum of squares and
    # coefficient-specific extra sum of squares; no 40-model refitting needed.
    extra_ss = fit.params[1:]**2 / np.diag(fit.normalized_cov_params)[1:]
    table = pd.DataFrame(dict(attribute=names, coef=fit.params[1:], cluster_se=fit.bse[1:],
        t=fit.tvalues[1:], pvalue=fit.pvalues[1:], ci_low=ci[:, 0], ci_high=ci[:, 1],
        bh_qvalue=multipletests(fit.pvalues[1:], method='fdr_bh')[1],
        prevalence=attributes.mean(axis=0), coef_per_sd=fit.params[1:]*attributes.std(axis=0),
        partial_r2=extra_ss/(extra_ss+fit.ssr)))
    metrics = dict(n=len(y), identities=len(np.unique(identities)), design_rank=x.shape[1],
        r_squared=float(fit.rsquared), adjusted_r_squared=float(fit.rsquared_adj),
        intercept=float(fit.params[0]), residual_rmse=float(np.sqrt(fit.ssr/len(y))),
        response_min=float(y.min()), response_max=float(y.max()), response_mean=float(y.mean()),
        design_condition_number=float(fit.condition_number))
    return table, metrics


@torch.no_grad()
def recover_response(model, arrays, alpha, batch_size):
    model.eval(); scores=[]; logits=[]
    for values in arrays:
        for start in range(0, len(values), batch_size):
            raw = model(_batch(values, slice(start, start+batch_size), 'cpu', 'feature'))
            scaled = alpha * raw
            scores.append((2*torch.sigmoid(scaled)).double().numpy())
            logits.append(scaled.double().numpy())
    logits = np.concatenate(logits)
    return stable_log_complement(logits), np.concatenate(scores), logits


def plots(output, coefficients, summary):
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.3), layout='constrained')
    colors = plt.get_cmap('tab10').colors
    for ax, branch, label in zip(axes, ('lower','upper'), ('l','u')):
        selected = summary[summary.branch == branch].nsmallest(15, 'rank_abs_mean')
        y = np.arange(len(selected))
        for repeat in range(5):
            rows = coefficients[(coefficients.branch == branch)&(coefficients.repeat == repeat)].set_index('attribute').loc[selected.attribute]
            ax.scatter(rows.coef, y+(repeat-2)*.09, s=16, color=colors[repeat], alpha=.7, label=f'Repeat {repeat+1}')
        ax.scatter(selected.coef_mean, y, s=36, color='black', marker='d', label='Mean')
        ax.axvline(0, color='.5', lw=.8)
        ax.set_yticks(y, selected.attribute.str.replace('_',' '), fontsize=10)
        ax.invert_yaxis(); ax.grid(axis='x', alpha=.2)
        ax.set_title(rf'$P$ vs $Q_{label}$', fontsize=15)
        ax.set_xlabel(r'Coefficient in $\log(2-\widehat r)$', fontsize=12)
        ax.tick_params(axis='x', labelsize=10)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, loc='outside lower center', ncol=6)
    fig.suptitle('Joint attribute associations: top 15 by absolute mean coefficient', fontsize=14)
    for ext in ('png','pdf'):
        fig.savefig(output/f'attribute_coefficients.{ext}', dpi=220, bbox_inches='tight',
                    metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else None)
    plt.close(fig)


def main(args):
    output=args.output.resolve(); merged=args.merged.resolve()
    require(not output.exists(), 'Use a new output directory')
    receipt=sealed(merged/'COMPLETE.json'); spec=read(merged/'protocol.json')
    for name,digest in receipt['files'].items(): require(sha(merged/name)==digest, 'Merged artifact changed')
    run=Path(spec['main_parent']);require(sha(run/'COMPLETE.json')==spec['main_parent_sha256'], 'Main parent changed')
    protocol,metadata,results,inputs=verify_run(run)
    for name in ('experiments/CelebA/selection_training.py','utils/networks.py','utils/losses.py'):
        require(sha(ROOT/name)==protocol['files']['source/'+name], 'Inference source differs from frozen model')
    manifest=pd.read_csv(merged/'test_manifest_real.csv')
    pieces=[]; arrays=[]
    for role in ROLES:
        row=metadata['roles'][role]['real'];path=run/row['path']
        require(sha(path)==row['receipt']['sha256'], 'Manifest changed')
        part=pd.read_csv(path);pieces.append(part[['source_id','identity']])
        info=metadata['arrays']['feature'][role]['real'];path=run/info['path']
        require(sha(path)==info['sha256'] and info['manifest_sha256']==row['receipt']['sha256'], 'Feature data changed')
        values=np.load(path,mmap_mode='r');require(list(values.shape)==info['shape'], 'Feature shape changed')
        arrays.append(values);inputs[str(path)]=sha(path)
    expected=pd.concat(pieces,ignore_index=True)
    pd.testing.assert_frame_equal(expected,manifest[['source_id','identity']],check_dtype=False)
    attributes,names=align_attributes(args.attributes,manifest)
    identities=manifest.identity.to_numpy();require(len(manifest)==39829 and len(np.unique(identities))==1985,'Unexpected real test pool')
    inputs[str(args.attributes.resolve())]=sha(args.attributes)
    inputs[str(merged/'test_manifest_real.csv')]=sha(merged/'test_manifest_real.csv')
    output.mkdir(parents=True); (output/'responses').mkdir()
    design=manifest.copy()
    for j,name in enumerate(names):design[name]=attributes[:,j].astype(int)
    design.to_csv(output/'design.csv',index=False)
    rows=[];metrics=[]
    for branch in ('lower','upper'):
        for repeat in range(5):
            task=results['feature',branch,repeat]['task']
            fitdir=Path(protocol['parent'])/'fits/feature'/branch/task['candidate']/f'repeat_{repeat:02d}'
            require(sha(fitdir/'COMPLETE.json')==protocol['checkpoint_receipts'][str(fitdir)],'Checkpoint receipt changed')
            complete=read(fitdir/'COMPLETE.json');path=fitdir/'model.pt'
            require(sha(path)==complete['files']['model.pt'], 'Checkpoint changed');inputs[str(path)]=sha(path)
            checkpoint=torch.load(path,map_location='cpu',weights_only=True)
            require(checkpoint['task']==task and checkpoint['config']==protocol['config'],'Checkpoint identity changed')
            model=make_model(protocol['config'],repeat,'feature',task['architecture'],'cpu')
            model.load_state_dict(checkpoint['model'],strict=True)
            batch_size=protocol['config']['feature'].get('evaluation_batch_size',4096)
            response,recovered,logits=recover_response(model,arrays,task['output_alpha'],batch_size)
            saved=np.concatenate([results['feature',branch,repeat]['predictions'][f'{role}_p'] for role in ROLES])
            error=float(np.max(np.abs(recovered-saved)))
            require(np.allclose(recovered,saved,rtol=2e-5,atol=3e-6),'Recovered RDR differs from saved evaluation')
            table,metric=regress(response,attributes,names,identities)
            table.insert(0,'repeat',repeat);table.insert(0,'branch',branch);rows.append(table)
            metric.update(branch=branch,repeat=repeat,saved_rdr_equal_two=int(np.sum(saved==2)),max_rdr_recovery_difference=error)
            metrics.append(metric)
            np.savez_compressed(output/'responses'/f'{branch}_{repeat:02d}.npz',scaled_logits=logits,log_2_minus_rdr=response,saved_rdr=saved)
            print(json.dumps(metric),flush=True)
    coefficients=pd.concat(rows,ignore_index=True);coefficients.to_csv(output/'coefficients_per_fit.csv',index=False)
    metrics=pd.DataFrame(metrics);metrics.to_csv(output/'regression_metrics.csv',index=False)
    summary=coefficients.groupby(['branch','attribute'],sort=False).agg(coef_mean=('coef','mean'),coef_sd=('coef','std'),
        coef_min=('coef','min'),coef_max=('coef','max'),coef_per_sd_mean=('coef_per_sd','mean'),
        partial_r2_mean=('partial_r2','mean'),prevalence=('prevalence','first'),positive_repeats=('coef',lambda x:int((x>0).sum())),
        bh_significant_repeats=('bh_qvalue',lambda x:int((x<.05).sum()))).reset_index()
    summary['rank_abs_mean']=summary.groupby('branch').coef_mean.transform(lambda x:x.abs().rank(ascending=False,method='first')).astype(int)
    summary=summary.sort_values(['branch','rank_abs_mean']);summary.to_csv(output/'attribute_summary.csv',index=False)
    plots(output,coefficients,summary)
    lines=['# CelebA feature RDR: joint attribute regression','',
        '**Complete: two comparisons, all five selected feature-model repetitions; 39,829 real test images, 1,985 identities, 40 attributes.**','',
        'The response is log(2 − RDR), regressed jointly on an intercept and all 40 binary CelebA annotations (absent=0, present=1). '
        'This follows the old DDIM attribute regression, now using the merged real held-out test pool and frozen JS feature models. '
        'The predictors are semantic annotations, not the 2048 Inception coordinates. Generated images are not assigned guessed annotations.','',
        '## Numerical handling and uncertainty','',
        'Some float32 sigmoid scores rounded to exactly 2. Recover alpha*z from the unchanged checkpoint and calculate '
        '`log(2) - logaddexp(0, alpha*z)` in float64. No epsilon clipping or image deletion is used. '
        'Recomputed RDR scores must agree with the archived predictions within rtol=2e-5, atol=3e-6; exact discrepancies are recorded.','',
        'OLS point estimates match the original joint-regression specification. Per-fit standard errors/95% intervals are clustered '
        'by real identity (small-sample correction, t reference with G−1 degrees of freedom). BH q-values adjust across 40 attributes '
        'separately within each fit. These are exploratory conditional-on-network intervals, not simultaneous inference over repetitions '
        'or uncertainty in neural training. Repetitions share data; coefficient SD is training variability, not a confidence interval.','',
        '## Interpretation','',
        'For r=2p/(p+q), 2−r=2q/(p+q). A positive coefficient therefore associates attribute presence with higher relative Q support; '
        'a negative coefficient associates it with lower relative Q support, holding the other annotations fixed. '
        'The coefficient is a fitted difference in the log response, not a causal effect or a marginal attribute prevalence ratio. '
        'Correlated attributes and the restricted real-image support limit attribution. This analysis does not identify which latent '
        'Inception coordinates cause the shift, nor characterize generated-only regions absent from real data.','',
        'The joint annotations explain only about 3% (Q_l) and 6% (Q_u) of the log-response variation here. These rankings therefore identify the strongest measured associations, not a comprehensive explanation of the shift. The log transformation is sensitive to extreme fitted logits, especially for Q_l.','',
        'Rank by absolute mean coefficient over all five repetitions, following the old coefficient-magnitude convention. '
        'Because all predictors are binary, these compare fitted absent-to-present contrasts. Full tables also report prevalence-scaled '
        'coefficients and drop-one partial R²; these can rank attributes differently.','',
        '![Top attribute coefficients](attribute_coefficients.png)','[Publication PDF](attribute_coefficients.pdf)','']
    for branch in ('lower','upper'):
        subset=metrics[metrics.branch==branch]
        lines += [f'## P versus Q_{"l" if branch=="lower" else "u"}','',
            f'R² mean (SD): {subset.r_squared.mean():.3f} ({subset.r_squared.std():.3f}).','',
            '| Attribute | Coefficient mean (SD) | Positive repeats / 5 | Mean partial R² |',
            '| --- | --- | --- | --- |']
        for row in summary[summary.branch==branch].head(10).itertuples():
            lines.append(f'| {row.attribute} | {row.coef_mean:.3f} ({row.coef_sd:.3f}) | {row.positive_repeats} | {row.partial_r2_mean:.4f} |')
        lines.append('')
    lines += ['[All 80 attribute summaries](attribute_summary.csv) · [400 per-fit coefficients and clustered intervals](coefficients_per_fit.csv) · [Regression diagnostics](regression_metrics.csv)','',
        'The archived design.csv joins every test image to its original identity and annotations. responses/ retains the stable response '
        'and scaled logits for each fit. Frozen source, input hashes and completion records support replay. No RDR network is retrained '
        'and no selected-model metrics or previous results are changed. Historical test reuse makes this a retrospective analysis.','']
    (output/'RESULTS.md').write_text('\n'.join(lines))
    shutil.copytree(run/'source',output/'source')
    for name in ('experiments/CelebA/attribute_regression.py','experiments/CelebA/final_figures.py'):
        target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    sources={str(p.relative_to(output/'source')):sha(p) for p in (output/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    protocol_out=dict(merged_parent=str(merged),merged_completion_sha256=sha(merged/'COMPLETE.json'),main_parent=str(run),
        input_hashes=inputs,source_files=sources,training_performed=False,network_inference=True,regression='joint OLS with intercept and 40 binary annotations',
        response='log(2)-softplus(alpha*z)',uncertainty='identity-clustered, G-1 t reference; BH per fit',rank_rule='absolute mean coefficient across five repetitions',
        attributes_path=str(args.attributes.resolve()),python=sys.version,numpy=np.__version__,torch=torch.__version__,statsmodels=sm.__version__)
    (output/'protocol.json').write_text(json.dumps(protocol_out,indent=2)+'\n')
    files={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and 'source' not in p.relative_to(output).parts}
    completion=dict(complete=True,regressions=10,coefficient_rows=400,real_test_images=len(manifest),identities=1985,figure_pairs=1,files=files,source_files=sources)
    (output/'COMPLETE.json').write_text(json.dumps(completion,indent=2)+'\n')
    (output/'COMPLETE.sha256').write_text(sha(output/'COMPLETE.json')+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--merged',type=Path,default=Path('/cwork/yx306/RDR/CelebA/merged_test_20261003'))
    parser.add_argument('--attributes',type=Path,default=Path('/hpc/group/mastatlab/yx306/CelebA/celeba/list_attr_celeba.txt'))
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
