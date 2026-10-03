#!/usr/bin/env python3
"""Identity-held-out EBM attribute explanations and RDR-response sensitivity."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
ENV=Path('/cwork/yx306/RDR/CelebA/environments/interpret_core_0_7_8')
sys.path.insert(0,str(ENV))
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR','/tmp/celeba-interaction-mpl')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
import sklearn
import interpret
from interpret.glassbox import ExplainableBoostingRegressor
import joblib

SEED=2026100401
GROUPS={
 'facial_hair':['5_o_Clock_Shadow','Goatee','Mustache','No_Beard','Sideburns'],
 'hair':['Bald','Bangs','Black_Hair','Blond_Hair','Brown_Hair','Gray_Hair','Receding_Hairline','Straight_Hair','Wavy_Hair'],
 'cosmetics':['Heavy_Makeup','Wearing_Lipstick','Rosy_Cheeks'],
 'accessories':['Eyeglasses','Wearing_Hat','Wearing_Earrings','Wearing_Necklace','Wearing_Necktie'],
 'expression':['Smiling','Mouth_Slightly_Open'],
 'facial_structure':['Big_Lips','Big_Nose','Pointy_Nose','Narrow_Eyes','Arched_Eyebrows','Bushy_Eyebrows','High_Cheekbones','Oval_Face','Double_Chin','Chubby'],
 'age_sex_annotations':['Male','Young'],
}
PARAMS=dict(interactions=10,outer_bags=1,max_bins=4,max_interaction_bins=4,n_jobs=1,
 random_state=SEED,max_rounds=2000,early_stopping_rounds=50,early_stopping_tolerance=1e-4,
 min_samples_leaf=30,smoothing_rounds=0,interaction_smoothing_rounds=0)


def require(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def seal(path,value):
    write(path,value);Path(path).with_suffix('.sha256').write_text(sha(path)+'\n')


def sealed(path):
    require(sha(path)==Path(path).with_suffix('.sha256').read_text().strip(),'Changed sealed record')
    return read(path)


def identity_split(identities,seed=SEED):
    ids=np.asarray(identities)
    # Preserve first-appearance ordering before a seeded permutation.
    unique=pd.unique(ids);order=np.random.default_rng(seed).permutation(unique)
    count=len(unique)//5
    role=np.full(len(ids),'train',dtype='U5')
    role[np.isin(ids,order[:count])]='test'
    role[np.isin(ids,order[count:2*count])]='val'
    return role


def fit_surrogate(x,y,roles,names,interactions=10,params=None):
    """No outer-test rows passed to fit; explicit identity-separated early stopping."""
    train=roles=='train';dev=roles!='test'
    mean=float(y[train].mean());scale=float(y[train].std())
    require(scale>0 and np.isfinite(y).all(),'Invalid response')
    settings=dict(PARAMS if params is None else params);settings['interactions']=interactions
    model=ExplainableBoostingRegressor(feature_names=list(names),feature_types=['nominal']*len(names),**settings)
    bags=np.where(roles[dev]=='train',1,-1).astype(np.int8)[:,None]
    model.fit(x[dev],(y[dev]-mean)/scale,bags=bags)
    return model,mean,scale


def score(y,prediction):
    require(len(y)>0 and np.isfinite(prediction).all(),'Invalid assessment predictions')
    mse=float(np.mean((y-prediction)**2));variance=float(np.var(y))
    require(variance>0,'Constant assessment response')
    return dict(mse=mse,r_squared=1-mse/variance,prediction_min=float(np.min(prediction)),prediction_max=float(np.max(prediction)))


def prepare(output,parent):
    require(not output.exists(),'Use a new output directory')
    completion=sealed(parent/'COMPLETE.json')
    require(completion['regressions']==10 and completion['real_test_images']==39829,'Wrong parent analysis')
    for name,digest in completion['files'].items():require(sha(parent/name)==digest,'Changed parent artifact')
    d=pd.read_csv(parent/'design.csv');names=list(d.columns[3:])
    require(len(names)==40 and d.source_id.is_unique,'Invalid attribute design')
    require(np.isin(d[names],(0,1)).all(),'Nonbinary annotation')
    roles=identity_split(d.identity)
    output.mkdir(parents=True);(output/'logs').mkdir();(output/'source/experiments/CelebA').mkdir(parents=True)
    split=d[['source_id','identity','original_final_role']].copy();split['surrogate_role']=roles
    split.to_csv(output/'split.csv',index=False)
    source_files={}
    for name in ('attribute_interactions.py','attribute_interactions.slurm'):
        target=output/'source/experiments/CelebA'/name;shutil.copyfile(Path(__file__).with_name(name),target)
        source_files[str(target.relative_to(output/'source'))]=sha(target)
    tasks=[dict(branch=b,response=response,repeat=r) for b in ('lower','upper') for response in ('rdr','log_complement') for r in range(5)]
    counts={role:dict(images=int(np.sum(roles==role)),identities=int(d.loc[roles==role,'identity'].nunique())) for role in ('train','val','test')}
    ablations=[dict(kind='attribute',name=name,members=[name]) for name in names]
    ablations += [dict(kind='group',name=name,members=members) for name,members in GROUPS.items()]
    env_files={str(p.relative_to(ENV)):sha(p) for p in ENV.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc',)}
    protocol=dict(parent=str(parent),parent_completion_sha256=sha(parent/'COMPLETE.json'),
      input_hashes={name:completion['files'][name] for name in ['design.csv']+[f'responses/{b}_{r:02d}.npz' for b in ('lower','upper') for r in range(5)]},
      split_sha256=sha(output/'split.csv'),source_files=source_files,seed=SEED,counts=counts,attributes=names,ablations=ablations,tasks=tasks,
      ebm_parameters=PARAMS,environment=str(ENV),environment_files=env_files,
      versions=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,interpret=interpret.__version__),
      scope='Exploratory post hoc surrogate models; RDR networks frozen; no new confirmatory data',
      training_scope='Fit attribute explanation models only; no RDR training',
      response_rule='rdr uses archived bounded scores; log_complement uses previously recovered stable log responses',
      ranking_rule='Held-out delta R2 = R2(full EBM) - R2(refitted reduced EBM); negatives retained',
      evaluation_rule='One fixed identity-disjoint 60/20/20 train/early-stop/test split; common to all tasks and ablations',
      tuning_rule='Fixed settings; automatic interaction selection and early stopping use development rows only; no test-based tuning',
      benchmark_note='A single lower/log repeat-00 runtime benchmark used these same prechosen settings and split, then printed its assessment R2; no settings were adapted to that value',
      expected=dict(tasks=20,ebm_fits=980,ols_fits=20,ablations=940))
    seal(output/'protocol.json',protocol)
    print(json.dumps(dict(prepared=str(output),counts=counts,expected=protocol['expected'])),flush=True)


def load(output):
    protocol=sealed(output/'protocol.json')
    require(sha(output/'split.csv')==protocol['split_sha256'],'Changed split')
    require(sha(Path(protocol['parent'])/'COMPLETE.json')==protocol['parent_completion_sha256'],'Changed parent')
    require(interpret.__version__==protocol['versions']['interpret'],'Changed InterpretML version')
    for name,digest in protocol['source_files'].items():
        require(sha(output/'source'/name)==digest,'Changed frozen source')
    require(sha(Path(__file__))==protocol['source_files']['experiments/CelebA/attribute_interactions.py'],'Run frozen source')
    return protocol


def fit_task(output,index):
    protocol=load(output);task=protocol['tasks'][index];directory=output/'tasks'/f'{index:02d}'
    if (directory/'COMPLETE.json').exists():
        receipt=sealed(directory/'COMPLETE.json')
        require(receipt['protocol_sha256']==sha(output/'protocol.json'),'Task protocol changed')
        for name,digest in receipt['files'].items():require(sha(directory/name)==digest,'Changed task artifact')
        return
    require(not directory.exists(),'Incomplete task directory; inspect before retry')
    for name,digest in protocol['environment_files'].items():require(sha(ENV/name)==digest,'Changed installed environment')
    parent=Path(protocol['parent']);d=pd.read_csv(parent/'design.csv')
    require(sha(parent/'design.csv')==protocol['input_hashes']['design.csv'],'Changed design')
    source=f"responses/{task['branch']}_{task['repeat']:02d}.npz"
    require(sha(parent/source)==protocol['input_hashes'][source],'Changed response')
    with np.load(parent/source) as data:y=data['saved_rdr' if task['response']=='rdr' else 'log_2_minus_rdr']
    split=pd.read_csv(output/'split.csv');require(np.array_equal(split.source_id,d.source_id),'Row ordering mismatch')
    roles=split.surrogate_role.to_numpy();names=protocol['attributes'];x=d[names].to_numpy(dtype=float)
    test=roles=='test';train=roles=='train';yt=y[test]
    directory.mkdir(parents=True);start=time.time();metric_rows=[];prediction_arrays={'y':yt,'test_rows':np.flatnonzero(test)}
    ols=LinearRegression().fit(x[train],y[train]);pred=ols.predict(x[test]);prediction_arrays['ols']=pred
    metric_rows.append({**task,'model':'ols',**score(yt,pred)})
    prediction_arrays['mean_only']=np.full(len(yt),y[train].mean())
    metric_rows.append({**task,'model':'mean_only',**score(yt,prediction_arrays['mean_only'])})
    full_model=None;full_score=None
    for kind,count in [('additive_ebm',0),('interaction_ebm',10)]:
        model,mean,scale=fit_surrogate(x,y,roles,names,count,protocol['ebm_parameters'])
        pred=model.predict(x[test])*scale+mean;prediction_arrays[kind]=pred;metrics=score(yt,pred)
        metric_rows.append({**task,'model':kind,**metrics,'best_iteration':json.dumps(model.best_iteration_.tolist())})
        joblib.dump(dict(model=model,mean=mean,scale=scale),directory/f'{kind}.joblib')
        if count:full_model=model;full_score=metrics;full_scale=scale
    # Learned term magnitudes summarize the full surrogate on training rows;
    # they are not held-out deletion importance or causal contributions.
    terms=full_model.eval_terms(x[train])*full_scale
    term_rows=[{**task,'term':name,'order':len(indices),'mean_abs_centered_train':float(np.mean(np.abs(terms[:,j]-terms[:,j].mean()))),
                'members':json.dumps([names[i] for i in indices])} for j,(name,indices) in enumerate(zip(full_model.term_names_,full_model.term_features_))]
    pd.DataFrame(term_rows).to_csv(directory/'terms.csv',index=False)
    rows=[]
    for j,ablation in enumerate(protocol['ablations']):
        keep=[i for i,name in enumerate(names) if name not in ablation['members']]
        model,mean,scale=fit_surrogate(x[:,keep],y,roles,[names[i] for i in keep],10,protocol['ebm_parameters'])
        pred=model.predict(x[test][:,keep])*scale+mean;prediction_arrays[f'ablation_{j:02d}']=pred
        reduced=score(yt,pred)
        rows.append({**task,**ablation,'members':json.dumps(ablation['members']),**reduced,
           'delta_mse':reduced['mse']-full_score['mse'],'delta_r2':full_score['r_squared']-reduced['r_squared'],
           'best_iteration':json.dumps(model.best_iteration_.tolist())})
        if (j+1)%10==0:print(f'task {index}: {j+1}/{len(protocol["ablations"])} deletions',flush=True)
    pd.DataFrame(rows).to_csv(directory/'importance.csv',index=False)
    pd.DataFrame(metric_rows).to_csv(directory/'metrics.csv',index=False)
    np.savez_compressed(directory/'assessment_predictions.npz',**prediction_arrays)
    write(directory/'task.json',dict(**task,seconds=time.time()-start,n_test=len(yt)))
    files={p.name:sha(p) for p in directory.iterdir() if p.is_file()}
    seal(directory/'COMPLETE.json',dict(complete=True,task=task,protocol_sha256=sha(output/'protocol.json'),ebm_fits=49,ols_fits=1,ablations=47,files=files))
    print(json.dumps(dict(task=index,complete=True,seconds=time.time()-start)),flush=True)


def plots(output,metrics,importance):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    colors={'ols':'#777777','additive_ebm':'#df8c22','interaction_ebm':'#2166ac'}
    fig,axes=plt.subplots(1,2,figsize=(8,3.8),layout='constrained')
    for ax,branch in zip(axes,('lower','upper')):
        for j,response in enumerate(('rdr','log_complement')):
            for i,model in enumerate(colors):
                vals=metrics[(metrics.branch==branch)&(metrics.response==response)&(metrics.model==model)].r_squared
                ax.scatter(j+(i-1)*.19+np.linspace(-.035,.035,5),vals,s=20,color=colors[model],label=model if j==0 else None)
                ax.scatter(j+(i-1)*.19,vals.mean(),s=50,color=colors[model],marker='_')
        ax.axhline(0,color='.5',lw=.7);ax.set_xticks([0,1],['RDR','log(2−RDR)']);ax.set_title(r'$P$ vs $Q_l$' if branch=='lower' else r'$P$ vs $Q_u$')
        ax.set_ylabel('Held-out R²');ax.grid(axis='y',alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,['OLS','Additive EBM','Interaction EBM'],loc='outside lower center',ncol=3)
    for ext in ('png','pdf'):fig.savefig(output/f'predictive_accuracy.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    for kind,limit in [('attribute',12),('group',7)]:
        fig,axes=plt.subplots(2,2,figsize=(11,9 if kind=='attribute' else 6.5),layout='constrained')
        for row,branch in enumerate(('lower','upper')):
            for col,response in enumerate(('rdr','log_complement')):
                ax=axes[row,col];s=importance[(importance.branch==branch)&(importance.response==response)&(importance.kind==kind)]
                selected=s.groupby('name').delta_r2.mean().nlargest(limit)
                for rank,name in enumerate(selected.index):
                    vals=s[s.name==name].sort_values('repeat').delta_r2.to_numpy()
                    ax.scatter(vals,np.full(5,rank),s=16,alpha=.55,color='#2166ac')
                    ax.scatter(vals.mean(),rank,color='black',s=24,marker='d')
                ax.set_yticks(range(len(selected)),selected.index.str.replace('_',' '),fontsize=9)
                ax.invert_yaxis();ax.axvline(0,color='.5',lw=.8);ax.grid(axis='x',alpha=.2)
                ax.set_title(f'Q_{"l" if branch=="lower" else "u"}: {"RDR" if response=="rdr" else "log(2−RDR)"}',fontsize=12)
                ax.set_xlabel('Held-out ΔR² after removal and refitting',fontsize=10)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,3))
                ax.tick_params(axis='x',labelsize=9)
        fig.suptitle('Individual fits (blue); five-fit means (black). Negative values retained.',fontsize=12)
        for ext in ('png','pdf'):fig.savefig(output/f'{kind}_importance.{ext}',dpi=220,bbox_inches='tight')
        plt.close(fig)


def report(output):
    protocol=load(output);all_metrics=[];all_importance=[];all_terms=[];receipts={}
    for index,task in enumerate(protocol['tasks']):
        directory=output/'tasks'/f'{index:02d}';receipt=sealed(directory/'COMPLETE.json')
        require(receipt['task']==task and receipt['protocol_sha256']==sha(output/'protocol.json'),'Wrong task receipt')
        for name,digest in receipt['files'].items():require(sha(directory/name)==digest,'Changed task output')
        receipts[str(directory.relative_to(output))]=sha(directory/'COMPLETE.json')
        all_metrics.append(pd.read_csv(directory/'metrics.csv'));all_importance.append(pd.read_csv(directory/'importance.csv'));all_terms.append(pd.read_csv(directory/'terms.csv'))
    metrics=pd.concat(all_metrics,ignore_index=True);importance=pd.concat(all_importance,ignore_index=True);terms=pd.concat(all_terms,ignore_index=True)
    require(len(metrics)==80 and len(importance)==940,'Incomplete fit tables')
    for name,table in [('metrics_per_fit',metrics),('importance_per_fit',importance),('terms_per_fit',terms)]:table.to_csv(output/f'{name}.csv',index=False)
    ms=metrics.groupby(['branch','response','model']).r_squared.agg(['mean','std','min','max']).reset_index();ms.to_csv(output/'metrics_summary.csv',index=False)
    summary=importance.groupby(['branch','response','kind','name']).delta_r2.agg(['mean','std','min','max',lambda x:int((x>0).sum())]).reset_index().rename(columns={'<lambda_0>':'positive_repeats'})
    summary['rank']=summary.groupby(['branch','response','kind'])['mean'].rank(ascending=False,method='first').astype(int)
    summary=summary.sort_values(['branch','response','kind','rank']);summary.to_csv(output/'importance_summary.csv',index=False)
    stability=[]
    for branch in ('lower','upper'):
        for kind in ('attribute','group'):
            a=summary[(summary.branch==branch)&(summary.response=='rdr')&(summary.kind==kind)].set_index('name')
            b=summary[(summary.branch==branch)&(summary.response=='log_complement')&(summary.kind==kind)].set_index('name').loc[a.index]
            top_a=set(a.nsmallest(5,'rank').index);top_b=set(b.nsmallest(5,'rank').index)
            stability.append(dict(branch=branch,kind=kind,spearman=float(spearmanr(a['mean'],b['mean']).statistic),top5_overlap=len(top_a&top_b),top5_common=json.dumps(sorted(top_a&top_b))))
    pd.DataFrame(stability).to_csv(output/'response_stability.csv',index=False)
    plots(output,metrics,importance)
    lines=['# CelebA attribute interactions and response sensitivity','',
      '**Complete: 20 scenarios (two pairs × two responses × five frozen RDR fits), 980 EBM fits and 20 OLS fits.**','',
      'These are post hoc attribute explanation models. No RDR network is trained or changed. '
      'The same 39,829 real images with 40 binary attributes are divided by identity into training, early-stopping and assessment sets. '
      'One common fixed split is used across all scenarios; this is a held-out comparison, not cross-validation. '
      'The underlying pool was historically inspected, so this remains exploratory rather than fresh confirmatory evidence.','',
      '| Surrogate role | Images | Identities |','| --- | --- | --- |']
    for role,c in protocol['counts'].items():lines.append(f'| {role} | {c["images"]} | {c["identities"]} |')
    lines+=['','## Methods','',
      'Compare train-only OLS, an additive EBM, and an EBM with 10 automatically selected pairwise interactions. '
      'EBM early stopping uses only the reserved validation identities through explicit bag assignments; assessment rows never enter fit. '
      'Each target is centered/scaled using training rows only, then predictions are returned to the original scale. '
      'RDR uses the saved bounded scores; log(2−RDR) uses the previously recovered stable logits, retaining all saturated observations. '
      'EBM predictions are not clipped. The fixed EBM specification is in protocol.json, using InterpretML 0.7.8.','',
      'Remove each of 40 attributes and each of seven predefined semantic groups, then refit the interaction EBM with the same split, '
      'settings and early-stopping rule. Interaction selection is repeated using the remaining predictors. '
      'Importance is held-out ΔR² = R²(full) − R²(reduced), equivalently the increase in MSE divided by assessment response variance. '
      'Positive values mean removal worsens prediction; negative values mean the reduced model predicts better. '
      'These are algorithm-dependent incremental predictive contributions, not causal effects or additive shares of total divergence. '
      'Grouped deletion captures shared information, but does not remove all dependence between retained and omitted annotations.','',
      'The seven groups are prespecified in protocol.json and are not a partition of all 40 attributes. '
      'The 10 fitted interaction terms are descriptive model components; their training magnitudes in terms_per_fit.csv are not held-out deletion importance. '
      'Five-fit SD and plotted dots reflect variation across frozen RDR targets on the same data. They are not confidence intervals. '
      'Rankings are assessment diagnostics, not a new selected model. No assessment-based tuning or best-repeat selection occurs.','',
      '## Held-out accuracy','',
      '| Pair | Response | OLS R² mean (SD) | Additive EBM | Interaction EBM |','| --- | --- | --- | --- | --- |']
    for branch in ('lower','upper'):
        for response in ('rdr','log_complement'):
            cells=[]
            for model in ('ols','additive_ebm','interaction_ebm'):
                r=ms[(ms.branch==branch)&(ms.response==response)&(ms.model==model)].iloc[0];cells.append(f'{r["mean"]:.4f} ({r["std"]:.4f})')
            lines.append(f'| Q_{"l" if branch=="lower" else "u"} | {response} | '+ ' | '.join(cells)+' |')
    lines+=['','![Held-out accuracy](predictive_accuracy.png)','[PDF](predictive_accuracy.pdf)','',
      'R² uses the variance of the same assessment response. Negative R² means worse than its assessment-mean reference; '
      'the independently fitted training-mean baseline is also included in metrics_per_fit.csv. '
      'Compare R² within each response and use rankings to assess response sensitivity; raw MSE values across responses are not comparable.','',
      '## Attribute and group importance','',
      '![Attribute importance](attribute_importance.png)','[PDF](attribute_importance.pdf)','',
      '![Group importance](group_importance.png)','[PDF](group_importance.pdf)','']
    for branch in ('lower','upper'):
        for response in ('rdr','log_complement'):
            lines += [f'### {branch} / {response}','', '| Attribute | Mean ΔR² | SD | Positive repeats / 5 |','| --- | --- | --- | --- |']
            subset=summary[(summary.branch==branch)&(summary.response==response)&(summary.kind=='attribute')].head(10)
            for r in subset.itertuples():lines.append(f'| {r.name} | {r.mean:.5f} | {r.std:.5f} | {r.positive_repeats} |')
            lines.append('')
    lines+=['## Response-ranking agreement','', '| Pair | Type | Spearman correlation | Top-five overlap |','| --- | --- | --- | --- |']
    for r in stability:lines.append(f'| {r["branch"]} | {r["kind"]} | {r["spearman"]:.3f} | {r["top5_overlap"]}/5 |')
    lines+=['','[Accuracy table](metrics_summary.csv) · [All importance rankings](importance_summary.csv) · [Per-fit importances](importance_per_fit.csv) · [Response stability](response_stability.csv) · [Interaction components](terms_per_fit.csv)','',
      'All per-task held-out predictions, the two EBM models, split membership, parent hashes, installed-environment hashes and frozen source '
      'remain in the HPC archive. The report, aggregate/per-fit tables, plots and reproducible source are suitable for compact repository export. '
      'Predictors are CelebA annotations on real images only; generated-only regions and unannotated visual properties are not explained. '
      'Attribute labels are measured annotations, not inferred causal mechanisms.','',
      'Implementation reference: [InterpretML EBM API](https://interpret.ml/docs/python/api/ExplainableBoostingRegressor.html).','']
    (output/'RESULTS.md').write_text('\n'.join(lines))
    files={p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name not in ('COMPLETE.json','COMPLETE.sha256','jobs.json')}
    seal(output/'COMPLETE.json',dict(complete=True,**protocol['expected'],figure_pairs=3,protocol_sha256=sha(output/'protocol.json'),files=files,task_receipts=receipts,source_files=protocol['source_files']))
    print(json.dumps(dict(complete=True,**protocol['expected'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','fit','report'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--parent',type=Path,default=Path('/cwork/yx306/RDR/CelebA/attribute_regression_expanded_20261003_v2'))
    parser.add_argument('--task-index',type=int)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.output.resolve(),args.parent.resolve())
    elif args.action=='fit':fit_task(args.output.resolve(),args.task_index)
    else:report(args.output.resolve())
