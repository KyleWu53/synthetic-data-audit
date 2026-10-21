"""Distribution fidelity benchmark. Demo results are never LLM results."""
import argparse
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder

NUMERIC = ['normal', 'lognormal', 'bimodal', 'x', 'y']
COLUMNS = NUMERIC + ['category']
SPEC = '''Columns in order: normal,lognormal,bimodal,x,y,category.
normal ~ Normal(mean=0, standard deviation=1).
lognormal = exp(Z), Z ~ Normal(0,0.75).
bimodal: with probability 0.4 Normal(-2,0.5), otherwise Normal(2,0.8).
x,y jointly standard normal, correlation 0.8.
category takes A,B,C with probabilities 0.5,0.3,0.2.
All groups are mutually independent except x,y. Rows are independent.'''


def reference(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame(dict(normal=rng.normal(size=n),
        lognormal=rng.lognormal(0, .75, n),
        bimodal=np.where(rng.random(n) < .4, rng.normal(-2,.5,n), rng.normal(2,.8,n)),
        x=x, y=.8*x+.6*rng.normal(size=n),
        category=rng.choice(['A','B','C'], n, p=[.5,.3,.2])))


def validate(df, n):
    if list(df.columns) != COLUMNS or len(df) != n:
        raise ValueError(f'Expected {n} rows and columns {COLUMNS}; got {df.shape}, {list(df.columns)}')
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors='raise')
    if not np.isfinite(df[NUMERIC].to_numpy()).all() or df.isna().any().any():
        raise ValueError('Missing or nonfinite values')
    if not df.category.isin(['A','B','C']).all():
        raise ValueError('Unknown category')
    if (df.lognormal <= 0).any():
        raise ValueError('Lognormal values must be positive')
    return df


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(order))
    highest = 0
    for rank, index in enumerate(order):
        highest = max(highest, (len(order)-rank)*pvalues[index])
        adjusted[index] = min(1, highest)
    return adjusted.tolist()


def score(real, synth, seed):
    ks = {c: dict(zip(['statistic','pvalue'], map(float, ks_2samp(real[c], synth[c])))) for c in NUMERIC}
    for col, p in zip(NUMERIC, holm([ks[c]['pvalue'] for c in NUMERIC])):
        ks[col]['holm_pvalue'] = p
    rc, sc = real[NUMERIC].corr(), synth[NUMERIC].corr()
    delta = (rc-sc).to_numpy()
    cats = ['A','B','C']
    rp = real.category.value_counts(normalize=True).reindex(cats, fill_value=0)
    sp = synth.category.value_counts(normalize=True).reindex(cats, fill_value=0)
    # All preprocessing occurs inside each CV fold. Balanced real/synthetic counts.
    from sklearn.pipeline import make_pipeline
    transform = ColumnTransformer([('numeric','passthrough',NUMERIC),
        ('category',OneHotEncoder(handle_unknown='ignore'),['category'])])
    model = make_pipeline(transform, RandomForestClassifier(n_estimators=150,
        min_samples_leaf=5, max_depth=8, random_state=seed, n_jobs=1))
    data = pd.concat([real,synth], ignore_index=True)
    labels = np.r_[np.zeros(len(real)),np.ones(len(synth))]
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    pred = cross_val_predict(model,data,labels,cv=cv,method='predict_proba')[:,1]
    auc = float(roc_auc_score(labels,pred))
    return dict(ks=ks, category_total_variation=float(abs(rp-sp).sum()/2),
        correlation_rmse=float(np.sqrt(np.mean(delta[np.triu_indices(5,1)]**2))),
        real_xy_correlation=float(rc.loc['x','y']), synthetic_xy_correlation=float(sc.loc['x','y']),
        discriminator_oof_auc=auc, discriminator_distance_from_chance=abs(auc-.5),
        duplicate_fraction=float(synth.duplicated().mean()))


def gemini(prompt, model):
    key = os.environ.get('GEMINI_API_KEY')
    if not key:
        raise ValueError('Set GEMINI_API_KEY locally before running Gemini mode.')
    if not re.fullmatch(r'[A-Za-z0-9._-]+', model):
        raise ValueError('Invalid model ID')
    payload = dict(contents=[dict(parts=[dict(text=prompt)])],
                   generationConfig=dict(temperature=.7, maxOutputTokens=32768))
    request = urllib.request.Request(
        f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
        data=json.dumps(payload).encode(), headers={'Content-Type':'application/json','x-goog-api-key':key})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.load(response)
            candidate = body.get('candidates', [{}])[0]
            if candidate.get('finishReason') != 'STOP':
                raise ValueError(f'Incomplete model output: {candidate.get("finishReason")}')
            parts = candidate.get('content',{}).get('parts',[])
            return '\n'.join(p['text'] for p in parts if 'text' in p and not p.get('thought')), body
        except urllib.error.HTTPError as exc:
            if exc.code not in (429,500,502,503,504) or attempt == 3:
                raise RuntimeError(f'Gemini request failed with HTTP {exc.code}; check key, model and quota.') from None
            time.sleep(2**attempt)


def unfence(text):
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```[^\n]*\n', '', text)
        text = re.sub(r'\n```\s*$', '', text)
    return text


def generate(method, n, seed, model, folder, batch_size):
    if method == 'csv':
        frames = []
        for batch, start in enumerate(range(0,n,batch_size)):
            count = min(batch_size,n-start)
            prompt = SPEC + f'\nEmit exactly {count} random rows as CSV with header. Output only CSV. Batch {batch}; experiment seed label {seed}. Do not write code.'
            (folder/f'prompt_{batch}.txt').write_text(prompt)
            raw, response = gemini(prompt,model)
            (folder/f'response_{batch}.json').write_text(json.dumps(response,indent=2))
            frames.append(validate(pd.read_csv(io.StringIO(unfence(raw))), count))
        return validate(pd.concat(frames,ignore_index=True),n)
    prompt = SPEC + '\nWrite only Python code using numpy and pandas. Read n and seed from sys.argv[1:3]. Write CSV including the header to stdout. Use numpy.random.default_rng(seed). No files, network, or external packages beyond numpy and pandas.'
    (folder/'prompt.txt').write_text(prompt)
    raw,response = gemini(prompt,model)
    (folder/'response.json').write_text(json.dumps(response,indent=2))
    (folder/'sampler.py').write_text(unfence(raw))
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['demo','gemini'],default='demo')
    p.add_argument('--model',default=os.environ.get('GEMINI_MODEL'))
    p.add_argument('--n',type=int,default=500)
    p.add_argument('--trials',type=int,default=3)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--batch-size',type=int,default=100)
    p.add_argument('--out',type=Path,default=Path('results/demo'))
    p.add_argument('--execute-reviewed-samplers',action='store_true',help='Run previously generated Python after YOU review it; this is not a sandbox.')
    args = p.parse_args()
    if min(args.n,args.batch_size,args.trials) < 1 or args.n < 50:
        p.error('Use n >= 50 and positive batch-size/trials')
    if args.mode == 'gemini' and not args.model:
        p.error('Provide --model MODEL_ID or set GEMINI_MODEL')
    args.out.mkdir(parents=True,exist_ok=True)
    manifest = vars(args).copy(); manifest['out']=str(args.out)
    (args.out/'config.json').write_text(json.dumps(manifest,indent=2))
    rows=[]
    for trial in range(args.trials):
        seed = args.seed+trial
        real=reference(args.n,seed)
        trial_dir=args.out/f'trial_{trial}'; trial_dir.mkdir(exist_ok=True)
        real.to_csv(trial_dir/'reference.csv',index=False)
        for method in ['baseline','csv','sampler']:
            folder=trial_dir/method; folder.mkdir(exist_ok=True)
            try:
                if method == 'baseline':
                    synth=reference(args.n,seed+10000)
                elif args.mode == 'demo':
                    # Deliberately flawed fixture demonstrates sensitivity, NOT an LLM simulation.
                    synth=reference(args.n,seed+20000)
                    if method == 'csv':
                        synth['y']=np.random.default_rng(seed+30000).normal(size=args.n)
                        synth['normal']+=.4
                elif method == 'sampler' and args.execute_reviewed_samplers:
                    script=folder/'sampler.py'
                    if not script.exists():
                        raise ValueError('Generate and review sampler.py first')
                    # Do not pass API keys to generated programs. This is NOT isolation.
                    env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ['KEY','TOKEN','SECRET','PASSWORD'])}
                    result=subprocess.run([sys.executable,str(script.resolve()),str(args.n),str(seed+20000)],
                        capture_output=True,text=True,timeout=30,env=env,cwd=folder)
                    if result.returncode:
                        raise ValueError('Sampler failed; inspect locally. '+result.stderr[-500:])
                    synth=validate(pd.read_csv(io.StringIO(result.stdout)),args.n)
                elif method == 'csv' and args.execute_reviewed_samplers:
                    synth=validate(pd.read_csv(folder/'synthetic.csv'),args.n)
                else:
                    synth=generate(method,args.n,seed,args.model,folder,args.batch_size)
                    if synth is None:
                        rows.append(dict(trial=trial,method=method,status='pending_code_review'))
                        continue
                validate(synth,args.n).to_csv(folder/'synthetic.csv',index=False)
                metrics=score(real,synth,seed)
                (folder/'audit.json').write_text(json.dumps(metrics,indent=2))
                rows.append(dict(trial=trial,method=method,status='ok',
                    mean_ks=np.mean([v['statistic'] for v in metrics['ks'].values()]),
                    **{k:v for k,v in metrics.items() if k != 'ks'}))
            except Exception as exc:
                rows.append(dict(trial=trial,method=method,status='failed',error=str(exc)))
        print(f'Completed trial {trial+1}/{args.trials}',flush=True)
    table=pd.DataFrame(rows); table.to_csv(args.out/'summary.csv',index=False)
    title='DEMO FIXTURES — NOT GEMINI RESULTS' if args.mode == 'demo' else f'Gemini: {args.model}'
    good=table[table.status=='ok']
    metrics=['mean_ks','category_total_variation','correlation_rmse','discriminator_distance_from_chance']
    report=[f'# {title}', '', 'Lower is better for the metrics below. Baseline is an independent sample from the known generator.', '', '```',table.to_string(index=False),'```','', '## Across-trial mean ± sample standard deviation','']
    if len(good):
        report += ['```',good.groupby('method')[metrics].agg(['mean','std']).to_string(),'```']
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(12,3.5))
        for ax,metric in zip(axes,metrics[:3]):
            good.boxplot(column=metric,by='method',ax=ax,grid=False)
            ax.set_title(metric.replace('_',' ')); ax.set_xlabel('')
        fig.suptitle(title); fig.tight_layout(); fig.savefig(args.out/'comparison.png',dpi=160); plt.close(fig)
        report+=['','![Comparison](comparison.png)']
    report+=['','KS p-values use Holm correction within each method/trial, not across the full experiment. A nonsignificant test is not proof of equality. Discriminator AUC uses pooled five-fold out-of-fold predictions; 0.5 is chance. This classifier cannot certify indistinguishability. Trial summaries are descriptive, not confidence intervals.']
    (args.out/'REPORT.md').write_text('\n'.join(report))
    print(f'Report: {args.out}/REPORT.md')
    if any(r['status']=='failed' for r in rows):
        sys.exit(1)

if __name__ == '__main__':
    main()
