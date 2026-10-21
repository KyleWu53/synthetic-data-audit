# LLM Synthetic Data: Generation vs. Statistical Fidelity

A reproducible benchmark comparing two ways to generate synthetic tabular data with Gemini: directly emit CSV rows, or write a Python sampler. The audit checks marginal distributions, categorical proportions, correlation structure, and whether a classifier can distinguish generated data from reference samples.

## Quick start (macOS/Linux, Python 3.10+)

Unzip this folder into your Home directory, open Terminal, then:

```bash
cd ~/synthetic-data-audit
bash run.sh --mode demo
```

Open `results/demo/REPORT.md`, `summary.csv`, and `comparison.png`. Included demo results use local fixtures, **not Gemini**. The demo CSV fixture deliberately shifts one mean and breaks the correlated pair. The demo sampler fixture is an independent reference sample. They test the harness; they do not predict LLM performance.

## Run the real experiment

Get an API key from https://aistudio.google.com/apikey and select an available text-generation model ID from https://ai.google.dev/gemini-api/docs/models. Set the key privately in Terminal (do not paste it into chat):

```bash
read -s -p 'Gemini API key: ' GEMINI_API_KEY; echo
export GEMINI_API_KEY
export GEMINI_MODEL='YOUR_AVAILABLE_MODEL_ID'
bash run.sh --mode gemini --n 500 --trials 3 --out results/gemini
```

On shells where `read -p` is unsupported, first run `bash`. This run makes 18 generation requests with the defaults: five CSV batches and one sampler per trial, across three trials. Retries can add requests. Provider charges/quota apply. The model is configurable because availability changes.

The first run audits CSV output and saves each `results/gemini/trial_*/sampler/sampler.py` with status `pending_code_review`. Inspect these scripts before running them. They should only sample arrays and print CSV. Generated Python is arbitrary code; execution below uses a timeout and strips common secret environment variables, but **is not a security sandbox**. Use a disposable environment for code you do not trust.

After reviewing all three scripts:

```bash
bash run.sh --mode gemini --n 500 --trials 3 --out results/gemini --execute-reviewed-samplers
unset GEMINI_API_KEY
```

Use exactly the same n, trial count, seed, and model on both passes. The second pass reuses CSVs and runs the reviewed scripts; it makes no Gemini requests. Use a new output directory for each fresh experiment; initial generation runs overwrite matching files. Raw API responses, prompts, generated scripts, CSVs, metrics, configuration, and aggregate reports are retained. Keys are never included in requests saved to disk.

## Experimental design

Both methods receive the same exact distribution specification, not the reference sample. This is a test of obeying a known data-generating process, not private-data learning. Reference data is simulated ground truth, not an observed real-world dataset.

- Normal: mean 0, SD 1.
- Lognormal: underlying normal mean 0, SD 0.75.
- Bimodal: 40% Normal(-2, 0.5 SD), 60% Normal(2, 0.8 SD).
- Correlated pair: standard-normal x,y with correlation 0.8.
- Category: A/B/C with probabilities 0.5/0.3/0.2.
- All groups independent except x,y.

Each trial uses 500 reference and 500 synthetic rows per method by default. An independent reference sample provides a finite-sample baseline. Three trials are a small exploratory experiment; use more trials for stronger conclusions. Seed labels in CSV prompts do not guarantee reproducible model output. Local generators are seeded; provider behavior and model versions may vary.

## Audit metrics

| Metric | Meaning |
|---|---|
| Two-sample KS statistic | Numeric marginal distribution discrepancy; smaller is better |
| Holm-adjusted KS p-value | Controls five numeric tests within a method/trial; does not establish equality |
| Category total variation | Half the sum of absolute category-probability differences |
| Correlation RMSE | RMSE across the ten unique off-diagonal Pearson correlations |
| Discriminator OOF ROC AUC | Random forest trained/evaluated with stratified five-fold CV; chance is 0.5 |
| Duplicate fraction | Fraction of repeated full rows |

The discriminator fits category encoding inside each fold to avoid preprocessing leakage. AUC is descriptive, with no claimed hypothesis-test p-value or confidence interval. Both low and high deviations from 0.5 merit inspection; no classifier can certify distributional identity. Across-trial mean and sample SD are reported. No single composite score is used because the metrics capture different failures.

CSV validation enforces exact row count/schema, numeric finite values, positive lognormal values, and allowed categories. Invalid output is reported as failure, not silently repaired. CSV batching reduces truncation but may affect diversity. Output-token limits, row rounding, model randomness, sample size, classifier capacity, and small trial counts limit interpretation. Failure rate should be reported alongside fidelity metrics. The study does not establish privacy guarantees or downstream task utility.

## Verify

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Tests verify reference correlation, input rejection, Holm adjustment, and sensitivity to a large shift/broken dependence. Dependencies are bounded rather than locked; after the real run, save the exact environment:

```bash
.venv/bin/python -m pip freeze > results/gemini/environment.txt
```

## Publish on GitHub

Review files before publishing. `.gitignore` excludes credentials, the virtual environment, and real-run output by default. To intentionally include reviewed experiment artifacts:

```bash
git init
git add audit.py run.sh requirements.txt README.md tests .gitignore results/demo
git commit -m "Build synthetic data generation and statistical audit benchmark"
# After completing and reviewing the real experiment:
git add -f results/gemini
git commit -m "Document Gemini experiment results"
```

Create an empty GitHub repository, then follow its instructions to add a remote and push. Do not commit API keys. No GitHub repository has been created by this package.

## Resume / application wording

After completing the real run, a defensible bullet is:

“Built a Gemini-powered synthetic tabular data benchmark comparing direct CSV generation with LLM-written Python samplers; evaluated distribution fidelity using KS tests, correlation error, categorical total variation, and cross-validated discriminator AUC.”

Add measured results only after reviewing the real report. Before that, describe it as an implemented benchmark with a validated demo, not a completed Gemini study. Be ready to explain the code and disclose AI assistance accurately if asked.

API reference: https://ai.google.dev/api/generate-content
