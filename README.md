# Synthetic Data Generation and Statistical Audit

I built this project to compare two ways of generating synthetic data with Gemini: asking it to write CSV rows directly and asking it to write a Python script that generates the data.

The goal was to check whether the output actually follows the requested distributions. I used statistical tests, correlation comparisons, and a classifier that tries to tell reference data from synthetic data.

## Results

I ran three trials using `gemini-2.5-flash`, requesting 500 rows per method in each trial.

All three Python samplers produced valid datasets. Direct CSV generation failed in all three trials:

- One trial hit the output-token limit.
- Two trials returned 113 and 114 rows when asked for a batch of 100.
- Invalid outputs were rejected instead of trimmed or repaired.

These are the average results across the three trials. The baseline is an independent sample from the same reference generator.

| Metric | Reference baseline | Python sampler |
|---|---:|---:|
| Mean KS statistic | 0.0501 | 0.0500 |
| Category total variation | 0.0280 | 0.0600 |
| Correlation RMSE | 0.0636 | 0.0596 |
| Discriminator ROC AUC | 0.4883 | 0.5128 |

The samplers produced correlations between 0.789 and 0.802 for a target of 0.8. Their discriminator AUCs ranged from 0.504 to 0.524, close to random guessing at 0.5.

Writing a Python sampler was more reliable in this experiment. However, three trials are a small sample, and the results do not prove that the generated data perfectly matches the reference distribution. The sampler's category proportions also differed more from the reference than the baseline did.

There are no fidelity scores for direct CSV generation because none of its complete datasets passed validation.

## Data

Each dataset contains six columns:

| Column | Distribution |
|---|---|
| `normal` | Normal with mean 0 and standard deviation 1 |
| `lognormal` | Exponential of a normal variable with mean 0 and standard deviation 0.75 |
| `bimodal` | 40% Normal(-2, 0.5 SD), 60% Normal(2, 0.8 SD) |
| `x`, `y` | Standard normal variables with correlation 0.8 |
| `category` | A, B, or C with probabilities 0.5, 0.3, and 0.2 |

The columns are independent except for `x` and `y`.

Both methods receive the same distribution description. Neither receives the reference dataset. The reference data is generated in Python from the known distributions.

## Statistical checks

| Check | What it measures |
|---|---|
| Two-sample KS test | Differences between numeric distributions |
| Holm correction | Adjusts the five numeric KS p-values within each method and trial |
| Category total variation | Differences in category proportions |
| Correlation RMSE | Differences across the ten unique numeric correlation pairs |
| Discriminator ROC AUC | How well a random forest separates reference and synthetic rows |
| Duplicate fraction | Fraction of repeated complete rows |

Lower KS statistics, total variation, and correlation errors indicate a closer match. A discriminator AUC near 0.5 means the classifier has difficulty separating the datasets.

The classifier uses five-fold stratified cross-validation, with category encoding fitted inside each training fold.

Before scoring, the program checks:

- Exact row count and column names.
- Missing or nonfinite values.
- Numeric column types.
- Positive lognormal values.
- Allowed category labels.

## Run the demo

Requires Python 3.10 or newer on macOS or Linux.

```bash
git clone https://github.com/KyleWu53/synthetic-data-audit.git
cd synthetic-data-audit
bash run.sh --mode demo
```

The script creates a virtual environment and installs the required packages.

Results are saved in `results/demo/`.

The demo uses local test data. One dataset deliberately shifts a mean and breaks the correlation between `x` and `y` so the audit has known problems to detect. Demo results are separate from the Gemini experiment.

## Run with Gemini

Create an API key through [Google AI Studio](https://aistudio.google.com/apikey).

In Terminal, start Bash:

```bash
bash
```

Enter your key at the prompt:

```bash
read -s -p 'Gemini API key: ' GEMINI_API_KEY
export GEMINI_API_KEY
echo
```

Set the model and run generation. Use a new output directory to preserve the committed results:

```bash
export GEMINI_MODEL='gemini-2.5-flash'

bash run.sh --mode gemini \
  --n 500 \
  --trials 3 \
  --out results/gemini-new
```

This configuration plans up to 18 generation requests before retries. If a CSV batch fails, the remaining CSV batches for that trial are skipped. API quota and charges depend on your account.

### Review and execute the samplers

The first run saves the generated Python scripts without executing them:

```text
results/gemini-new/trial_0/sampler/sampler.py
results/gemini-new/trial_1/sampler/sampler.py
results/gemini-new/trial_2/sampler/sampler.py
```

Review these files before running them. The runner uses a timeout, but it is not a security sandbox.

Preserve the original generation outcomes:

```bash
cp results/gemini-new/summary.csv results/gemini-new/generation_summary.csv
cp results/gemini-new/REPORT.md results/gemini-new/GENERATION_REPORT.md
```

Then execute the reviewed scripts:

```bash
bash run.sh --mode gemini \
  --n 500 \
  --trials 3 \
  --out results/gemini-new \
  --execute-reviewed-samplers
```

Use the same model, row count, trial count, seed, and output directory for both steps.

The execution step makes no new Gemini requests. It runs the saved samplers and reuses any complete CSV datasets.

Save the package versions and clear the API key from the shell:

```bash
.venv/bin/python -m pip freeze > results/gemini-new/environment.txt
unset GEMINI_API_KEY
```

## Output files

The committed experiment is in `results/gemini/`.

| File | Contents |
|---|---|
| `generation_summary.csv` | Original generation outcomes and failure reasons |
| `summary.csv` | Final evaluation results |
| `REPORT.md` | Generated report and averages |
| `comparison.png` | Comparison charts |
| `environment.txt` | Package versions used for the experiment |
| `trial_*/sampler/sampler.py` | Generated Python samplers |
| `trial_*/sampler/audit.json` | Detailed sampler audit results |

Prompts, available model responses, reference data, and generated datasets are also stored in the trial folders.

The final summary shows missing-file errors for direct CSV generation because the original generation attempts never produced complete valid datasets. The original failure reasons are preserved in `generation_summary.csv`.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The tests check reference correlation, input validation, Holm adjustment, and whether the audit detects a large distribution shift and broken correlation.

## Limitations

This is a small experiment using three trials and simulated reference data. It does not measure performance on real-world datasets or downstream prediction tasks.

A high KS p-value or an AUC near 0.5 does not prove that two distributions are identical. Results depend on the prompt, model, sample size, batch size, token limit, and classifier.

The token-limited response was not saved because the program raised an error before writing it to disk. Its failure reason is preserved in the generation summary.

Local sampling uses fixed seeds, but Gemini output can vary between runs. This project does not test privacy guarantees.
