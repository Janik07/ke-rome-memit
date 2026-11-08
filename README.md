# Sequential knowledge editing in GPT-2 XL: ROME vs MEMIT

Course project for *Research Methods and Tools* (AITU, group se-2414).
Authors: Adilbekov Daniyal, Ermekov Zhandaulet, Kissamenov Alan.

**Research question.** How do ROME and MEMIT differ in the retention of earlier edited facts
after 100 sequential CounterFact edits to GPT-2 XL?

**Main result.** After 100 sequential edits, MEMIT retained 100% of the earlier edits that had
worked under both methods in each of three edit orders; ROME retained 68.4% on average
(66.7%, 58.0%, 80.7%). Exact McNemar test per order: p ≤ 1.5 × 10⁻⁵.

## Repository structure

```
src/run_experiment.py   experiment: sample selection, sequential editing, checkpoint evaluation
src/analyze.py          retention, paired differences, exact McNemar test, bootstrap CI -> summary.json
src/plots.py            Figures 1-2 from summary.json
src/from_compact.py     converts the compact per-run records (RESULT_JSON lines) into run files
src/error_analysis.py   error analysis and per-relation results (Assignment 4, Part 4)
results/                sample.json (100 records, 3 orders), one JSON log per run, summary.json,
                        error_cases.csv + error_summary.json (error/subgroup analysis),
                        compact_2nov/ (saved compact logs of the three runs executed on 2 Nov 2026)
figures/                figures used in the report
requirements.txt        package versions used for the reported runs
```

## Environment

The reported runs used Google Colab with an NVIDIA Tesla T4 (15 GiB), Python 3.13,
PyTorch 2.11.0+cu130 and Transformers 5.21.0 (see `requirements.txt`). A GPU with at least
~12 GiB of memory is needed: ROME on GPT-2 XL in fp32 peaked at 8.5 GiB in our pilot.

## Reproduce the main results (Colab, T4 GPU)

```bash
# 0. This repository at the release used in the report
git clone https://github.com/Janik07/ke-rome-memit.git /content/ke-rome-memit
cd /content/ke-rome-memit && git checkout assignment-4

# 1. EasyEdit at the exact commit used, and the project dependencies
#    (do not install EasyEdit's own requirements.txt: it fails to build on Python 3.13)
git clone https://github.com/zjunlp/EasyEdit.git /content/EasyEdit
cd /content/EasyEdit && git checkout 4c109870955a4522ac3d7cf10ad00f34de8e4f0d
pip install -r /content/ke-rome-memit/requirements.txt

# 2. CounterFact and the precomputed covariance statistics released by the ROME/MEMIT authors
wget https://rome.baulab.info/data/dsets/counterfact.json -O /content/counterfact.json
mkdir -p /content/EasyEdit/data/stats/gpt2-xl/wikipedia_stats
cd /content/EasyEdit/data/stats/gpt2-xl/wikipedia_stats
for L in 13 14 15 16 17; do
  wget https://rome.baulab.info/data/stats/gpt2-xl/wikipedia_stats/transformer.h.$L.mlp.c_proj_float32_mom2_100000.npz
done

# 3. Sample (100 records, seed 42) and the six runs (~2 h 50 min to ~3 h 10 min on a T4)
cd /content/ke-rome-memit/src
python run_experiment.py prepare
for o in 0 1 2; do python run_experiment.py run memit $o; python run_experiment.py run rome $o; done

# 4. Statistics, error analysis and figures
python analyze.py /content/ke_results
python error_analysis.py /content/ke_results
python plots.py /content/ke_results ../figures
```

Default paths: EasyEdit in `/content/EasyEdit`, CounterFact in `/content/counterfact.json`, results in
`/content/ke_results`. They can be changed with the environment variables `EE_DIR`, `KE_CF`, `KE_STATS`
and `KE_OUT`; on Colab, pointing `KE_OUT` to a Google Drive folder keeps the run files if the session ends.

To compare a reproduction with this release, compare the new run files and `summary.json` with the files in
`results/`, ignoring run times. `summary.json` stores McNemar p-values rounded to six decimals, so values
below 5 × 10⁻⁷ appear as 0.0 (exact values: 2 · 0.5^k for k discordant pairs, see the report).

Seeds: sample 42; edit orders 1, 2, 3; PyTorch/Python seed 0. ROME uses the original GPT-2 XL
configuration (covariance adjustment on); MEMIT uses the published GPT-2 XL configuration and
receives one request per update.

## Provenance of the result files

ROME order 0 and MEMIT orders 0-1 were run on 2 Nov 2026; their per-edit logs were recovered from the
notebook output (`results/compact_2nov`, converted with `src/from_compact.py`). ROME orders 1-2 and
MEMIT order 2 were re-run on 3 Nov 2026 because their logs were lost when the Colab session ended; all
their quality metrics matched the 2 Nov values exactly at every checkpoint. These three re-runs recorded
Transformers 5.22.0; all other runs used 5.21.0.

An independent reproduction of all six runs from this README (Adilbekov Daniyal, 5-6 Nov 2026, Colab T4,
Transformers 5.21.0) matched the sample, edit orders, per-edit success flags, checkpoint metrics and
`summary.json` exactly; only run times differed.

## Data and licences

CounterFact and the covariance statistics are published by Meng et al. (https://rome.baulab.info)
and are downloaded, not redistributed here. GPT-2 XL is downloaded from Hugging Face. The edited
models are not saved or shared. Code in this repository: MIT licence (see `LICENSE`).
