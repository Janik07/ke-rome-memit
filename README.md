# Sequential knowledge editing in GPT-2 XL: ROME vs MEMIT

Code, data sample, run logs and analysis for the manuscript
**“Retention of Sequential Knowledge Edits in GPT-2 XL: A Paired Comparison of ROME and MEMIT”**
(Adilbekov D., Ermekov Z., Kissamenov A.; Astana IT University). Release **`final`** corresponds to the
submitted manuscript; release `assignment-4` is the earlier course version with the same results.

**Research question.** How do ROME and MEMIT differ in the retention of earlier edited facts
after 100 sequential CounterFact edits to GPT-2 XL?

**Main result.** After 100 sequential edits, MEMIT retained 100% of the earlier edits that had
worked under both methods in each of three edit orders; ROME retained 68.4% on average
(66.7%, 58.0%, 80.7%). Exact McNemar test per order: p ≤ 1.5 × 10⁻⁵; Cohen's h 0.91–1.41.

## Repository structure

```
src/run_experiment.py   experiment: sample selection, sequential editing, checkpoint evaluation (GPU)
src/analyze.py          retention, paired differences, exact McNemar test, bootstrap CI -> summary.json
src/report_numbers.py   prints every number of Tables 3-4 (means, SDs, exact p, Cohen's h)
src/error_analysis.py   error and subgroup analysis -> error_cases.csv, error_summary.json (Table 5)
src/plots.py            Figures 1-3
src/from_compact.py     converts the compact per-run records (RESULT_JSON lines) into run files
results/                sample.json (100 records, 3 orders), one JSON log per run, summary.json,
                        report_numbers.txt, error_cases.csv, error_summary.json,
                        compact_2nov/ (saved compact logs of the three runs executed on 2 Nov 2026)
figures/                Figures 1-3 of the manuscript (300 dpi)
reproduction/           notebook and evidence archive of the independent reproduction (5-6 Nov 2026)
requirements.txt        package versions used for the reported runs
```

## Where each table and figure of the manuscript comes from

| Manuscript item | Produced by | Output file |
|---|---|---|
| Table 1 (related work) | literature review, no code | – |
| Table 2 (configuration) | `run_experiment.py`, `requirements.txt` | settings recorded in each run file |
| Table 3 (outcomes after 100 edits) | `analyze.py`, then `report_numbers.py` | `results/summary.json`, `results/report_numbers.txt` |
| Table 4 (per-order test) | `analyze.py`, then `report_numbers.py` | `results/summary.json`, `results/report_numbers.txt` |
| Table 5 (failure cases), Section 4.5 subgroups | `error_analysis.py` | `results/error_cases.csv`, `results/error_summary.json` |
| Figures 1–2 | `plots.py` | `figures/fig1_retention.png`, `figures/fig2_para_neigh.png` |
| Figure 3 | `plots.py` (reads `error_summary.json`) | `figures/fig3_subgroups_errors.png` |

### Quick check from the released logs (no GPU, about one minute)

These commands regenerate the statistics, the error analysis and the figures from the six run logs in
`results/`; `summary.json`, `report_numbers.txt` and `error_summary.json` come out identical.

```bash
git clone https://github.com/Janik07/ke-rome-memit.git && cd ke-rome-memit && git checkout final
pip install matplotlib transformers      # transformers is used only for the tokenizer in error_analysis.py
cd src
python analyze.py ../results                          # Tables 3-4 -> results/summary.json
python report_numbers.py ../results/summary.json      # prints every number of Tables 3-4
python error_analysis.py ../results                   # Table 5 -> error_cases.csv, error_summary.json
python plots.py ../results ../figures                 # Figures 1-3
```

## Environment

The reported runs used Google Colab with an NVIDIA Tesla T4 (15 GiB), Python 3.13,
PyTorch 2.11.0+cu130 and Transformers 5.21.0 (see `requirements.txt`). A GPU with at least
~12 GiB of memory is needed: ROME on GPT-2 XL in fp32 peaked at 8.5 GiB in our pilot.

## Full reproduction of the experiment (Colab, T4 GPU)

```bash
# 0. This repository at the release used in the manuscript
git clone https://github.com/Janik07/ke-rome-memit.git /content/ke-rome-memit
cd /content/ke-rome-memit && git checkout final

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

# 4. Every table and figure of the Results section
python analyze.py /content/ke_results
python report_numbers.py /content/ke_results/summary.json
python error_analysis.py /content/ke_results
python plots.py /content/ke_results ../figures
```

In a Colab notebook, prefix each command with `!` and use `%cd <dir>` instead of `cd <dir>`:
each `!` line runs in its own shell, so a plain `!cd` does not change the directory for the next line.

Default paths: EasyEdit in `/content/EasyEdit`, CounterFact in `/content/counterfact.json`, results in
`/content/ke_results`. They can be changed with the environment variables `EE_DIR`, `KE_CF`, `KE_STATS`
and `KE_OUT`; on Colab, pointing `KE_OUT` to a Google Drive folder keeps the run files if the session ends.

To compare a reproduction with this release, compare the new run files and `summary.json` with the files in
`results/`, ignoring run times. `summary.json` stores McNemar p-values rounded to six decimals, so values
below 5 × 10⁻⁷ appear as 0.0; `report_numbers.py` prints the exact values (2 · 0.5^k for k discordant pairs).

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

A second, partial reproduction (Kissamenov Alan, 6 Nov 2026) repeated ROME order 0 in full and MEMIT
order 0 up to checkpoint 50; all observed values were identical to the files in `results/`.

## Data and code availability

Code in this repository is released under the MIT licence (see `LICENSE`). The 100-record sample, the
three edit orders, all run logs and all analysis outputs are included. CounterFact and the covariance
statistics are published by Meng et al. (https://rome.baulab.info) and are downloaded, not redistributed
here. GPT-2 XL is downloaded from Hugging Face. The edited model weights are not saved or shared,
because they can be regenerated from the code. The repository contains no personal data, credentials
or confidential material.

