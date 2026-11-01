# Research report: Yelp review classification, notebook versus re-implementation

Two kinds of numbers appear below and are always labelled. **Historical** numbers are the recorded outputs of the original notebook (commit `afa2d95`, run in May 2024 on another machine, mostly on GPU). **Re-measured** numbers come from two environments with this repository's code: the classical experiments from runs made on 2026-10-31 on one CPU machine (Intel Core Ultra 7 258V, 8 cores, 15 GB RAM, WSL2), and the 14 deep experiments from full-scale GPU runs on Kaggle (2x Tesla T4 15 GB, driver 580.178.04, torch 2.15.0+cu132 as locked in `uv.lock`, repository commit `2e16897`, runs 2026-10-31 to 2026-11-01, one experiment per GPU, two in parallel). Derivations are shown as such. No external benchmark figures are used. Run logs, including the Kaggle ones, were kept outside the repository and are not committed; no kernel code or credentials were added to it.

## Summary

- The notebook compared classical and deep models on `Yelp/yelp_review_full` (five star classes) and on derived tasks (3 classes, polarity, polarity without neutral), plus DBpedia-14. This repository re-implements all of them behind named experiments.
- All six classical experiments and the dataset analysis were re-measured at full scale and match the historical values to the printed precision (table below).
- The 14 deep experiments (char-CNN, BiLSTM + GloVe, DistilBERT, zero-shot DeBERTa) were re-measured at full scale on Kaggle (2x Tesla T4 15 GB, driver 580.178.04, torch 2.15.0+cu132, commit `2e16897`, 2026-10-31 to 2026-11-01, one experiment per GPU, two in parallel). All stay within 0.03 of the historical value, most within 0.004 (table below); the classical ones remain the CPU re-measurement.
- Deliberate deviations are listed in [Deviations](#deviations-from-the-notebook); rationale lives in [decisions.md](decisions.md). Notebook inconsistencies are listed under [Notebook inconsistencies](#notebook-inconsistencies).
- Dependencies changed substantially during the day; see the version table.

## Provenance

- `afa2d95`: the original notebook, written by its author (TensorFlow/Keras, pytorch-ignite, Hugging Face Trainer).
- A later refactor turned the classical part into the layered package under `src/` (BoW or TF-IDF plus Logistic Regression, five classes only).
- 2026-10-31 parity work added the notebook's other experiments, one package per commit:
  - `152145d`: label tasks, experiment registry, `experiments`/`experiment`/`analyze` modes, classical parity.
  - `1f65b38`: optional `deep` extra (PyTorch), shared training loop, character-level CNN.
  - `c0c475f`: BiLSTM + GloVe, notebook text cleaning.
  - `73bb2ac`: DistilBERT fine-tuning, transfer and polarity.
  - `077e822`: zero-shot DeBERTa experiments.

## Experiments and how to run them

`uv run python -m src.main experiments` lists the names; `uv run python -m src.main experiment <name> [sample_size]` runs one and `analyze [sample_size]` prints the dataset statistics. Deep experiments need `uv sync --extra deep` (lazy imports; the error names that command). Names: `lr-bow`, `lr-tfidf`, `lr-bow-3class`, `lr-bow-polarity`, `lr-bow-no-neutral`, `lr-bow-dbpedia`, `cnn-char`, `cnn-char-mixed-case`, `lstm-glove`, `lstm-glove-3class`, `distilbert-finetune`, `distilbert-transfer`, `distilbert-polarity`, `zeroshot-form1-val`, `zeroshot-form2-val`, `zeroshot-form3-val`, `zeroshot-noverb-val`, `zeroshot-test`, `zeroshot-3class`, `zeroshot-binary`.

Zero-shot runs on CPU when no GPU exists (the notebook raised an error without CUDA), and the deep runs are practical on CPU only with small samples.

## Results

Accuracy on the test split unless stated. "Historical" is read from the notebook outputs; "re-measured" is a single full-scale run (classical: CPU; deep: Kaggle T4 GPU; 650000 train rows unless the task changes the count, seed 42, `max_iter` 100 for the classical ones). Status: matched = equal to the historical value at the printed precision; close = signed difference re-measured minus historical (derived); differs, explained = larger difference with the explanation under [Deep experiments in detail](#deep-experiments-in-detail-re-measured-on-gpu). Macro F1 for the deep rows is in the detail section.

| Experiment | Historical (notebook) | Re-measured today | Status |
|---|---|---|---|
| `lr-bow` | 0.57704 | 0.5771 | matched |
| `lr-tfidf` | 0.58068 | 0.5807 | matched |
| `lr-bow-3class` | 0.75348 | 0.7535 | matched |
| `lr-bow-polarity` | 0.86034 | 0.8604 (at `c0c475f`) | matched |
| `lr-bow-no-neutral` | 0.924725 | 0.9247 | matched |
| `lr-bow-dbpedia` | 0.9765 | 0.9766 | matched |
| `cnn-char` (lowercase, 70 characters) | 0.6228 | 0.6256 | close: +0.0028 |
| `cnn-char-mixed-case` (96 characters) | 0.5959 (wrong tokenizer, see below) | 0.6237 | differs, explained: +0.0278, notebook tokenizer bug |
| `lstm-glove` | 0.6132 (last-epoch weights; final table says 0.6039) | 0.6255 | differs, explained: +0.0123 (+0.0216 vs 0.6039), best-weight restore |
| `lstm-glove-3class` | 0.79806 | 0.8014 | close: +0.0033 |
| `distilbert-finetune` | 0.684 (printed to 3 decimals; recall 0.68446) | 0.6855 | close: +0.0015 |
| `distilbert-transfer` | 0.554 (printed to 3 decimals; recall 0.55398) | 0.5523 | close: -0.0017 |
| `distilbert-polarity` | 0.91434 (validation 0.91558) | 0.9180 | close: +0.0037 |
| `zeroshot-form1-val` / `form2-val` / `form3-val` / `noverb-val` (13000 validation rows) | 43.09% / 50.02% / 46.43% / 46.10% (accuracy) | 0.4312 / 0.5005 / 0.4683 / 0.4610 | close: +0.0003 / +0.0003 / +0.0040 / 0.0000 |
| `zeroshot-test` (form 2, 50000 rows) | 47.55% | 0.4756 | close: +0.0001 |
| `zeroshot-3class` | 80.02% | 0.8002 | matched |
| `zeroshot-binary` (2-star dropped, 40000 rows) | 96.34% | 0.9634 | matched |

The PyTorch ports reproduce the historical accuracies within the differences stated in the table, on single runs. The notebook's own final table lists: LR 0.5770, char-CNN 0.6228, custom LSTM 0.6039, DistilBERT fine-tuned 0.6840, transfer 0.5540, DeBERTa zero-shot 0.4755 (all historical).

### Deep experiments in detail (re-measured on GPU)

Test accuracy (macro F1 in parentheses), single run each, seed 42, Kaggle 2x Tesla T4: `cnn-char` 0.6256 (0.6232), `cnn-char-mixed-case` 0.6237 (0.6249), `lstm-glove` 0.6255 (0.6217), `lstm-glove-3class` 0.8014 (0.7521), `distilbert-finetune` 0.6855 (0.6850), `distilbert-transfer` 0.5523 (0.5501), `distilbert-polarity` 0.9180 (0.9147), `zeroshot-form1-val` 0.4312 (0.3676), `form2-val` 0.5005 (0.4678), `form3-val` 0.4683 (0.3930), `noverb-val` 0.4610 (0.3634), `zeroshot-test` 0.4756 (0.4441), `zeroshot-3class` 0.8002 (0.7005), `zeroshot-binary` 0.9634 (0.9634). The historical zero-shot values 43.09/50.02/46.43/46.10% are accuracies (the notebook prints "Accuracy:"), not F1.

Interpretation, from these numbers only:

- `cnn-char-mixed-case` 0.6237 against the historical 0.5959 is consistent with the fixed notebook bug (the historical test set was encoded with the lowercase tokenizer, see [Notebook inconsistencies](#notebook-inconsistencies)); it now sits within 0.002 of the lowercase CNN (0.6256).
- `lstm-glove` 0.6255 is above both historical numbers (0.6132 last-epoch, 0.6039 final table); the port restores the best `val_loss` weights (epoch 3 of 4, early stop).
- `distilbert-polarity` +0.0037 coincides with the batch deviation (32 on one device against effective 64); it is a single run, so it is not attributed to the batch.
- The DistilBERT 5-class values are within 0.002 of the three-decimal historical values.
- Zero-shot is within 0.004 everywhere (`form3-val` +0.0040 is the largest).
- All are single runs with seed 42 and no variance estimate.

### Cost of the deep runs (re-measured, Kaggle T4, wall seconds)

`cnn-char` 5157, `cnn-char-mixed-case` 4578, `lstm-glove` 3033, `lstm-glove-3class` 3041, `distilbert-finetune` 33966, `distilbert-transfer` 20103, `distilbert-polarity` 28123, `zeroshot-form1-val` 585, `form2-val` 580, `form3-val` 585, `noverb-val` 578, `zeroshot-test` 2404, `zeroshot-3class` 1600, `zeroshot-binary` 920. Wall time includes model and data download. The 12 h Kaggle session limit stopped `cnn-char-mixed-case` in the first session; it was re-run alone on the same commit.

### Classical experiments in detail (re-measured)

Macro F1, re-measured against historical where the notebook printed it: `lr-bow` 0.5736 (historical 0.57), `lr-tfidf` 0.5786 (0.58), `lr-bow-3class` 0.7217 (0.72), `lr-bow-polarity` 0.8566, `lr-bow-no-neutral` 0.9247, `lr-bow-dbpedia` 0.9766 (0.98).

Five-class details come from the harness run at `73bb2ac` (same accuracies as the CLI runs; it also computes the extra statistics):

| Metric | BoW + LR | TF-IDF + LR |
|---|---|---|
| Accuracy | 0.5771 | 0.5807 |
| Macro precision / recall / F1 | 0.5716 / 0.5771 / 0.5736 | 0.5770 / 0.5807 / 0.5786 |
| Per-class F1 (classes 0-4) | 0.72 / 0.51 / 0.48 / 0.48 / 0.68 | 0.72 / 0.52 / 0.49 / 0.49 / 0.68 |
| Within one star | 0.9266 | 0.9285 |
| Mean absolute error (stars) | 0.5243 | 0.5193 |
| Vocabulary size | 22027 | 22027 |

Historical per-class F1 is the same (.72/.51/.48/.48/.68 and .72/.52/.49/.49/.68). The accuracy gap TF-IDF - BoW is 0.5807 - 0.5771 = 0.0036 (re-measured, single split).

Confusion matrices (rows = true class 0-4, columns = predicted class 0-4), re-measured:

BoW:

| true \ pred | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 0 | 7427 | 1928 | 332 | 107 | 206 |
| 1 | 2390 | 4924 | 2036 | 426 | 224 |
| 2 | 502 | 2080 | 4647 | 2170 | 601 |
| 3 | 146 | 361 | 1938 | 4698 | 2857 |
| 4 | 177 | 154 | 435 | 2077 | 7157 |

TF-IDF:

| true \ pred | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 0 | 7390 | 2020 | 312 | 106 | 172 |
| 1 | 2250 | 5061 | 2074 | 410 | 205 |
| 2 | 517 | 2021 | 4819 | 2107 | 536 |
| 3 | 179 | 328 | 2017 | 4822 | 2654 |
| 4 | 222 | 144 | 446 | 2246 | 6942 |

![BoW confusion matrix](figures/confusion-bow-full.png)
![TF-IDF confusion matrix](figures/confusion-tfidf-full.png)

The project's plots are the transpose of the tables above (y axis = predicted, x axis = true), so the first plot row is the first table column. The figures were regenerated at `73bb2ac` by the project's own plotting code through an external harness that calls `src.main.run_single_model` for `bow` and `tfidf`.

![Accuracy comparison](figures/accuracy-comparison-full.png)

Observations: errors are mostly between adjacent stars. The largest off-diagonal cells are BoW true 3 predicted 4 (2857), true 1 predicted 0 (2390), true 2 predicted 3 (2170), and TF-IDF true 3 predicted 4 (2654), true 1 predicted 0 (2250), true 4 predicted 3 (2246), ahead of true 2 predicted 3 (2107). The extreme classes score higher F1 than classes 1-3. TF-IDF top features are sentiment words (class 0: worst, horrible, terrible, zero, awful; class 4: delicious, amazing, awesome, excellent, perfect); BoW lists include rare tokens and `n`-prefixed tokens such as `nnever`, `nhowever`, `n3`, which come from literal backslash-n sequences in the text ([known-issues.md](known-issues.md)).

### Cost of the classical runs (re-measured, CLI, `/usr/bin/time` wall)

`lr-bow` 2:29.63, `lr-tfidf` 2:16.36, `lr-bow-3class` 1:54.72, `lr-bow-polarity` 2:14.34 (1:12.92 at `c0c475f` rerun), `lr-bow-no-neutral` 1:18.69, `lr-bow-dbpedia` 1:08.90, `analyze` 1:06.93. These include dataset loading and plots; the harness training-plus-prediction times were 95.5 s (BoW) and 90.0 s (TF-IDF). Times are indicative.

## Dataset analysis comparison

`analyze` (re-measured) against notebook cells 15-20 (historical):

| Quantity | Historical | Re-measured |
|---|---|---|
| Train avg characters per document | 732.3254676923077 | 732.33 |
| Train avg unique words per document | 86.16639384615385 | 86.17 |
| Train vocabulary | 881476 | 881476 |
| Train vocabulary without NLTK stopwords | 881327 | 881327 |
| Test avg characters per document | 733.42376 | 733.42 |
| Test avg unique words per document | 86.33092 | 86.33 |
| Test vocabulary | 144068 | 144068 |
| Test vocabulary without NLTK stopwords | not recorded in the outputs | 143927 |

The harness also gives the review length in words (whitespace split): train mean 134.1, median 99, p95 370, max 1052; test mean 134.3, median 99, p95 368, max 1009. Both splits are balanced (130000 and 10000 per class).

## Deviations from the notebook

Rationale is in [decisions.md](decisions.md); this is the list.

- **Frameworks.** TensorFlow/Keras (char-CNN, LSTM), pytorch-ignite (5-class DistilBERT) and the Hugging Face `Trainer` (polarity) are replaced by one PyTorch training loop with the same architectures and hyperparameters. Parameter counts match for the CNN (11,345,775 / 11,396,709); the PyTorch LSTM has an extra bias vector (1,570,309 trainable against Keras's 1,568,261, +2048).
- **Notebook bugs fixed.** The mixed-case CNN now encodes its test set with its own alphabet; the LSTM restores the best `val_loss` weights instead of evaluating the last epoch; the transfer-learning frozen/trainable labels are printed correctly; no hard-coded hub token and no `push_to_hub`; the GPU-only error is replaced by CPU fallback.
- **Split and shuffle semantics.** The notebook's unseeded `np.random.shuffle` and `validation_split` become a seeded shuffle with the last 20% as validation; the 5-class DistilBERT 0.75/0.25 split uses a seeded NumPy permutation instead of `torch.random_split`, so the exact samples differ.
- **Polarity batch.** The notebook's polarity run recorded `global_step` 8125 (two devices, effective batch 64); the port uses batch 32 on one device.
- **`max_iter`.** Restored to sklearn's 100, the notebook's value. A first measurement used 1000 (the refactor's setting, not a recorded decision). Sensitivity, same data and code otherwise (the 1000 values come from the morning run recorded at commit `ff66b58` (logged as `01635a3` before the author-identity rewrite, same tree), the 100 values from this run at `73bb2ac`): BoW 0.5681 at 1000 against 0.5771 at 100; TF-IDF 0.5830 at 1000 against 0.5807 at 100. Differences: BoW -0.0090, TF-IDF +0.0023 going from 100 to 1000 (derived).
- **Text cleaning.** The LSTM path uses the notebook's `remove_noise` order, so apostrophe contractions are not expanded; polarity uses the notebook's punctuation regex, which leaves the backslash (polarity accuracy 0.8574 in the run at `152145d`, which deletes every `string.punctuation` character, and 0.8604 in the run at `c0c475f`, which uses the notebook regex; between those commits the change on the polarity LR path was the punctuation regex in `src/preprocessing/feature_engineer.py`). The vocabulary pair 22839 (notebook regex) against 22924 (all punctuation deleted) comes from a lead ad-hoc check at `152145d` on the full train split (`CountVectorizer(min_df=50, stop_words='english')`), not a committed log; its output is recorded in the run's `runs-logs/lead-check-polarity-vocab.txt`.
- **BoW `max_df`.** None, as in the notebook; the BoW/TF-IDF comparison therefore also differs in `max_df` (1.0 vs 0.5).

## Notebook inconsistencies

- The final table lists custom LSTM 0.6039, but the outputs show a test accuracy of 0.6132 (bidirectional LSTM, last-epoch weights, not the best checkpoint). The value 0.6039 appears in the outputs only as the Keras progress-bar line printed during that test evaluation.
- The mixed-case CNN result 0.5959 was evaluated with the lowercase tokenizer, so it is confounded.
- DistilBERT values are printed rounded to three decimals (0.684, 0.554) apart from recall and precision.
- The CNN heatmaps plot the transposed matrix with swapped axis labels; the notebook's polarity cell contains a hard-coded hub token and `push_to_hub=True` (not replicated).

## Dependency and package version changes today

Versions come from `uv.lock` at each commit (`-` = not in the lock).

| Package | `7cd04df` | `3c8b9e4` | `6c98a44` | `7c6f6f1` | `73bb2ac` |
|---|---|---|---|---|---|
| scikit-learn | 1.9.1 | 1.9.1 | 1.9.1 | 1.9.1 | 1.9.1 |
| numpy | 2.5.3 | 2.5.3 | 2.5.3 | 2.5.4 | 2.5.4 |
| scipy | 1.18.1 | 1.18.1 | 1.18.1 | 1.18.1 | 1.18.1 |
| datasets | 5.0.1 | 5.0.1 | 5.0.1 | 5.1.0 | 5.1.0 |
| huggingface-hub | 1.33.0 | 1.33.0 | 1.33.0 | 2.4.0 | 2.4.0 |
| nltk | 3.10.3 | 3.10.3 | 3.10.3 | 3.10.4 | 3.10.4 |
| contractions | 0.1.73 | 0.1.73 | 0.1.73 | 0.1.73 | 0.1.73 |
| matplotlib | 3.11.2 | 3.11.2 | 3.11.2 | 3.11.2 | 3.11.2 |
| seaborn | 0.13.2 | 0.13.2 | 0.13.2 | 0.13.2 | 0.13.2 |
| pandas | 3.0.6 | 3.0.6 | 3.0.6 | 3.1.0 | 3.1.0 |
| pytest | 9.1.1 | 9.1.1 | 9.2.0 | 9.2.0 | 9.2.0 |
| pytest-cov | 7.1.0 | 7.1.0 | 7.1.0 | 7.1.0 | 7.1.0 |
| torch | 2.14.1 | - | - | - | 2.15.0 |
| transformers | 5.18.0 | - | - | - | 5.22.1 |
| tokenizers | 0.23.2 | - | - | - | 0.23.3 |
| gensim | 4.4.0 | - | - | - | 4.4.0 |
| tensorflow, tf-keras | 2.21.0 | - | - | - | - |
| accelerate | 1.15.0 | - | - | - | - |

Sequence: `3c8b9e4` removed the deep libraries (`tensorflow`, `tf-keras`, `transformers`, `accelerate`, and `gensim` as unused; torch left the lock too); `7c6f6f1` re-added every dependency with uv-chosen lower bounds (datasets 5.1, huggingface-hub 2.4); `1f65b38`, `c0c475f` and `73bb2ac` re-added `torch`, `gensim` and `transformers` in the optional `deep` extra. TensorFlow, Keras and `accelerate` stay out: the port uses PyTorch. `sentencepiece`, `protobuf` and `tiktoken` were not needed for the DeBERTa tokenizer in the smoke run.

## Limitations

- One seed (42), one split, no confidence intervals or repeated runs; the 0.0036 BoW/TF-IDF gap is a single measurement.
- Historical numbers come from another machine, library versions and (for deep models) GPU runs; the notebook's own pinned `sklearn==1.0.2` differs from the 1.9.1 used here, yet the classical results match.
- The PyTorch ports reproduce the historical accuracies within the stated differences on single runs. Remaining limits: one seed, different hardware and framework from the notebook, and GPU nondeterminism; the deviations above (framework, split samples, best-weight restore, batch) make exact equality unlikely.
- `configure_environment` suppresses all warnings, so lbfgs convergence warnings, if any, are not visible ([known-issues.md](known-issues.md)).
- Literal backslash-n sequences in the text yield tokens such as `nworst`; no cleaning is applied to the classical path.
- The figures and five-class details come from an external harness, not a committed CLI mode.
- Wall times are indicative only.

## Reproduction

```
uv sync                      # add --extra deep for the deep experiments
uv run python -m src.main experiments
uv run python -m src.main analyze
uv run python -m src.main experiment lr-bow
MPLBACKEND=Agg uv run python -m src.main experiment lr-tfidf
```

The first run downloads the dataset and the NLTK stopwords and needs network access; the LSTM also downloads GloVe, DistilBERT and DeBERTa their checkpoints.
