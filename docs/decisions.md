# Decision log

History only; current rules live in the AGENTS.md files and docs/known-issues.md.

Dated 2026-10-31. Each entry: decision and reason. Commits by short hash.

- AGENTS.md layering chosen; see AGENTS.md Instruction files. (3fba368, c05fc4d)
- Misspelled `gensin` fixed to `gensim`, then `gensim` removed as unused; `uv.lock` tracked for reproducible installs. (7cd04df, 3c8b9e4)
- `tensorflow`, `tf-keras`, `transformers` and `accelerate` removed: nothing in `src/` needs them. Only the guarded optional import of tensorflow/torch in `configure_environment` remains. (3c8b9e4)
- DistilBERT preset removed instead of implemented: it would need `transformers` and torch, and no strategy was registered. (58b2bfa)
- `single` validates the model name and rejects invalid sample sizes and extra arguments with exit 1, instead of silently falling back. (58b2bfa, d99e7c5)
- `matplotlib` and `numpy` declared because `src/` imports them directly; `pytest` and `pytest-cov` moved to the `dev` group; unused `slow` marker dropped. (6c98a44)
- All dependencies re-added one by one so uv chose the lower bounds. Observed 2026-10-31 (HF Hub API and a real run): this upgraded to datasets 5.1 and huggingface-hub 2.4 (httpx2 stack; provenance checked). (7c6f6f1)
- Dataset id changed to `Yelp/yelp_review_full`; same revision (c1f9ee93). Observed 2026-10-31 (HF Hub API and a real run): datasets 5 / huggingface-hub 2 rejected the legacy bare id. (7f6f07c)
- Notebook-parity work (2026-10-31): `ModelConfig.max_iter` restored to sklearn's 100, the original notebook's value (1000 came from the refactor commit 7f15775, not a recorded decision); still configurable. `ModelConfig.num_classes` removed: the class count comes from the label task.
- Label tasks (`stars5`, `stars3`, `polarity`, `polarity_no_neutral`, `dbpedia14`) live in the data layer (`src/data/tasks.py`) and are applied by `DataLoader` before first-N sampling, so `sample_size` counts rows that survive the task.
- Named experiment registry (`EXPERIMENTS` in `src/config.py`) exposed as CLI modes `experiments`, `experiment <name>` and `analyze`; deep-model experiments will use `ModelFactory.register_lazy` so torch/transformers are never imported at CLI start-up. `class_weight='balanced'` equals the notebook's inverse-frequency weights.
- BoW deliberately gets no `max_df` (the notebook's `CountVectorizer(min_df=50, stop_words='english')` has none); the known issue was closed as intended. The analyzer deletes punctuation instead of replacing it with a space, as notebook cell 17 does.
- Author and committer of the 11 commits made with the machine-local address `icela@DAT38251.replynet.prv` rewritten to the GitHub no-reply identity, so GitHub links them to the account; dates and trees unchanged, hashes in this log and in docs/research-report.md updated to the rewritten commits.
