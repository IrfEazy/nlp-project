# Decision log

History only; current rules live in the AGENTS.md files and docs/known-issues.md.

Dated 2026-10-31. Each entry: decision and reason. Commits by short hash.

- AGENTS.md layering chosen; see AGENTS.md Instruction files. (09ccae4, 752485c)
- Misspelled `gensin` fixed to `gensim`, then `gensim` removed as unused; `uv.lock` tracked for reproducible installs. (787c601, 226a21f)
- `tensorflow`, `tf-keras`, `transformers` and `accelerate` removed: nothing in `src/` needs them. Only the guarded optional import of tensorflow/torch in `configure_environment` remains. (226a21f)
- DistilBERT preset removed instead of implemented: it would need `transformers` and torch, and no strategy was registered. (0ac7102)
- `single` validates the model name and rejects invalid sample sizes and extra arguments with exit 1, instead of silently falling back. (0ac7102, 7b9f3aa)
- `matplotlib` and `numpy` declared because `src/` imports them directly; `pytest` and `pytest-cov` moved to the `dev` group; unused `slow` marker dropped. (f54599a)
- All dependencies re-added one by one so uv chose the lower bounds. Observed 2026-10-31 (HF Hub API and a real run): this upgraded to datasets 5.1 and huggingface-hub 2.4 (httpx2 stack; provenance checked). (6e5100b)
- Dataset id changed to `Yelp/yelp_review_full`; same revision (c1f9ee93). Observed 2026-10-31 (HF Hub API and a real run): datasets 5 / huggingface-hub 2 rejected the legacy bare id. (f229335)
