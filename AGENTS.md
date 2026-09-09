# NavIO — dev environment notes

Gradio 6 Space (single Python process, no database). Run it with:

```bash
docker compose -f docker-compose.base44.yml up -d   # serves on host port 3000
```

## Non-obvious things

**The `artifacts/` files are broken Git LFS pointers.** All five (`dataset_embeddings.npy`
and the four `.joblib` models) are ~130-byte pointer files; the LFS objects return 404 from
the GitHub remote, so `git lfs pull` cannot restore them. `recsys.py` asserts on embedding
shape at import time and would crash.

Workaround: `scripts/make_mock_artifacts.py` fits throwaway Ridge/LogisticRegression models
on random 768-dim hybrid vectors and writes them to `NAVIO_ARTIFACT_DIR` (a Docker volume,
so the repo's real pointer files are never overwritten). `recsys.py` reads `ARTIFACT_DIR`
from that env var, defaulting to `./artifacts`.
**Predictions are therefore meaningless in dev.** To restore real behaviour, put the real
artifacts in `artifacts/` and delete `NAVIO_ARTIFACT_DIR` from the compose file.

**`HF_TOKEN` is optional.** The dataset (`ofekponzo/navio-synthetic-dataset`, 11,000 rows /
2,200 patients) and both models (`all-mpnet-base-v2`, `distilbert-...-sst-2-english`) are
public and download anonymously. The token is only needed for the Inference API calls in
`generation.py`; without it `MOCK_GENERATION` is on and both clinical notes come back as
templated text prefixed `[MOCK — set HF_TOKEN for real generation]`.

**First boot takes ~2-4 minutes** (installs CPU torch, downloads ~500 MB of models, fits
mock artifacts). Later boots take ~90 s — the `pip-cache`, `hf-cache` and `mock-artifacts`
volumes persist. Everything loads at import time, not per request.

**`spaces` is not installed** (HF-only package). `app.py` guards the import, so `HAS_SPACES`
is False locally and the ZeroGPU probe is skipped. Don't add it to requirements.

## Verifying it works

Login is a demo gate in `auth.py` — no real auth. Use **Dr. Sarah Cohen** / `000000010`
(other demo clinicians: `000000028`, `000000036`). After login the sidebar shows
Dashboard / Patients / Sessions; the dashboard should list a caseload count and 8
appointments. To exercise the full pipeline, open an appointment, paste a transcript using
`[MM:SS] Speaker:` line markers (Graph 2 skips turns without them), and click
*Analyze Session*.

New patients/sessions created in the UI live only in that browser session's `gr.State`
and are lost on refresh — by design, not a bug.
