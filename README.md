# mil

Mech Interp Lab: lightweight IPython capture plus a small set of practical
mechanistic-interpretability tools.

Core loop:

```python
%load_ext mil.capture

import mil
from mil.tools import load_model, get_activations, patch, check_controls

model = load_model("gpt2", device="cpu")
acts = get_activations(model, prompts=["The cat sat on"], hooks=["blocks.0.hook_resid_post"])
result = patch(
    model,
    hook="blocks.0.hook_resid_post",
    source=0.0,
    target_prompts=["The cat sat on"],
    metric="target_logit_delta",
    target_tokens=[" the"],
)
report = check_controls(result.target_mean, result.control_deltas)
ref = mil.track(result, "first_patch")
mil.recent()
```

`mil` does not provide claim adjudication, evidence graphs, run manifests,
dashboards, or automatic tensor serialization.

## Running the explorer (Ubuntu 24 / WSL2, RTX 5060 Ti)

Requirement: `uv` must be installed and on `PATH`. Ubuntu 24 marks the system
Python as externally managed, so do not install project packages into the system
interpreter.

```bash
uv --version
```

Create or reuse the project `.venv`, exact-sync the extras, install Playwright's
Chromium binary, and run the Blackwell CUDA kernel gate:

```bash
./scripts/bootstrap_uv.sh
```

That script expands to:

```bash
uv venv --python 3.12 --allow-existing .venv
uv pip install --python .venv/bin/python --torch-backend cu128 --exact -e ".[models,app,viz,sae,evidence,dev]"
.venv/bin/python -m playwright install chromium
.venv/bin/python scripts/gpu_check.py
```

The documented Torch backend is `cu128` because it passed on the tested RTX
5060 Ti (`sm_120`). This may change as PyTorch updates stable Blackwell support;
check https://pytorch.org/get-started/locally/ before changing `TORCH_BACKEND`.

### Quick Guided Demo

This is the onboarding path. It opens a pre-populated SELF-GROUND-style negation
walkthrough with behavior bars, tokenization, logit-lens curves, layer/token
causal heatmaps, SAE feature specificity, intervention deltas, residual
trajectory geometry, and attention drill-down already populated from
`artifacts/guided_demo/`.

```bash
.venv/bin/streamlit run apps/explorer.py -- --demo guided
```

For WSL browser access with explicit binding and port:

```bash
.venv/bin/streamlit run apps/explorer.py --server.address 0.0.0.0 --server.port 8501 -- --demo guided
```

Open from Windows:

```text
http://localhost:8501
```

If localhost forwarding is not working for this WSL version, get the WSL VM IP:

```bash
hostname -I
```

Then open `http://<wsl-ip>:8501`.

### Free Exploration

Free exploration exposes the model, device, hook, prompt source, attention, patch,
control-check, token-delta, activation, and SAE controls directly. It may be slow
because model loads, activation collection, patching, and SAE ranking happen on
demand.

```bash
.venv/bin/streamlit run apps/explorer.py -- --demo free
```

Use `pythia-70m-deduped` with
`pythia-70m-deduped-res-sm/blocks.2.hook_resid_post` for the SAE views. Plain
`pythia-70m` can run attention, activation, patch, and token-delta views, but it
does not match that hosted SAELens release, so the app will not encode its
activations with the deduped SAE.

### Artifact Generation

The guided demo prefers committed artifacts so the first screen is useful
immediately. Rebuild them after changing the model, hook, prompt family, or viz
contracts:

```bash
.venv/bin/python scripts/build_guided_demo_artifacts.py --device cuda --artifacts-dir artifacts/guided_demo
```

The builder writes:

```text
artifacts/guided_demo/demo_manifest.json
artifacts/guided_demo/prompt_family.json
artifacts/guided_demo/tokenization.json
artifacts/guided_demo/behavior_logits.json
artifacts/guided_demo/logit_lens.json
artifacts/guided_demo/causal_heatmap.json
artifacts/guided_demo/*heatmap*.html
artifacts/guided_demo/feature_raster.html
artifacts/guided_demo/token_deltas.html
artifacts/guided_demo/residual_trajectory.html
```

### Compatibility Defaults

The guided demo default is:

```text
model: pythia-70m-deduped
hook: blocks.2.hook_resid_post
SAE release: pythia-70m-deduped-res-sm
SAE id: blocks.2.hook_resid_post
device: cuda
```

That pair is intentionally different from plain `pythia-70m`: the compatible SAE
release is for the deduped model. If an SAE load fails in your environment,
rerun the bootstrap, then rebuild the artifacts with the command above and check
the printed `feature_status`.

### Troubleshooting

- If you see `externally-managed-environment`, you used the wrong installer path.
  Use `uv` and the project `.venv`; do not install into Ubuntu's system Python.
- If `nvidia-smi` works but `torch.cuda.is_available()` is false, or
  `torch.__version__` ends in `+cpu`, rerun `./scripts/bootstrap_uv.sh` and do
  not proceed until `.venv/bin/python scripts/gpu_check.py` passes.
- If CUDA is visible but matmul fails with a kernel-image error, check the current
  PyTorch CUDA backend recommendation and rerun the bootstrap with, for example,
  `TORCH_BACKEND=cu129 ./scripts/bootstrap_uv.sh`.
- If CUDA is visible in WSL generally but not to one application, make sure that
  application is using the same `.venv` interpreter that passed
  `scripts/gpu_check.py`.
