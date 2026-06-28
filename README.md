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

Requirement: `uv` must be installed and on `PATH`. Do not use the system Python
installer on Ubuntu 24; it is externally managed and will reject global package
installs.

```bash
uv --version
```

Create a dedicated `.venv`, install the model/app/viz/SAE/evidence dependencies,
install Playwright's browser binary, and run the GPU kernel gate:

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

The `--exact` flag is intentional: this is a dedicated environment, and exact
sync prevents stale packages from surviving. In particular, this path does not
install `torchaudio`; a mismatched leftover `torchaudio` wheel can break
`transformers` imports.

The documented backend is `cu128` because it passed on the tested Blackwell
RTX 5060 Ti (`sm_120`). This may change as PyTorch ships newer stable sm_120
support. Check https://pytorch.org/get-started/locally/ for the current CUDA
recommendation before changing `TORCH_BACKEND`.

Run the real demo:

```bash
.venv/bin/python scripts/run_demo.py --device cuda --model-name pythia-70m --limit 8 --artifacts-dir artifacts
```

Run the explorer bound to all interfaces:

```bash
.venv/bin/streamlit run apps/explorer.py --server.address 0.0.0.0 --server.port 8501
```

From the Windows browser, open:

```text
http://localhost:8501
```

If localhost forwarding is not working for this WSL version, get the WSL VM IP:

```bash
hostname -I
```

Then open `http://<wsl-ip>:8501`.

Troubleshooting:

- If you see `externally-managed-environment`, you ran the wrong installer path.
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
