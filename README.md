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

Install a Blackwell-compatible PyTorch build from a CUDA wheel index, not the
default pip index. At the time this was tested, the CUDA 12.8 index worked:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

This may change as PyTorch ships newer stable sm_120 support. Check
https://pytorch.org/get-started/locally/ for the current CUDA 12.x recommendation
before trusting a hardcoded wheel index long-term.

Install the app/viz/model dependencies in the environment you will run from:

```bash
pip install -e ".[app,viz,sae]" transformer-lens
```

Run the GPU check again after installing these packages. If the resolver replaced
the CUDA torch build, reinstall torch from the CUDA wheel index and rerun the
check before starting the explorer.

Verify CUDA kernels before loading models:

```bash
python scripts/gpu_check.py
```

Run the explorer bound to all interfaces:

```bash
streamlit run apps/explorer.py --server.address 0.0.0.0 --server.port 8501
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

- If `nvidia-smi` works but `torch.cuda.is_available()` is false, check whether
  `torch.__version__` ends in `+cpu` or `torch.version.cuda` is `None`. That is a
  CPU-only torch install; reinstall from the CUDA wheel index above.
- If CUDA is visible in WSL generally but not to one application, make sure that
  application is using the same virtualenv/interpreter that passed
  `scripts/gpu_check.py`.
