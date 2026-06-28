# RUN_LOG

All evidence below was produced from the project-local uv environment:

```bash
./scripts/bootstrap_uv.sh
```

The bootstrap command creates/reuses `.venv`, exact-syncs dependencies with
`uv pip install --torch-backend cu128 --exact`, installs the Playwright Chromium
binary, and runs `scripts/gpu_check.py`.

## Environment setup

Command:

```bash
./scripts/bootstrap_uv.sh
```

Key output:

```text
Using CPython 3.12.2
Creating virtual environment at: .venv
Resolved 157 packages
PASS: CUDA tensor matmul ran on the GPU with real timing.
```

## Phase 0 - GPU verification gate

Command run by the bootstrap script:

```bash
.venv/bin/python scripts/gpu_check.py
```

Output:

```text
=== nvidia-smi ===
Sat Jun 27 15:36:46 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5060 Ti     On  |   00000000:01:00.0  On |                  N/A |
|  0%   48C    P8              7W /  180W |    3471MiB /  16311MiB |      1%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A             720      G   /Xwayland                             N/A      |
+-----------------------------------------------------------------------------------------+
=== torch ===
torch.__version__ = 2.7.1+cu128
torch.version.cuda = 12.8
torch.cuda.is_available() = True
torch.cuda.get_device_name() = NVIDIA GeForce RTX 5060 Ti
torch.cuda.get_device_capability() = (12, 0)
cuda matmul: 2048x2048 @ 2048x2048, dtype=torch.float16, best_ms=0.442, median_ms=0.459
PASS: CUDA tensor matmul ran on the GPU with real timing.
```

## Phase 1 - Real model demo

Command:

```bash
.venv/bin/python scripts/run_demo.py --device cuda --model-name pythia-70m --limit 8 --artifacts-dir artifacts
```

Key output:

```text
torch.__version__=2.7.1+cu128
torch.version.cuda=12.8
torch.cuda.is_available()=True
torch.cuda.get_device_capability()=(12, 0)
prompts_file=/home/home/p/g/n/learning/ml_research/self-ground/data/phase3_task_bank/pythia70m_negation_candidate_tasks.jsonl
loaded_tasks=8
first_task_id=sentiment_negation_84a9540d6745
hook=blocks.2.hook_resid_post
Loaded pretrained model pythia-70m into HookedTransformer
model_name=pythia-70m
first_parameter_device=cuda:0
activation_summary.blocks.2.hook_resid_post.shape=[8, 9, 512]
activation_summary.blocks.2.hook_resid_post.device=cuda:0
PatchResult target_mean=1.2690460681915283
PatchResult control_mean=-1.6967530250549316
ControlReport control_max=-3.2503719329833984
ControlReport ratio=2.5612718201911977
ControlReport flagged=true
artifact_attention_html=artifacts/attention_layer0_head0.html
artifact_attention_screenshot=artifacts/attention_layer0_head0.png (ok)
artifact_patch_bar=artifacts/patch_bar.html
artifact_token_deltas=artifacts/token_deltas.html
artifact_activation_heatmap=artifacts/activation_heatmap.html
sae_status=No matching public SAE wired for model 'pythia-70m' at 'blocks.2.hook_resid_post'. SAELens release 'pythia-70m-deduped-res-sm' is for pythia-70m-deduped, so the demo does not encode non-deduped activations with it.
```

Full structured outputs:

```text
artifacts/patch_result.json
artifacts/control_report.json
```

## Phase 2 - Real visualization artifacts

Artifacts generated from real GPU-computed tensors:

```text
artifacts/attention_layer0_head0.html
artifacts/attention_layer0_head0.png
artifacts/patch_bar.html
artifacts/token_deltas.html
artifacts/activation_heatmap.html
artifacts/patch_result.json
artifacts/control_report.json
```

CircuitsVis attention screenshot was inspected after generation. The first
attempt during development was blank because the wrapper passed a 2D head matrix
to `attention_heads`; the final artifact uses a one-head attention stack and
renders correctly.

SAE note: I found and loaded `pythia-70m-deduped-res-sm/blocks.2.hook_resid_post`
for the compatible `pythia-70m-deduped` model. I did not encode non-deduped
`pythia-70m` activations with that SAE.

Compatible SAE artifact command:

```bash
.venv/bin/python scripts/run_demo.py --device cuda --model-name pythia-70m-deduped --limit 8 --artifacts-dir artifacts/deduped_sae
```

Key output:

```text
model_name=pythia-70m-deduped
first_parameter_device=cuda:0
hook=blocks.2.hook_resid_post
PatchResult target_mean=-1.5195279121398926
ControlReport ratio=0.7920353496317956 flagged=false
sae_status=loaded pythia-70m-deduped-res-sm/blocks.2.hook_resid_post; exported feature_table.html and sae_activation_heatmap.html
```

Additional SAE artifacts:

```text
artifacts/deduped_sae/feature_table.html
artifacts/deduped_sae/sae_activation_heatmap.html
```

## Phase 3 - Streamlit explorer

Command:

```bash
.venv/bin/streamlit run apps/explorer.py --server.address 0.0.0.0 --server.port 8501
```

Streamlit reported:

```text
Local URL: http://localhost:8501
Network URL: http://172.24.75.206:8501
```

Curl acceptance command:

```bash
curl -sf http://localhost:8501 -o /dev/null -w "%{http_code}"
```

Output:

```text
200
```

## Phase 4 - Research-question UI check

Browser check opened the explorer, clicked `Run patch`, waited for the control
banner, and saved:

```text
artifacts/explorer_patch_controls.png
```

Observed visible banner:

```text
Control leakiness FLAGGED: ratio 2.561, control max -3.2504, target mean 1.2690
```

## QC

Commands:

```bash
.venv/bin/ruff check .
.venv/bin/python -m pytest
python3 -c "import mil; print('import mil ok')"
.venv/bin/python -c "import mil; print('import mil ok')"
```

Output:

```text
All checks passed!
19 passed in 0.79s
import mil ok
import mil ok
```

## Guided walkthrough redesign

Environment refresh:

```bash
./scripts/bootstrap_uv.sh
```

Key output:

```text
torch.__version__ = 2.7.1+cu128
torch.version.cuda = 12.8
torch.cuda.is_available() = True
torch.cuda.get_device_capability() = (12, 0)
cuda matmul: 2048x2048 @ 2048x2048, dtype=torch.float16, best_ms=0.449, median_ms=0.478
PASS: CUDA tensor matmul ran on the GPU with real timing.
```

Guided artifact build:

```bash
.venv/bin/python scripts/build_guided_demo_artifacts.py --device cuda --artifacts-dir artifacts/guided_demo
```

Key output:

```text
model_name=pythia-70m-deduped
hook=blocks.2.hook_resid_post
device=cuda
sae=pythia-70m-deduped-res-sm/blocks.2.hook_resid_post
torch.cuda.get_device_capability()=(12, 0)
Loaded pretrained model pythia-70m-deduped into HookedTransformer
first_parameter_device=cuda:0
attention=artifacts/guided_demo/attention_layer0_head0.html (ok)
feature_status=loaded pythia-70m-deduped-res-sm/blocks.2.hook_resid_post
manifest=artifacts/guided_demo/demo_manifest.json
```

Generated guided artifacts:

```text
artifacts/guided_demo/prompt_family.json
artifacts/guided_demo/tokenization.json
artifacts/guided_demo/behavior_logits.json
artifacts/guided_demo/logit_lens.json
artifacts/guided_demo/causal_heatmap.json
artifacts/guided_demo/causal_heatmap_target.html
artifacts/guided_demo/causal_heatmap_control.html
artifacts/guided_demo/causal_heatmap_difference.html
artifacts/guided_demo/patch_summary.json
artifacts/guided_demo/token_deltas.html
artifacts/guided_demo/feature_table.html
artifacts/guided_demo/feature_raster.html
artifacts/guided_demo/residual_trajectory.html
artifacts/guided_demo/attention_layer0_head0.html
artifacts/guided_demo/app_guided_smoke.png
```

Guided Streamlit smoke:

```bash
.venv/bin/streamlit run apps/explorer.py --server.address 0.0.0.0 --server.port 8501 -- --demo guided
curl -sf http://localhost:8501 -o /dev/null -w '%{http_code}'
```

Output:

```text
200
```

Browser smoke found the guided title, `Control ratio`, four summary metrics, and
ten rendered Plotly plots before saving:

```text
artifacts/guided_demo/app_guided_smoke.png
```

Final QC after the guided redesign:

```bash
.venv/bin/ruff check .
.venv/bin/python -m pytest
```

Output:

```text
All checks passed!
31 passed in 0.90s
```

## Dense guided dashboard styling

Grid view smoke:

```bash
.venv/bin/streamlit run apps/explorer.py --server.address 0.0.0.0 --server.port 8501 -- --demo guided --view grid
curl -sf http://localhost:8501 -o /dev/null -w '%{http_code}'
```

Output:

```text
200
```

Browser smoke found eight rendered Plotly plots, four compact metric cards, and
the SAE feature strip before saving:

```text
artifacts/guided_demo/app_guided_grid_tight.png
```

Scroll narrative smoke:

```bash
.venv/bin/streamlit run apps/explorer.py --server.address 0.0.0.0 --server.port 8502 -- --demo guided --view scroll
curl -sf http://localhost:8502 -o /dev/null -w '%{http_code}'
```

Output:

```text
200
```

Browser smoke found ten rendered Plotly plots and four native metric cards before
saving:

```text
artifacts/guided_demo/app_guided_scroll_compact.png
```

Final QC after the density pass:

```bash
.venv/bin/ruff check .
.venv/bin/python -m pytest
```

Output:

```text
All checks passed!
44 passed in 1.01s
```

## Scroll rendering fixes

Focused fixes for the guided scroll view:

- Step 3 feature-specificity charts now reserve enough left/bottom margin and rotate labels.
- The SAE feature raster artifact was rebuilt with readable prompt/token tick labels.
- Step 6 now prefers a dark Plotly attention heatmap artifact over the CircuitsVis iframe.

Artifact rebuild:

```bash
.venv/bin/python scripts/build_guided_demo_artifacts.py --device cuda --artifacts-dir artifacts/guided_demo
```

Key output:

```text
model_name=pythia-70m-deduped
hook=blocks.2.hook_resid_post
device=cuda
torch.cuda.get_device_capability()=(12, 0)
first_parameter_device=cuda:0
feature_status=loaded pythia-70m-deduped-res-sm/blocks.2.hook_resid_post
manifest=artifacts/guided_demo/demo_manifest.json
```

Scroll smoke:

```bash
.venv/bin/streamlit run apps/explorer.py --server.address 0.0.0.0 --server.port 8501 -- --demo guided --view scroll
curl -sf http://localhost:8501 -o /dev/null -w '%{http_code}'
```

Output:

```text
200
```

Evidence screenshots:

```text
artifacts/guided_demo/app_guided_scroll_top_fixed.png
artifacts/guided_demo/app_guided_step3_fixed.png
artifacts/guided_demo/app_guided_attention_fixed.png
artifacts/guided_demo/attention_layer0_head0_heatmap.html
```
