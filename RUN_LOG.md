# RUN_LOG

## Phase 0 - GPU verification gate

Command:

```bash
.venv/bin/python scripts/gpu_check.py
```

Output:

```text
=== nvidia-smi ===
Sat Jun 27 15:06:37 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.01              Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5060 Ti     On  |   00000000:01:00.0  On |                  N/A |
|  0%   45C    P8              9W /  180W |    3405MiB /  16311MiB |      4%      Default |
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
torch.__version__ = 2.11.0+cu128
torch.version.cuda = 12.8
torch.cuda.is_available() = True
torch.cuda.get_device_name() = NVIDIA GeForce RTX 5060 Ti
torch.cuda.get_device_capability() = (12, 0)
cuda matmul: 2048x2048 @ 2048x2048, dtype=torch.float16, best_ms=0.387, median_ms=0.461
PASS: CUDA tensor matmul ran on the GPU with real timing.
```

## Phase 1 - Real model demo

Command:

```bash
/home/home/p/g/n/learning/ml_research/self-ground/.venv/bin/python scripts/run_demo.py --device cuda --model-name pythia-70m --limit 8 --artifacts-dir artifacts
```

Output:

```text
torch.__version__=2.12.1+cu130
torch.version.cuda=13.0
torch.cuda.is_available()=True
torch.cuda.get_device_capability()=(12, 0)
prompts_file=/home/home/p/g/n/learning/ml_research/self-ground/data/phase3_task_bank/pythia70m_negation_candidate_tasks.jsonl
loaded_tasks=8
first_task_id=sentiment_negation_84a9540d6745
hook=blocks.2.hook_resid_post
Loaded pretrained model pythia-70m into HookedTransformer
model_name=pythia-70m
first_parameter_device=cuda:0
activation_summary={
  "blocks.0.attn.hook_pattern": {
    "device": "cuda:0",
    "dtype": "torch.float32",
    "mean": 0.1111111119389534,
    "shape": [
      8,
      8,
      9,
      9
    ],
    "std": 0.21376681327819824
  },
  "blocks.2.hook_resid_post": {
    "device": "cuda:0",
    "dtype": "torch.float32",
    "mean": 6.622738357719982e-09,
    "shape": [
      8,
      9,
      512
    ],
    "std": 2.9415462017059326
  }
}
PatchResult={
  "control_deltas": [
    -1.505732536315918,
    -2.062851905822754,
    0.33937740325927734,
    -3.2503862380981445,
    -1.536585807800293,
    -2.513181686401367,
    -0.9019899368286133,
    -2.1425294876098633
  ],
  "control_mean": -1.6967350244522095,
  "hook": "blocks.2.hook_resid_post",
  "metric": "logit_diff_delta",
  "mode": "ablation",
  "target_deltas": [
    1.116805076599121,
    1.299546241760254,
    -0.3820314407348633,
    2.335148811340332,
    1.570969581604004,
    2.3678693771362305,
    0.7125329971313477,
    1.1319799423217773
  ],
  "target_mean": 1.2691025733947754,
  "token_deltas": [
    {"delta": 59.12715148925781, "direction": "positive", "token": "\ufffd", "token_id": 101},
    {"delta": 58.439720153808594, "direction": "positive", "token": "\ufffd", "token_id": 97},
    {"delta": 57.92962646484375, "direction": "positive", "token": "\ufffd\ufffd", "token_id": 44140},
    {"delta": 57.21588897705078, "direction": "positive", "token": "\ufffd\ufffd", "token_id": 8318},
    {"delta": 56.96025848388672, "direction": "positive", "token": "\ufffd", "token_id": 233},
    {"delta": 56.58726501464844, "direction": "positive", "token": "\ufffd", "token_id": 100},
    {"delta": 55.652610778808594, "direction": "positive", "token": "\ufffd", "token_id": 234},
    {"delta": 55.347694396972656, "direction": "positive", "token": "\ufffd", "token_id": 114},
    {"delta": 55.30510711669922, "direction": "positive", "token": "\ufffd", "token_id": 99},
    {"delta": 55.08271789550781, "direction": "positive", "token": "\ufffd", "token_id": 216},
    {"delta": -21.18157386779785, "direction": "negative", "token": ":`", "token_id": 30337},
    {"delta": -20.09361457824707, "direction": "negative", "token": " his", "token_id": 521},
    {"delta": -19.074026107788086, "direction": "negative", "token": "'\">", "token_id": 46768},
    {"delta": -19.015546798706055, "direction": "negative", "token": ":&", "token_id": 44662},
    {"delta": -18.26312828063965, "direction": "negative", "token": "\n\u00a0\u00a0", "token_id": 39318},
    {"delta": -18.250436782836914, "direction": "negative", "token": "iei", "token_id": 44710},
    {"delta": -18.151241302490234, "direction": "negative", "token": ":*", "token_id": 27506},
    {"delta": -17.970794677734375, "direction": "negative", "token": " her", "token_id": 617},
    {"delta": -17.93769073486328, "direction": "negative", "token": "---|---", "token_id": 20782},
    {"delta": -17.863788604736328, "direction": "negative", "token": "noreply", "token_id": 46352}
  ]
}
ControlReport={
  "control_max": -3.2503862380981445,
  "flagged": true,
  "gap": -1.9812836647033691,
  "ratio": 2.5611690546049015,
  "target_delta": 1.2691025733947754,
  "threshold": 0.8
}
artifact_attention_html=artifacts/attention_layer0_head0.html
artifact_attention_screenshot=artifacts/attention_layer0_head0.png (ok)
artifact_patch_bar=artifacts/patch_bar.html
artifact_token_deltas=artifacts/token_deltas.html
artifact_activation_heatmap=artifacts/activation_heatmap.html
sae_status=No matching public SAE wired for model 'pythia-70m' at 'blocks.2.hook_resid_post'. SAELens release 'pythia-70m-deduped-res-sm' is for pythia-70m-deduped, so the demo does not encode non-deduped activations with it.
```

## Phase 2 - Real visualization artifacts

Artifacts generated from the Phase 1 run:

```text
artifacts/attention_layer0_head0.html
artifacts/attention_layer0_head0.png
artifacts/patch_bar.html
artifacts/token_deltas.html
artifacts/activation_heatmap.html
artifacts/patch_result.json
artifacts/control_report.json
```

CircuitsVis attention screenshot was inspected after generation. The first attempt was blank because
the wrapper passed a 2D head matrix to `attention_heads`; the final committed artifact uses a one-head
attention stack and renders correctly.

SAE note: I found and loaded `pythia-70m-deduped-res-sm/blocks.2.hook_resid_post` for the compatible
`pythia-70m-deduped` model. I did not use that SAE on the non-deduped `pythia-70m` run because it is
not the same model family.

Compatible SAE artifact command:

```bash
/home/home/p/g/n/learning/ml_research/self-ground/.venv/bin/python scripts/run_demo.py --device cuda --model-name pythia-70m-deduped --limit 8 --artifacts-dir artifacts/deduped_sae
```

Key output:

```text
model_name=pythia-70m-deduped
first_parameter_device=cuda:0
hook=blocks.2.hook_resid_post
PatchResult target_mean=-1.5195348262786865
ControlReport ratio=0.7918378144381852 flagged=false
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
/home/home/p/g/n/learning/ml_research/self-ground/.venv/bin/streamlit run apps/explorer.py --server.address 0.0.0.0 --server.port 8501
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

Browser check command opened the explorer with Playwright, clicked `Run patch`, waited for the control
banner, and saved `artifacts/explorer_patch_controls.png`.

Observed visible banner:

```text
Control leakiness FLAGGED: ratio 2.561, control max -3.2504, target mean 1.2691
```

## QC

Commands:

```bash
/home/home/p/g/n/learning/ml_research/self-ground/.venv/bin/ruff check .
/home/home/p/g/n/learning/ml_research/self-ground/.venv/bin/python -m pytest
python3 -c "import mil; print('import mil ok')"
/home/home/p/g/n/learning/ml_research/self-ground/.venv/bin/python -c "import mil; print('import mil ok')"
```

Output:

```text
All checks passed!
19 passed in 0.72s
import mil ok
import mil ok
```
