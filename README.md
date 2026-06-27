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
