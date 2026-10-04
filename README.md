# LLM Compressor

Constraint-driven **structural** compression for Hugging Face language models.

Most "compression" tools leave a model the same size and multiply by zero. This one
physically deletes parts of the network — MLP neurons, attention heads, key/value
heads and whole transformer layers — and writes a genuinely smaller checkpoint that
`transformers` can reload. Where two layers are mathematically equivalent it fuses
them exactly; where they are not, it can distill the survivor from the original.
Nothing is ever faked: no masks, no zeroing, no random calibration tensors, and no
checkpoint is written unless it passes structural validation and reloads.

- Physically smaller tensors: `[6144, 4096]` becomes `[4352, 4096]`, not a mask.
- Real calibration only: dataset mode uses your tokenized text; dataset-free mode
  uses weight/structure analysis and never invents activations.
- Honest capabilities: every operation on a model reports `SUPPORTED`, `UNKNOWN`
  or `UNSUPPORTED` with a reason, decided by module-graph introspection rather
  than a hardcoded list of architectures.

---

## Contents

1. [Install](#install)
2. [Quick start](#quick-start)
3. [Usage](#usage)
4. [What the tool does](#what-the-tool-does)
5. [Methods it uses](#methods-it-uses)
6. [How it works](#how-it-works)
7. [Calibration data policy](#calibration-data-policy)
8. [Architecture support](#architecture-support)
9. [Output artifacts](#output-artifacts)
10. [Safety guarantees and limitations](#safety-guarantees-and-limitations)
11. [Development](#development)

---

## Install

```bash
pip install -e .
```

Requires Python 3.10+ and installs `torch`, `transformers`, `accelerate`, `typer`,
`rich` and `safetensors`. Installing the package gives you the `llm-compress`
command; `python -m llm_compressor` works identically without it.

```bash
llm-compress --help
python -m llm_compressor --help
```

## Quick start

Inspect a model's architecture and see what can and cannot be compressed:

```bash
llm-compress inspect gpt2
llm-compress inspect facebook/opt-125m --json
```

Remove 30% of all MLP neurons from a model, keeping everything else intact:

```bash
llm-compress compress gpt2 --remove-percent 30 --output ./gpt2-pruned
```

Remove an exact number of neurons instead of a percentage:

```bash
llm-compress compress facebook/opt-125m --remove-count 12000 --output ./opt-pruned
```

Shrink depth as well, targeting a 2-layer student, and fuse the layers that are
provably equivalent:

```bash
llm-compress compress mistralai/Mistral-7B-v0.1 \
    --remove-percent 25 \
    --target-student-layers 2 \
    --fuse exact \
    --output ./mistral-small
```

Run the guided wizard if you would rather be asked each question:

```bash
llm-compress compress gpt2 --interactive
```

Load the result like any other model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./gpt2-pruned")
tok = AutoTokenizer.from_pretrained("./gpt2-pruned")
print(model.config.n_embd, model.config.n_layer)  # smaller than the original
```

## Usage

### `inspect`

Loads a model, builds its architecture graph, and prints (or emits as JSON) the
block stacks it found, the neuron/MLP definition, embeddings, LM head, tied
weight groups, layer inventory, protected components and the per-operation
capability table.

| Option | Meaning |
| --- | --- |
| `--device`, `-d` | Device to load on (default `cpu`). |
| `--dtype` | `auto`, `float32`, `float16` or `bfloat16`. |
| `--trust-remote-code` | Allow custom model code. |
| `--revision` | Load a specific revision. |
| `--json` | Emit machine-readable JSON instead of the Rich report. |

`inspect` never mutates the model.

### `compress`

Compresses a model and writes a new checkpoint plus a report.

**Data**

| Option | Meaning |
| --- | --- |
| `--dataset` | Dataset mode: a HF dataset id, or a local `.txt`/`.jsonl` file of real text. |
| `--dataset-mode` | `fast`, `balanced`, `accurate` or `custom` (sample count and sequence length presets). |
| `--dataset-split`, `--dataset-field` | Which split and which column holds the text. |
| `--num-samples`, `--seq-len` | Explicit example count and token length (used with `--dataset-mode custom`). |
| `--dataset-free` | Dataset-free mode: only data-free methods; activations are never invented. |

**Neuron budget** (one MLP intermediate channel = one neuron)

| Option | Meaning |
| --- | --- |
| `--remove-percent` | Remove this percent of all MLP neurons. |
| `--remove-count` | Remove exactly this many MLP neurons. |
| `--allocation` | How the global budget is spread: `global`, `uniform` or `hybrid`. |
| `--scoring` | `weight_l1`, `weight_l2`, `weight_combined`, `activation`, `activation_weighted`. |

**Depth, fusion and attention**

| Option | Meaning |
| --- | --- |
| `--layer-mode` | `search` (beam-search which layers to drop) or `none`. |
| `--target-student-layers` | Target student depth; must be smaller than the teacher depth. |
| `--fuse` | `none`, `exact` (provably equivalent) or `learned` (fusion + distillation). |
| `--distill-steps`, `--distill-lr`, `--distill-losses` | Distillation schedule and loss set (`auto`, `all`, `kl`, `ce`, `hidden`, `kl+hidden`). |
| `--beam-width` | Beam width for the joint layer + neuron search. |
| `--remove-attention-percent` | Remove this percent of query heads, always deleting whole KV groups. |

**Protection and validation**

| Option | Meaning |
| --- | --- |
| `--protect` | Extra glob patterns to protect from rewriting. |
| `--unprotect` | Category names or globs to release from protection. |
| `--no-validate` | Skip the validation pass (not recommended). |
| `--device`, `--dtype`, `--trust-remote-code`, `--revision` | Model loading controls. |
| `--interactive`, `-i` | Run the interactive wizard. |

Protection defaults cover `embeddings`, `lm_head`, `norm`, `rotary`, `moe`,
`adapter`, `tied_weights`, `auxiliary`, `vision` and `audio`. Protected
components are never pruned or fused.

### Python API

```python
from llm_compressor import compress_model

result = compress_model(
    "gpt2",
    output="./gpt2-pruned",
    remove_percent=30.0,
    dataset_free=True,
)
print(result["report"]["summary"])
```

## What the tool does

1. **Discovers the architecture** of whatever model you point it at, generically:
   it locates the transformer block stacks, the MLP/FFN projections, the
   attention projections and their head geometry, the normalisation layers, the
   embedding and the LM head.
2. **Scores units** — MLP intermediate channels, attention heads, KV heads — by
   weight magnitude and/or captured activations.
3. **Allocates a budget** across the model so a total percentage or exact count
   is met, either globally, uniformly or by a hybrid rule.
4. **Physically deletes** the lowest-scoring units, slicing every dependent
   tensor so the shapes genuinely shrink.
5. **Shrinks depth** when asked, by choosing which layers to drop (and which to
   fuse) with a joint layer + neuron beam search.
6. **Fuses equivalent layers** exactly, or distills a fused survivor when the
   merge is not provable.
7. **Validates** the mutated model structurally (loadable skeleton, consistent
   shapes, live manifest entries) and only then writes the checkpoint, a
   manifest and a human-readable report.

## Methods it uses

### Physical structural deletion

Deletion is real tensor surgery, not masking. Removing MLP neuron `j` of a
layer means slicing row `j` out of the up-projection, column `j` out of the
down-projection, and any matching bias entries; the parameter shape changes.
Attention heads are removed as whole units, and KV heads are only ever removed
together with every query head in their group, so the attention arithmetic stays
valid.

### Layer deletion

Whole blocks are dropped from the module list, residual paths are renumbered,
`layer_idx` values are re-indexed, dotted parameter paths are remapped, and the
model's depth configuration is synchronised. Two situations are detected and
handled rather than papered over:

- **Shared depth config.** Some architectures describe two stacks (for example a
  seq2seq encoder and decoder) whose depths live in different config keys; a
  naive rewrite of one key can silently resize the other stack. The tool refuses
  to touch a stack whose depth key is shared, and re-verifies every stack's
  config depth against its real module count after mutation.
- **Unowned parameters.** A block may own a parameter that no other block
  provides (a shared relative-attention bias is the classic case). Deleting such
  a block cannot be expressed in config and would leave an unloadable model, so
  those indices are protected up front and reported as `UNSUPPORTED`, and a safe
  layer is chosen instead.

### Exact affine fusion

When two adjacent projections compose as a chain of affine maps, the tool
proves the composition symbolically and replaces the pair with a single fused
weight: `W = W2 @ W1`, `b = W2 @ b1 + b2` (with correct handling of
transposed `Conv1D`-style weights and biasless layers). The fused parameters are
written back physically, so the result is smaller and numerically equivalent
within floating-point tolerance.

### Learned fusion and distillation

When an exact merge is not provable, the survivor of a fused group can be
trained to match the original model. Distillation uses modular losses — logit KL,
next-token cross-entropy and hidden-state MSE — selected automatically from what
the architecture exposes, run for a configurable number of steps and learning
rate, using the same real calibration batches as the rest of the run.

### Joint layer + neuron search

Depth and width are not chosen independently: a beam search over regions marked
`KEEP`, `DELETE`, `EXACT_FUSE` and `DISTILL_FUSE` explores joint layer/neuron
decisions against the budget, so the width budget is met after the depth choice
rather than in isolation.

### Neuron scoring

Units are ranked by `weight_l1`, `weight_l2` or `weight_combined`. In dataset
mode, `activation` and `activation_weighted` additionally hook real forward
passes and score the actual signal flowing through each channel.

### Protection and validation

A protection policy decides, per module and category, whether a rewrite is even
allowed; defaults keep embeddings, the LM head, norms, rotary embeddings, MoE
routers, adapters, tied weights and auxiliary modules intact. After mutation, a
validation pass checks that the skeleton can be rebuilt from config, that the
manifest's entries all exist in the model, and that shapes agree; the checkpoint
is only written if it passes.

## How it works

The pipeline is a sequence of stages, and each stage is allowed to say "no":

1. **Load** the model and tokenizer with an eager attention implementation so
   attention internals are addressable.
2. **Introspect** the module graph: find block stacks, classify each projection
   by name tokens *and* tensor shape, and build a layout-normalised projection
   abstraction that hides whether the underlying layer is `nn.Linear` or a
   transposed `Conv1D`. Every capability is then tri-state, with a reason
   attached when it is not supported.
3. **Prepare data** (dataset mode) or confirm a data-free run (dataset-free
   mode). Dataset mode tokenises once and caches the encoded examples so
   scoring, search and distillation all replay the exact same batch layout.
4. **Score** candidate units, allocate the budget, and record the protected set.
5. **Search** if depth is being reduced; otherwise plan width-only deletion.
6. **Mutate** physically: slice neurons and heads, drop and re-index layers,
   fuse chains. Anything that cannot be done safely is skipped and reported as
   an `unsupported` note rather than forced.
7. **Validate** the mutated model structurally and numerically.
8. **Save** the compressed checkpoint, the compression manifest and the report.

Because discovery is generic — module graph plus shape reasoning, not a
per-architecture recipe table — new architectures work as far as their structure
is safely understood, and everything beyond that is reported honestly instead of
being attempted blindly.

## Calibration data policy

- **Dataset mode** (`--dataset`) uses real tokenised examples from a Hugging
  Face dataset or a local text/JSONL file. The encoded examples are cached by a
  key derived from the tokenizer, sequence length, packing mode and texts, so a
  re-run replays identical batches. If the supplied text cannot produce usable
  examples, the run fails with a clear error.
- **Dataset-free mode** (`--dataset-free`) uses only weight and structural
  analysis. It never fabricates activations, and activation-based scoring is
  reported as unavailable rather than approximated with random tensors. Random
  data is used nowhere as a stand-in for calibration — only as a deterministic
  fixed-token input for latency and equivalence checks, and that is labelled as
  such.

## Architecture support

Capabilities are decided at runtime and reported per model. The two clearest
cases worth knowing about in advance:

- **Fused QKV attention (e.g. GPT-2).** When query, key and value live in one
  packed projection, the tool can still prune whole key/value groups only if the
  packing is unambiguous. Where it is not, attention-head and KV-head pruning
  are reported `UNKNOWN` — inspection and compression still succeed, and the
  width budget is met through MLP neurons instead.
- **Shared relative-attention bias in seq2seq stacks (e.g. T5).** The first
  block of each stack owns the bias shared by every block in that stack, so
  deleting it cannot be rebuilt from config. That index is never deleted; a safe
  layer is chosen instead, and the reason appears in the report's `unsupported`
  notes.

In general: if a structure cannot be understood well enough to be rewritten and
reloaded correctly, the tool reports `UNSUPPORTED`/`UNKNOWN` and leaves it alone.
It never produces a corrupted checkpoint in order to claim support.

## Output artifacts

Compressing to `--output <dir>` writes:

- the compressed model weights and config, directly loadable with
  `AutoModelForCausalLM.from_pretrained` (or the matching auto class);
- the tokenizer, when one is available;
- `compression_manifest.json` — what was removed, what was fused, which regions
  were protected or skipped, and the budgets that were requested versus met;
- a human-readable compression report (also returned by the Python API).

## Safety guarantees and limitations

Guaranteed:

- deletion is physical, so size and parameter counts actually drop;
- every written checkpoint is structurally validated and reloads;
- protected components are never pruned or fused;
- unsafe operations degrade to a reported `UNSUPPORTED` note instead of a crash
  or a silently wrong model.

Limitations:

- compression quality depends on available signal: weight-based scoring needs no
  data, but `activation` scoring requires dataset mode;
- exact fusion requires a provable composition; otherwise only learned fusion or
  plain deletion applies;
- architectures whose attention packing or weight sharing cannot be interpreted
  unambiguously report `UNKNOWN` for the affected operations and compress through
  the operations that are safe;
- very aggressive budgets can degrade quality; always compare the original and
  compressed model with your own evaluation before shipping.

## Development

```bash
pip install -e .
python -m pytest tests -q
```

The test suite covers unit logic (scoring, budgets, projection slicing,
protection), model-level structural tests across several model families, and
end-to-end compress → save → reload checks. Tests use small real checkpoints;
no random calibration is used anywhere in the pipeline.

```
llm_compressor/
  model/        introspection, architecture graph, projection abstraction, protection
  scoring/      neuron and head scoring
  budget.py     budget allocation
  search/       joint layer + neuron beam search
  pruning/      physical deletion of neurons, heads, layers
  fusion/       exact affine fusion
  distill.py    distillation losses and schedule
  data/         dataset mode (cached tokenisation) and dataset-free inputs
  evaluation/   equivalence / agreement metrics
  checkpoint.py save, validate and reload compressed checkpoints
  output/       reports
  cli/          typer commands (inspect, compress)
  pipeline.py   orchestration
```
