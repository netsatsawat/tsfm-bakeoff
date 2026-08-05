# One environment per model family

## Why this is necessary, not fussy

Python resolves exactly one version of each package per environment. Two of the model
packages here demand mutually incompatible numeric stacks, measured with
`pip install --dry-run` before touching anything:

| package | forces | extra damage |
|---|---|---|
| `toto-ts==0.2.0` (Toto, checkpoint Toto-Open-Base-1.0) | torch 2.11.0 to **2.7.0**, numpy 2.2.6 to **1.26.4**, pandas 2.3.3 to **2.2.3** | hard-pins `jupyter==1.1.1` |
| `uni2ts==2.0.0` (Moirai-2) | torch to **2.4.1**, numpy to **1.26.4**, pandas to **2.1.4** | pulls jax, lightning, tensorboard |

The numpy downgrade is the one that actually breaks things. NumPy 2.0 changed the
C ABI, so any extension compiled against 2.x fails to load against 1.26. This is
binary incompatibility, not a version-label disagreement; no resolver flag gets around
it.

So: **one virtualenv per family, forecasts handed off on disk.** Each runner writes
under `forecasts/`; `score.py` reads those and computes every metric while importing
no model code at all. That also means all families are scored under identical rules,
which is harder to guarantee when each model is scored inside its own process.

This costs some disk and buys a field that is otherwise impossible to assemble.

## Setup

```bash
bash envs/setup.sh          # builds .venv-core, .venv-toto, .venv-moirai
```

Or individually; install the torch build matching your accelerator first:

```bash
python3 -m venv .venv-core   && .venv-core/bin/pip   install -r envs/core.txt
python3 -m venv .venv-toto   && .venv-toto/bin/pip   install -r envs/toto.txt
python3 -m venv .venv-moirai && .venv-moirai/bin/pip install -r envs/moirai.txt
```

## Running

Run `run_core.py` **first**: it writes the `forecasts/_truth/` files that carry the
origins, contexts and MASE denominators every other runner and the scorer depend on.
After that, the two satellite runners are independent of each other and can run on
different days or different machines, which is the point.

```bash
.venv-core/bin/python   runners/run_core.py             # TimesFM, Chronos-2/Bolt, classical, naive
.venv-toto/bin/python   runners/run_toto.py             # Toto (Toto-Open-Base-1.0)
.venv-moirai/bin/python runners/run_moirai.py           # Moirai-2

python3 score.py                                        # stdlib only; no venv needed
```

`score.py` deliberately requires no virtualenv. If scoring needed one of these
environments, the metric definitions would live next to a model and drift.

## Disk and memory

Each venv carries its own torch (roughly 2 to 3 GB). Budget ~10 GB for all three; the
same figure appears in the README quick start.

Peak RSS is per-runner, not cumulative, since they run one at a time, but a single
foundation model still wants several GB. On a 24 GB machine, close other large
processes first; a local LLM server will use enough to make these runs swap.
