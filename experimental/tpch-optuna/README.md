# TPC-H Optuna tuning (portable)

Tunes SiriusDB configuration to minimize total TPC-H time (sum of cold+warm run
times across all 22 queries) using Optuna's TPE sampler. The search space is
**hardware-relative**, so the same `tune.py` runs unchanged on wildly different
machines (laptop GPU → multi-GPU server) and scales itself to each.

## How a trial works

Per trial the driver samples a config, writes a per-trial `sirius.yaml` (pointed
at via `SIRIUS_CONFIG_FILE`), runs all 22 queries (1 cold + N-1 warm) in one
`build/release/duckdb` session, and sums every `Run Time (s): real` = the
objective. Crash / query error / timeout → penalty (relative to best-so-far) so
the sampler avoids that region instead of the study aborting. Per-trial timeout
is `--timeout-mult` × best-so-far.

## Portability — how it scales to any machine

At startup `detect_hardware()` reads GPU VRAM (`nvidia-smi`), GPU count, system
RAM, and CPU cores. The search space is expressed relative to those:

- **Byte-size knobs** (`scan_task_batch_size`, `hash_partition_bytes`,
  `concat_batch_bytes`, `sort_sample_bytes`, `max_build_hash_table_bytes`) are
  sampled as a **fraction of GPU VRAM** (log-uniform), so the same range means
  ~48 MB–14 GB on a 96 GB GPU and ~8 MB–2.4 GB on a 16 GB GPU.
- **Thread counts** (`pipeline`, `task_creator`, `scan_manager`, `downgrade`)
  scale to **CPU core count**.
- **Fraction knobs** (`gpu_usage_limit_fraction`, `host_capacity_fraction`, GPU
  spill `trigger`/`stop`, `max_sort_partition_memory_fraction`) are
  hardware-independent by construction.
- **Booleans / ratios** (`enable_prefetch_cache`, `uring_n_reactors`,
  dynamic-filter, mark-join) are kept in the search — they may matter on one
  machine's disk/GPU and not another's, so we don't fix them.

The optuna parameters recorded are the *fractions* (portable); resolved bytes go
into the yaml. `topology.num_gpus` is set from detection. Each machine gets its
own study — the name defaults to `tpch_sf<SF>_<hostname>` — so results from
different boxes never collide and can be compared later.

## Prerequisites (per machine)

1. Release build: `pixi run make` (fresh worktree first needs
   `git submodule update --init --recursive`).
2. TPC-H parquet at the target SF: `cd test/tpch_performance &&
   pixi run bash generate_tpch_data.sh <SF>` → `test_datasets/tpch_parquet_sf<SF>`.
3. `pixi add --pypi optuna optuna-dashboard` (already in this repo's pixi env).

## Run

```bash
pixi run python experimental/tpch-optuna/tune.py --sf 300 --n-trials 300
```

Detached (survives SSH disconnect):
```bash
setsid nohup pixi run python experimental/tpch-optuna/tune.py \
  --sf 300 --n-trials 300 --iterations 2 \
  > experimental/tpch-optuna/runs/study.log 2>&1 < /dev/null & disown
```

Useful flags: `--study-name` (override per-machine default), `--spill-dir`
(disk-spill dir for downgrades; default `<workdir>/disk_spill` — point at a fast
NVMe), `--queries 1,3,6-10`, `--iterations`, `--timeout-mult`.

Study DB: `experimental/tpch-optuna/runs/<study-name>.db` (SQLite, resumable —
re-run the same command to continue). Trial 0 is the in-source default config
(baseline), expressed in this machine's fractions.

## Moving to another machine

Copy the repo (or the `experimental/tpch-optuna/` dir), build Sirius and
generate the dataset there, then run the same command. It auto-detects that
box's VRAM/RAM/cores and creates a new `tpch_sf<SF>_<newhostname>` study — no
edits needed. Compare machines by loading each study's `best_value` / best
config from its own DB.

## Watch

```bash
pixi run optuna-dashboard sqlite:///experimental/tpch-optuna/runs/<study-name>.db
# then: ssh -L 8080:localhost:8080 <box>  and open http://localhost:8080
```
