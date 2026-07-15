#!/usr/bin/env python3
"""Optuna tuning of SiriusDB configuration for the TPC-H benchmark.

Each trial samples a configuration, writes a per-trial sirius.yaml (memory
sizing, executor threads, operator_params) pointed at via SIRIUS_CONFIG_FILE,
runs all 22 TPC-H queries (1 cold + N-1 warm iterations each) in a single
build/release/duckdb session, parses every `Run Time (s): real` line, and
returns the sum (cold+warm across all queries) as the objective to minimize.

The search space is HARDWARE-RELATIVE so the same driver is portable across
wildly different machines: byte-size knobs are sampled as fractions of the
detected GPU VRAM, and thread counts scale to the detected CPU core count.
Fraction-based knobs (memory limits, spill thresholds) are machine-independent
by construction. This means a laptop GPU and an H100 node each explore a range
appropriate to that box instead of a fixed absolute range that would OOM the
small one and under-explore the large one.

A crash / query error / timeout yields a penalty so the TPE sampler learns to
avoid that region instead of the study aborting.

Usage:
  pixi run python experimental/tpch-optuna/tune.py --sf 300 --n-trials 300

Each machine gets its own study (name includes the hostname), so results from
different boxes don't collide and can be compared later.

Resume: re-run the same command; the SQLite study picks up where it left off.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import optuna
from sqlalchemy import event

# --------------------------------------------------------------------------
# Paths / constants
# --------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DUCKDB = PROJECT_DIR / "build" / "release" / "duckdb"
DEFAULT_QUERY_DIR = PROJECT_DIR / "test" / "tpch_performance" / "tpch_queries" / "orig"
TPCH_TABLES = [
    "customer", "lineitem", "nation", "orders",
    "part", "partsupp", "region", "supplier",
]

MiB = 1024 * 1024
GiB = 1024 * 1024 * 1024
SYS_RAM_BYTES = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


# --------------------------------------------------------------------------
# Hardware detection — the search space scales to whatever machine we run on.
# --------------------------------------------------------------------------
def detect_hardware() -> dict:
    """Detect GPU VRAM (per device), GPU count, system RAM and CPU cores so the
    sampler can express byte-size / thread knobs relative to this machine."""
    vram_bytes = 0
    num_gpus = 1
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip().splitlines()
        mibs = [int(x) for x in out if x.strip()]
        if mibs:
            num_gpus = len(mibs)
            vram_bytes = min(mibs) * MiB  # size to the smallest GPU (safe on mixed rigs)
    except Exception:
        pass
    if vram_bytes == 0:
        # Fallback so the driver still runs (e.g. detection blocked); assume 16 GB.
        vram_bytes = 16 * GiB
    hw = {
        "vram_bytes": vram_bytes,
        "num_gpus": num_gpus,
        "ram_bytes": SYS_RAM_BYTES,
        "cores": os.cpu_count() or 8,
    }
    return hw

RUN_TIME_RE = re.compile(r"Run Time \(s\): real (\d+\.\d+)")
# Error markers that mean the run is invalid (not just the word "error" in data).
ERROR_RE = re.compile(
    r"(^Error:|Invalid Error|IO Error|Catalog Error|Parser Error|Binder Error|"
    r"Out of Memory|std::bad_alloc|CUDA|RMM error|terminate called|Segmentation)",
    re.MULTILINE,
)


# --------------------------------------------------------------------------
# Search space
# --------------------------------------------------------------------------
def sample_params(trial: optuna.Trial, hw: dict) -> dict:
    """Sample one configuration, scaled to this machine's hardware.

    Byte-size knobs are sampled as a fraction of GPU VRAM and thread counts as a
    function of CPU cores, so the same ranges are meaningful on any box. The
    optuna parameters recorded are the *fractions* (portable); the resolved
    absolute bytes are computed here for the yaml writer."""
    vram = hw["vram_bytes"]
    cores = hw["cores"]
    p = {}

    def vram_bytes(name: str, lo: float, hi: float) -> int:
        # fraction of VRAM, log-uniform — same relative range on every GPU
        return int(trial.suggest_float(name, lo, hi, log=True) * vram)

    # --- memory limits (fractions — hardware-independent) ---
    p["gpu_usage_limit_fraction"] = trial.suggest_float("gpu_usage_limit_fraction", 0.4, 0.95)
    p["host_capacity_fraction"] = trial.suggest_float("host_capacity_fraction", 0.1, 0.8)

    # --- executor threads (scaled to core count) ---
    p["pipeline_num_threads"] = trial.suggest_int("pipeline_num_threads", 2, max(2, cores))
    p["task_creator_num_threads"] = trial.suggest_int(
        "task_creator_num_threads", 1, max(2, cores // 2))
    p["scan_manager_num_threads"] = trial.suggest_int(
        "scan_manager_num_threads", 2, max(2, cores))
    p["downgrade_num_threads"] = trial.suggest_int(
        "downgrade_num_threads", 1, max(2, cores // 4))

    # --- operator_params byte sizes (fraction of VRAM) ---
    p["scan_task_batch_size"] = vram_bytes("scan_task_batch_vram_frac", 0.0005, 0.15)
    p["hash_partition_bytes"] = vram_bytes("hash_partition_vram_frac", 0.0005, 0.15)
    p["concat_batch_bytes"] = vram_bytes("concat_batch_vram_frac", 0.0005, 0.15)
    p["sort_sample_bytes"] = vram_bytes("sort_sample_vram_frac", 0.0005, 0.08)
    p["max_build_hash_table_bytes"] = vram_bytes("max_build_hash_vram_frac", 0.0005, 0.20)
    p["max_sort_partition_memory_fraction"] = trial.suggest_float(
        "max_sort_partition_memory_fraction", 0.1, 0.9)

    # --- join / dynamic-filter strategy ---
    if trial.suggest_categorical("enable_mark_join_switch", [True, False]):
        p["mark_join_build_switch_ratio"] = trial.suggest_float(
            "mark_join_build_switch_ratio", 2.0, 16.0)
    else:
        p["mark_join_build_switch_ratio"] = 0.0  # disabled -> always filtered_join

    p["enable_dynamic_filter_pushdown"] = trial.suggest_categorical(
        "enable_dynamic_filter_pushdown", [True, False])
    if p["enable_dynamic_filter_pushdown"]:
        p["dynamic_filter_domain_coverage_threshold"] = trial.suggest_float(
            "dynamic_filter_domain_coverage_threshold", 0.5, 1.0)
        p["dynamic_filter_keep_threshold"] = trial.suggest_float(
            "dynamic_filter_keep_threshold", 0.5, 1.0)
    else:
        p["dynamic_filter_domain_coverage_threshold"] = 0.9
        p["dynamic_filter_keep_threshold"] = 0.9

    # --- GPU memory spill thresholds (fractions). stop must sit below trigger. ---
    p["gpu_downgrade_trigger_fraction"] = trial.suggest_float(
        "gpu_downgrade_trigger_fraction", 0.55, 0.95)
    gap = trial.suggest_float("gpu_downgrade_stop_gap", 0.05, 0.4)
    p["gpu_downgrade_stop_fraction"] = round(max(0.1, p["gpu_downgrade_trigger_fraction"] - gap), 4)

    # --- scan_manager IO parallelism ---
    p["uring_n_reactors"] = trial.suggest_int("uring_n_reactors", 1, max(2, min(8, cores)))
    p["enable_prefetch_cache"] = trial.suggest_categorical(
        "enable_prefetch_cache", [True, False])

    return p


def baseline_params(hw: dict) -> dict:
    """Sirius in-source defaults expressed in this study's parameter names, so the
    reference point is comparable on every machine. Byte defaults become VRAM
    fractions; thread defaults are clamped into the core-scaled ranges."""
    vram = hw["vram_bytes"]
    cores = hw["cores"]

    def clampi(v, lo, hi):
        return max(lo, min(hi, v))

    return {
        "gpu_usage_limit_fraction": 0.9,
        "host_capacity_fraction": 0.25,
        "pipeline_num_threads": clampi(4, 2, max(2, cores)),
        "task_creator_num_threads": clampi(2, 1, max(2, cores // 2)),
        "scan_manager_num_threads": clampi(8, 2, max(2, cores)),
        "downgrade_num_threads": clampi(1, 1, max(2, cores // 4)),
        "scan_task_batch_vram_frac": 512 * MiB / vram,
        "hash_partition_vram_frac": 512 * MiB / vram,
        "concat_batch_vram_frac": 512 * MiB / vram,
        "sort_sample_vram_frac": 512 * MiB / vram,
        "max_build_hash_vram_frac": 500 * MiB / vram,
        "max_sort_partition_memory_fraction": 0.33,
        "enable_mark_join_switch": True,
        "mark_join_build_switch_ratio": 8.0,
        "enable_dynamic_filter_pushdown": True,
        "dynamic_filter_domain_coverage_threshold": 0.9,
        "dynamic_filter_keep_threshold": 0.9,
        "gpu_downgrade_trigger_fraction": 0.75,
        "gpu_downgrade_stop_gap": 0.10,
        "uring_n_reactors": 1,
        "enable_prefetch_cache": False,
    }


# --------------------------------------------------------------------------
# Config materialization
# --------------------------------------------------------------------------
def write_yaml(p: dict, path: Path, disk_dir: Path, hw: dict) -> None:
    host_capacity = int(hw["ram_bytes"] * p["host_capacity_fraction"])
    yaml = f"""sirius:
  topology:
    num_gpus: {hw["num_gpus"]}
  memory:
    gpu:
      usage_limit_fraction: {p["gpu_usage_limit_fraction"]:.4f}
      reservation_limit_fraction: 1.0
      downgrade_trigger_fraction: {p["gpu_downgrade_trigger_fraction"]:.4f}
      downgrade_stop_fraction: {p["gpu_downgrade_stop_fraction"]:.4f}
    host:
      capacity_bytes: {host_capacity}
      initial_number_pools: 4
      pool_size: 128
      block_size: 1048576
    disk:
      disk_id: 0
      capacity_bytes: {2 * (1 << 40)}
      downgrade_root_dirs: "{disk_dir}"
  executor:
    pipeline:
      num_threads: {p["pipeline_num_threads"]}
    task_creator:
      num_threads: {p["task_creator_num_threads"]}
    scan_manager:
      num_threads: {p["scan_manager_num_threads"]}
      uring_n_reactors: {p["uring_n_reactors"]}
      enable_prefetch_cache: {str(p["enable_prefetch_cache"]).lower()}
    downgrade:
      num_threads: {p["downgrade_num_threads"]}
      monitor_period: 10ms
  operator_params:
    scan_task_batch_size: {p["scan_task_batch_size"]}
    default_scan_task_varchar_size: 256
    max_sort_partition_bytes: 0
    max_sort_partition_memory_fraction: {p["max_sort_partition_memory_fraction"]:.4f}
    hash_partition_bytes: {p["hash_partition_bytes"]}
    concat_batch_bytes: {p["concat_batch_bytes"]}
    sort_sample_bytes: {p["sort_sample_bytes"]}
    max_build_hash_table_bytes: {p["max_build_hash_table_bytes"]}
    mark_join_build_switch_ratio: {p["mark_join_build_switch_ratio"]:.4f}
    enable_dynamic_filter_pushdown: {str(p["enable_dynamic_filter_pushdown"]).lower()}
    dynamic_filter_domain_coverage_threshold: {p["dynamic_filter_domain_coverage_threshold"]:.4f}
    dynamic_filter_keep_threshold: {p["dynamic_filter_keep_threshold"]:.4f}
  telemetry:
    enable_quent: false
"""
    path.write_text(yaml)


def build_view_sql(parquet_dir: Path) -> str:
    stmts = []
    for table in TPCH_TABLES:
        files = []
        for pat in (f"{table}.parquet", f"{table}_*.parquet", f"{table}/*.parquet"):
            files.extend(sorted(glob.glob(str(parquet_dir / pat))))
        if not files:
            raise FileNotFoundError(f"No parquet files for table '{table}' in {parquet_dir}")
        file_list = ",".join(f"'{f}'" for f in files)
        stmts.append(f"CREATE VIEW {table} AS SELECT * FROM read_parquet([{file_list}]);")
    return "\n".join(stmts) + "\n"


def prime_os_cache(parquet_dir: Path) -> int:
    """Read every parquet file once to warm the OS page cache before the study,
    so the first measured trials aren't penalized by cold-disk reads relative to
    later ones (by which point the data is already resident). Returns bytes read.
    Note: only fully effective when the dataset fits in RAM."""
    files = []
    for table in TPCH_TABLES:
        for pat in (f"{table}.parquet", f"{table}_*.parquet", f"{table}/*.parquet"):
            files.extend(glob.glob(str(parquet_dir / pat)))
    total = 0
    for f in files:
        try:
            with open(f, "rb", buffering=0) as fh:
                while chunk := fh.read(16 * 1024 * 1024):
                    total += len(chunk)
        except OSError:
            pass
    return total


def build_sql(p: dict, view_sql: str, query_dir: Path, iterations: int, queries: list[int]) -> str:
    parts = [view_sql, ".timer on\n"]
    for q in queries:
        qfile = query_dir / f"q{q}.sql"
        qtext = qfile.read_text()
        parts.append(f".print __Q{q}__\n")
        for _ in range(iterations):
            parts.append(qtext)
            if not qtext.endswith("\n"):
                parts.append("\n")
    return "".join(parts)


# --------------------------------------------------------------------------
# Trial execution
# --------------------------------------------------------------------------
def run_config(
    p: dict,
    duckdb: Path,
    view_sql: str,
    query_dir: Path,
    iterations: int,
    queries: list[int],
    workdir: Path,
    spill_dir: Path,
    hw: dict,
    timeout_s: float | None,
) -> tuple[float | None, str]:
    """Run one config. Returns (total_seconds, status). total is None on failure."""
    spill_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = workdir / "sirius.yaml"
    write_yaml(p, yaml_path, spill_dir, hw)

    sql = build_sql(p, view_sql, query_dir, iterations, queries)
    with tempfile.NamedTemporaryFile("w", suffix=".sql", dir=workdir, delete=False) as f:
        sql_path = f.name
        f.write(sql)

    env = dict(os.environ)
    env["SIRIUS_CONFIG_FILE"] = str(yaml_path)
    env["SIRIUS_LOG_DIR"] = str(workdir)
    env["SIRIUS_LOG_LEVEL"] = "warn"

    try:
        proc = subprocess.run(
            [str(duckdb), "-f", sql_path],
            env=env, capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return None, "timeout"
    finally:
        try:
            os.unlink(sql_path)
        except OSError:
            pass

    out = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        return None, f"exit={proc.returncode}"
    if ERROR_RE.search(out):
        return None, "query_error"

    times = [float(m) for m in RUN_TIME_RE.findall(out)]
    expected = len(queries) * iterations
    if len(times) != expected:
        return None, f"parse_mismatch({len(times)}/{expected})"
    return sum(times), "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sf", type=int, default=300)
    ap.add_argument("--parquet-dir", type=Path, default=None)
    ap.add_argument("--duckdb", type=Path, default=DEFAULT_DUCKDB)
    ap.add_argument("--query-dir", type=Path, default=DEFAULT_QUERY_DIR)
    ap.add_argument("--iterations", type=int, default=2,
                    help="iterations per query (1 cold + N-1 warm); default 2")
    ap.add_argument("--n-trials", type=int, default=300)
    ap.add_argument("--queries", default="1-22",
                    help="query subset, e.g. '1,3,6-10' (default all 22)")
    ap.add_argument("--study-name", default=None)
    ap.add_argument("--workdir", type=Path, default=Path("experimental/tpch-optuna/runs"))
    ap.add_argument("--spill-dir", type=Path, default=None,
                    help="disk-spill dir for downgrades (default: <workdir>/disk_spill)")
    ap.add_argument("--timeout-mult", type=float, default=4.0,
                    help="per-trial timeout = timeout-mult x best-so-far seconds")
    ap.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True,
                    help="read the dataset once to warm the OS page cache before the "
                         "study, so early trials aren't penalized by cold-disk reads "
                         "(--no-warmup to skip)")
    args = ap.parse_args()

    def parse_queries(spec: str) -> list[int]:
        out = []
        for part in spec.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-")
                out.extend(range(int(a), int(b) + 1))
            elif part:
                out.append(int(part))
        return out

    queries = parse_queries(args.queries)

    hw = detect_hardware()
    hostname = socket.gethostname().split(".")[0]
    print(f"hardware: host={hostname}  GPUs={hw['num_gpus']}x{hw['vram_bytes']/GiB:.0f}GB VRAM  "
          f"RAM={hw['ram_bytes']/GiB:.0f}GB  cores={hw['cores']}")

    parquet_dir = args.parquet_dir or (
        PROJECT_DIR / "test_datasets" / f"tpch_parquet_sf{args.sf}")
    study_name = args.study_name or f"tpch_sf{args.sf}_{hostname}"
    workdir = (PROJECT_DIR / args.workdir).resolve() if not args.workdir.is_absolute() else args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    trial_dir = workdir / "trial"
    trial_dir.mkdir(exist_ok=True)
    spill_dir = args.spill_dir or (workdir / "disk_spill")

    if not args.duckdb.exists():
        print(f"ERROR: duckdb binary not found: {args.duckdb}", file=sys.stderr)
        return 1
    if not parquet_dir.exists():
        print(f"ERROR: parquet dir not found: {parquet_dir}", file=sys.stderr)
        return 1

    view_sql = build_view_sql(parquet_dir)

    if args.warmup:
        t0 = time.time()
        n = prime_os_cache(parquet_dir)
        print(f"warmup: primed OS page cache with {n / GiB:.1f} GB "
              f"in {time.time() - t0:.0f}s")

    # SQLite in WAL mode + a busy timeout so the live dashboard (a concurrent
    # reader) doesn't collide with the study writer — plain rollback-journal
    # SQLite serializes them and the dashboard 500s on every commit, especially
    # at small scale factors where trials finish every couple of seconds.
    # WAL must be set up-front on a single fresh connection: SQLite refuses to
    # switch journal mode while any other connection is open, so once optuna's
    # pool is up it's too late. WAL is persisted in the db header, so every later
    # connection (optuna's pool and the dashboard's) inherits it.
    db_path = workdir / (study_name + ".db")
    _wal = sqlite3.connect(str(db_path))
    _wal.execute("PRAGMA journal_mode=WAL")
    _wal.close()

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{db_path}",
        engine_kwargs={"connect_args": {"timeout": 60}},
    )

    @event.listens_for(storage.engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _rec):  # noqa: ANN001
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA busy_timeout=60000")
        cur.close()

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True, group=True),
        load_if_exists=True,
    )
    if not study.trials:
        study.enqueue_trial(baseline_params(hw))

    def objective(trial: optuna.Trial) -> float:
        p = sample_params(trial, hw)
        try:
            best = study.best_value
        except ValueError:
            best = None  # no completed trial yet
        timeout_s = best * args.timeout_mult if best else None
        t0 = time.time()
        total, status = run_config(
            p, args.duckdb, view_sql, args.query_dir,
            args.iterations, queries, trial_dir, spill_dir, hw, timeout_s,
        )
        wall = time.time() - t0
        trial.set_user_attr("status", status)
        trial.set_user_attr("wall_s", round(wall, 1))
        if total is None:
            # Penalty relative to best so the sampler still gets a gradient.
            penalty = (best * args.timeout_mult if best else 1e6)
            print(f"[trial {trial.number}] FAIL {status} ({wall:.0f}s wall) -> penalty {penalty:.1f}")
            return penalty
        print(f"[trial {trial.number}] {status} total={total:.2f}s ({wall:.0f}s wall)")
        return total

    study.optimize(objective, n_trials=args.n_trials)

    print("\n=== best ===")
    print(f"value (sum cold+warm): {study.best_value:.2f}s")
    for k, v in sorted(study.best_params.items()):
        print(f"  {k} = {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
