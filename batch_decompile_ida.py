# batch_decompile_ida.py
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import signal
import time
from pathlib import Path
from typing import Optional, Tuple

# =========================
# Configuration
# =========================
REPO_ROOT = Path(__file__).resolve().parent
DECOMPILE_ROOT = REPO_ROOT / "Decompile"
TARGET_FOLDER = str(DECOMPILE_ROOT / "Decompile_Input_Raw_PE")
OUTPUT_FOLDER = str(DECOMPILE_ROOT / "Decompile_Output_C_Code")

# Failure log: in Decompile folder
FAIL_LOG = str(DECOMPILE_ROOT / "decompile_failure_log.txt")

# Execution parameters
TIMEOUT = 600  # seconds

# Output size policy (post-check)
MIN_C_OUTPUT_BYTES = 10 * 1024  # 10 KB
MAX_C_OUTPUT_BYTES = 3 * 1024 * 1024  # 3 MB

# IDA lookup: ONLY via IDA_PATH
IDA_ENV = "IDA_PATH"

# Windows-specific flags (safe no-ops on non-Windows imports)
CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
CTRL_BREAK_EVENT = getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM)


# =========================
# Logging helpers
# =========================
def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


def bytes_human(n: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    i = 0
    x = float(n)
    while x >= 1024 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    return f"{x:.2f} {units[i]}"


# =========================
# IDA discovery (no version check)
# =========================
def _err_and_exit(msg: str, code: int = 1) -> None:
    log(f"[ERROR] {msg}")
    log(
        f"TIP: Set environment variable {IDA_ENV} to the IDA installation directory "
        f"(or directly to idat.exe). Example: set {IDA_ENV}=C:\\Program Files\\IDA Pro"
    )
    sys.exit(code)


def _candidate_from_env(var: str) -> Optional[str]:
    """Return a valid idat.exe path hinted by env var `var`, or None."""
    val = os.getenv(var, "").strip('"').strip()
    log(f"Checking {var} ..." if val else f"{var} is not set.")
    if not val:
        return None
    # If the env var points directly to idat.exe, use it.
    if val.lower().endswith("idat.exe") and os.path.isfile(val):
        log(f"Resolved idat.exe directly from {var}: {val}")
        return val
    # Otherwise assume it's an install directory.
    cand = os.path.join(val, "idat.exe")
    if os.path.isfile(cand):
        log(f"Resolved idat.exe from {var} directory: {cand}")
        return cand
    log(f"Could not find idat.exe under {val}")
    return None


def locate_idat() -> Optional[str]:
    """
    Find idat.exe ONLY via IDA_PATH environment variable.
    Accept either:
      • IDA_PATH=<full path to idat.exe>
      • IDA_PATH=<IDA install folder containing idat.exe>
    """
    return _candidate_from_env(IDA_ENV)


def resolve_ida_path() -> str:
    """Locate idat.exe via IDA_PATH. No version probing/checking."""
    if os.name != "nt":
        _err_and_exit("This script supports Windows only (requires idat.exe).")

    idat = locate_idat()
    if not idat:
        _err_and_exit(
            f"Could not find 'idat.exe' via {IDA_ENV}. Please install IDA and set {IDA_ENV}."
        )

    base = os.path.basename(idat).lower()
    if base != "idat.exe":
        _err_and_exit(
            f"Found '{base}', but only 'idat.exe' is allowed. Please point {IDA_ENV} to 'idat.exe' (or its folder)."
        )

    log(f"[OK] Using IDA at: {idat}")
    # NOTE: Version check intentionally removed per user request.
    return idat


# =========================
# Utilities
# =========================
def ensure_output():
    os.makedirs(DECOMPILE_ROOT, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    if not os.path.exists(FAIL_LOG):
        open(FAIL_LOG, "w", encoding="utf-8").close()
        log(f"Created failure log: {FAIL_LOG}")
    log(
        "Paths:\n"
        f"  REPO_ROOT={REPO_ROOT}\n"
        f"  DECOMPILE_ROOT={DECOMPILE_ROOT}\n"
        f"  INPUT={TARGET_FOLDER}\n"
        f"  OUTPUT={OUTPUT_FOLDER}\n"
        f"  FAIL_LOG={FAIL_LOG}"
    )


def load_failed_set() -> set:
    """Read FAIL_LOG and return a set of failed filenames only (one per line as 'filename<TAB>reason')."""
    failed = set()
    if not os.path.exists(FAIL_LOG):
        log("Failure log not found (will create later).")
        return failed
    with open(FAIL_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fname, _sep, _reason = line.partition("\t")
            fname = fname.strip()
            if fname:
                failed.add(fname)
    log(f"Loaded {len(failed)} previously failed entries from log.")
    return failed


def append_failure(failed: set, filename: str, reason: Optional[str] = None) -> None:
    """Append a failure record to FAIL_LOG once per filename. Format: 'filename<TAB>reason'."""
    if filename in failed:
        return
    with open(FAIL_LOG, "a", encoding="utf-8") as flog:
        if reason:
            flog.write(f"{filename}\t{reason}\n")
        else:
            flog.write(filename + "\n")
    failed.add(filename)
    log(f"[FAIL] {filename} | {reason or 'decompilation failed'} (logged to {FAIL_LOG})")


def kill_process_tree(proc: subprocess.Popen) -> None:
    """Attempt to gracefully stop then force-kill the process tree on Windows."""
    try:
        proc.send_signal(CTRL_BREAK_EVENT)  # requires CREATE_NEW_PROCESS_GROUP
        time.sleep(1)
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass
    finally:
        try:
            proc.wait(timeout=5)
        except Exception:
            pass


def output_size_is_abnormal(path: str) -> Tuple[bool, int]:
    """Return (is_abnormal, size_bytes). Abnormal if size outside [10KB, 3MB]."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return True, -1
    return (size < MIN_C_OUTPUT_BYTES or size > MAX_C_OUTPUT_BYTES), size


def list_input_files(root_dir: str) -> list:
    files = []
    for root, _dirs, fs in os.walk(root_dir):
        for name in fs:
            files.append(os.path.join(root, name))
    return files


# =========================
# Main decompilation loop
# =========================
def main():
    start_ts = time.time()
    log("=== PE-Sleuth | Batch Decompile (IDA Pro Hex-Rays) ===")
    log(f"Process PID: {os.getpid()}")

    idat_exe = resolve_ida_path()  # no version check anymore
    ensure_output()

    failed = load_failed_set()

    # Enumerate inputs first for progress reporting
    inputs = list_input_files(TARGET_FOLDER)
    if not inputs:
        log("No input files found. Please place PE files under 'Decompile/Decompile_Input_Raw_PE'.")
        return

    log(f"Discovered {len(inputs)} input file(s). Starting decompilation...")
    counters = {
        "processed": 0,
        "skipped": 0,
        "succeeded": 0,
        "failed": 0,
        "size_failed": 0,
        "timeout": 0,
        "spawn_error": 0,
        "no_output": 0,
        "already_exists": 0,
        "prev_failed_skip": 0,
        "cleaned_artifacts": 0,
    }

    for idx, input_file in enumerate(inputs, start=1):
        filename = os.path.basename(input_file)
        log(f"[{idx}/{len(inputs)}] Processing: {filename}")
        counters["processed"] += 1

        if filename in failed:
            log(f"  → Skip (prev failed): {filename}")
            counters["skipped"] += 1
            counters["prev_failed_skip"] += 1
            continue

        base_name = os.path.splitext(filename)[0]
        out_name = f"{base_name}.c"
        output_c = os.path.join(OUTPUT_FOLDER, out_name)
        if os.path.exists(output_c):
            size_now = os.path.getsize(output_c)
            log(f"  → Skip (exists): {out_name} [{bytes_human(size_now)}]")
            counters["skipped"] += 1
            counters["already_exists"] += 1
            continue

        # Build command (Hex-Rays output; requires decompiler license)
        cmd = (
            f'cd /d "{OUTPUT_FOLDER}" && '
            f'"{idat_exe}" -Ohexrays:-new:-nosave:{out_name}:ALL -A "{input_file}"'
        )
        log(f"  CMD: {cmd}")

        t0 = time.time()
        try:
            proc = subprocess.Popen(
                cmd,
                shell=True,
                cwd=OUTPUT_FOLDER,
                creationflags=CREATE_NEW_PROCESS_GROUP,
            )
        except Exception as e:
            log(f"  [-] Failed to spawn IDA for {filename}: {e}")
            append_failure(failed, filename, "spawn error")
            counters["failed"] += 1
            counters["spawn_error"] += 1
            continue

        succeeded = False
        size_failed = False
        try:
            proc.wait(timeout=TIMEOUT)
            elapsed = time.time() - t0
            if os.path.exists(output_c):
                abnormal, size_bytes = output_size_is_abnormal(output_c)
                if abnormal:
                    log(
                        f"  [!] Abnormal output size: {out_name} "
                        f"({bytes_human(size_bytes)}; "
                        f"policy {bytes_human(MIN_C_OUTPUT_BYTES)}–{bytes_human(MAX_C_OUTPUT_BYTES)}). "
                        "Treating as failure and removing file."
                    )
                    try:
                        os.remove(output_c)
                        log("  • Removed abnormal output file.")
                    except Exception as rm_e:
                        log(f"  [!] Failed to remove abnormal output: {rm_e}")
                    append_failure(failed, filename, "decompilation failed")
                    size_failed = True
                    counters["failed"] += 1
                    counters["size_failed"] += 1
                else:
                    log(f"  [+] Decompiled OK → {out_name} [{bytes_human(size_bytes)}] in {elapsed:.1f}s")
                    succeeded = True
                    counters["succeeded"] += 1
            else:
                log("  [!] No .c output produced.")
                counters["failed"] += 1
                counters["no_output"] += 1
        except subprocess.TimeoutExpired:
            log(f"  [!] Timeout after {TIMEOUT}s. Killing IDA process ...")
            kill_process_tree(proc)
            counters["failed"] += 1
            counters["timeout"] += 1
        except Exception as e:
            log(f"  [-] Error running idat: {e}")
            counters["failed"] += 1

        # Clean temporary IDA db artifacts located next to the input file
        cleaned_for_this = 0
        tmp_i64 = os.path.join(os.path.dirname(input_file), filename + ".i64")
        if succeeded and os.path.exists(tmp_i64):
            try:
                os.remove(tmp_i64)
                cleaned_for_this += 1
            except Exception:
                pass
        else:
            # If not already logged (size_failed covers one specific path), log generic failure.
            if not size_failed and not succeeded:
                append_failure(failed, filename, "decompilation failed")

            for ext in (".i64", ".id0", ".id1", ".id2", ".nam", ".til"):
                tmp = os.path.join(os.path.dirname(input_file), filename + ext)
                if os.path.exists(tmp):
                    for _ in range(3):
                        try:
                            os.remove(tmp)
                            cleaned_for_this += 1
                            break
                        except PermissionError:
                            time.sleep(0.5)

        counters["cleaned_artifacts"] += cleaned_for_this
        if cleaned_for_this:
            log(f"  • Cleaned {cleaned_for_this} temp artifact(s).")

    # Summary
    total_elapsed = time.time() - start_ts
    log("=== Summary ===")
    log(f"  Processed:   {counters['processed']}")
    log(f"  Succeeded:   {counters['succeeded']}")
    log(f"  Failed:      {counters['failed']}  "
        f"(timeouts={counters['timeout']}, size_failed={counters['size_failed']}, no_output={counters['no_output']}, spawn_error={counters['spawn_error']})")
    log(f"  Skipped:     {counters['skipped']}  "
        f"(prev_failed={counters['prev_failed_skip'] if 'prev_failed_skip' in counters else 0}, exists={counters['already_exists']})")
    log(f"  Artifacts cleaned: {counters['cleaned_artifacts']}")
    log(f"  Failure log: {FAIL_LOG}")
    log(f"All done in {total_elapsed:.1f}s.")


if __name__ == "__main__":
    main()
