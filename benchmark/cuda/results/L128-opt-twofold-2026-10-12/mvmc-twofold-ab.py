"""Run two unchanged source checkouts serially, with warmed interleaved trials."""
import os
from pathlib import Path
import subprocess
import sys

baseline, candidate, input_file, checkpoint, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
worker = Path(__file__).with_name("mvmc-twofold-worker.jl")
processes = {}
logs = {}

def until(variant, marker):
    process = processes[variant]
    while True:
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"{variant} exited {process.poll()} before {marker}")
        logs[variant].write(line)
        logs[variant].flush()
        print(f"{variant}: {line}", end="", flush=True)
        if line.rstrip() == marker:
            return

try:
    for variant, checkout in (("baseline", baseline), ("candidate", candidate)):
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
        print(f"{variant} checkout={checkout} source_commit={commit}", flush=True)
        logs[variant] = (output / f"{variant}-worker.log").open("w")
        processes[variant] = subprocess.Popen(
            ["julia", "+1.13.1", "--startup-file=no", f"--project={checkout / 'test/gpu'}",
             str(worker), str(input_file), str(checkpoint), str(output), variant],
            cwd=checkout, env={**os.environ, "JULIA_NUM_THREADS": "1"},
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        until(variant, "READY")
        processes[variant].stdin.write("0\n")
        processes[variant].stdin.flush()
        until(variant, "DONE 0")
    for rep in range(1, 6):
        order = ("baseline", "candidate") if rep % 2 else ("candidate", "baseline")
        for variant in order:
            processes[variant].stdin.write(f"{rep}\n")
            processes[variant].stdin.flush()
            until(variant, f"DONE {rep}")
finally:
    for process in processes.values():
        if process.poll() is None:
            process.stdin.write("quit\n")
            process.stdin.flush()
            process.wait(timeout=60)
    for log in logs.values():
        log.close()
