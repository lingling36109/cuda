#!/usr/bin/env python3
"""Submit HW5 leaderboard results to the CS5220 leaderboard server."""

import re
import sys
import urllib.request
import json

LEADERBOARD_URL = "https://leaderboard-zm07.onrender.com/api/hw5/submit"
HEADER = "===== CS5220 HW5 LEADERBOARD SUBMISSION ====="
FOOTER = "===== END CS5220 HW5 LEADERBOARD SUBMISSION ====="

MATRICES = ["nlpkkt120", "delaunay_n24", "Cube_Coup_dt0"]


def strip_ansi(text):
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


def parse_gflops(submission):
    """Extract per-matrix average GFLOPS/s (averaged over k values) from submission text."""
    clean = strip_ansi(submission)

    pattern = (
        r'Running SpMM \(matrices/([^/]+)/[^,]+, k = (\d+)\)'
        r'.*?\[SpMM_student\]:\s+(\d+)\s+gflops/s'
    )
    runs = re.findall(pattern, clean, re.DOTALL)

    matrix_gflops = {}
    for matrix_name, k, gflops_val in runs:
        matrix_gflops.setdefault(matrix_name, []).append(int(gflops_val))

    results = {}
    for mat in MATRICES:
        if mat in matrix_gflops:
            results[mat] = sum(matrix_gflops[mat]) / len(matrix_gflops[mat])

    if results:
        results["avg_gflops"] = sum(results.values()) / len(results)

    return results


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <output-file>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        raw = f.read()

    if HEADER not in raw:
        print("ERROR: Submission header not found in output file.")
        print("This means the job may not have started correctly.")
        print("Re-run: sbatch job-leaderboard")
        sys.exit(1)

    if FOOTER not in raw:
        print("ERROR: Submission footer not found in output file.")
        print("The job may not have completed successfully.")
        sys.exit(1)

    # Extract the submission block
    start = raw.index(HEADER)
    end = raw.index(FOOTER) + len(FOOTER)
    submission = raw[start:end]

    # Parse name
    name_match = re.search(r"LEADERBOARD_NAME:\s*(\S+)", submission)
    name = name_match.group(1) if name_match else "unknown"

    # Parse and display GFLOPS/s for confirmation
    results = parse_gflops(submission)

    print(f"Name: {name}")
    print()
    for mat in MATRICES:
        val = results.get(mat)
        if val is not None:
            print(f"  {mat:20s}  {val:8.1f} GFLOPS/s")
        else:
            print(f"  {mat:20s}  --")
    avg = results.get("avg_gflops")
    if avg is not None:
        print(f"  {'Average':20s}  {avg:8.1f} GFLOPS/s")
    else:
        print(f"  {'Average':20s}  --")
    print()

    if not results:
        print("ERROR: Could not parse any GFLOPS/s values from output.")
        print("Check that correctness tests passed and benchmarks completed.")
        sys.exit(1)

    # Submit
    print(f"Submitting to {LEADERBOARD_URL} ...")
    req = urllib.request.Request(
        LEADERBOARD_URL,
        data=submission.encode("utf-8"),
        headers={"Content-Type": "text/plain"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            print(f"Success! Submitted as: {result.get('name', name)}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"ERROR: Server returned {e.code}: {body}")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"ERROR: Could not reach server: {e.reason}")
        sys.exit(1)


if __name__ == "__main__":
    main()