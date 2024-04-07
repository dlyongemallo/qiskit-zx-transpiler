#!/bin/bash
set -o pipefail
python run_benchmarks.py | tee benchmarks_output.txt
