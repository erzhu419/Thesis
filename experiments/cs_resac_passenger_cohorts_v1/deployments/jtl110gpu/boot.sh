#!/bin/sh
exec /home/erzhu419/.venvs/scheduleurm-torch-bench/bin/python -u "$(dirname "$0")/node_run.py" "$@"
