"""Use the selected server's resident interpreter and data paths."""

import json
import os
from pathlib import Path
import sys


def command(config, here, arguments):
    runtime = config["evaluate" if "--evaluate" in arguments else "train"]
    environment = dict(os.environ, **runtime["env"])
    return [runtime["python"], "-u", str(here/"run.py"), *arguments], environment


def main():
    here = Path(__file__).resolve().parent
    config = json.loads((here/"node_runtime.json").read_text())
    argv, environment = command(config, here, sys.argv[1:])
    os.chdir(here)
    os.execve(argv[0], argv, environment)


if __name__ == "__main__":
    main()
