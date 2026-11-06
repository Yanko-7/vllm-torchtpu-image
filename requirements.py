"""Print pip requirements read from the checked-out repos' pyproject files.

Usage:
  requirements.py build REPO...            build-system requirements of REPOs
  requirements.py runtime REPO TORCH_TPU   REPO's dependencies for the image
"""

import sys
import tomllib
from pathlib import Path

# Pins the held-pod recipe adds on top of vllm-torchtpu's own dependencies.
EXTRA = [
    "numpy==2.3.5",
    "transformers==5.18.0",
    "frozendict==2.4.7",
    "immutabledict==4.3.1",
    "tensorboard==2.21.0",
    "runai-model-streamer[gcs]==0.17.0",
    "runai-model-streamer-gcs==0.17.0",
    "pytest",
    "pytest-mock",
    "pytest-timeout",
    "absl-py",
    "parameterized",
]


def project(repo):
    return tomllib.loads((Path(repo) / "pyproject.toml").read_text())


mode, *args = sys.argv[1:]
if mode == "build":
    lines = sorted({r for repo in args for r in project(repo)["build-system"]["requires"]})
elif mode == "runtime":
    repo, torch_tpu = args
    # vllm is installed from source; torch-tpu comes from PyPI, which only
    # carries the released version.
    lines = [
        r
        for r in project(repo)["project"]["dependencies"]
        if not r.startswith(("vllm @", "torch-tpu=="))
    ]
    lines += [f"torch-tpu=={torch_tpu}", *EXTRA]
else:
    sys.exit(__doc__)
print("\n".join(lines))
