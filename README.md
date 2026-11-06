# vllm-torchtpu dependency image

A Linux AMD64 TPU environment from `python:3.12` with every dependency of
[`vllm-project/vllm-torchtpu`](https://github.com/vllm-project/vllm-torchtpu)
preinstalled: torch 2.13.0 (CPU), torch-tpu 0.2.0 from PyPI, JAX/jaxlib
0.11.2, libtpu 0.0.50, and vLLM built from source at its pinned commit.

The vllm-torchtpu source is not included. Mount a checkout and install it
without dependencies, which takes seconds:

```bash
pip install --no-deps --no-build-isolation -e /workspace/vllm-torchtpu
```

Rebuild only when vllm-torchtpu's dependency pins change.

## Build and publish

Open **Actions → Build TPU image → Run workflow** and give the
vllm-torchtpu commit whose `pyproject.toml` the image should follow. The
workflow runs on a GitHub-hosted runner and publishes to GHCR with its
temporary `GITHUB_TOKEN`.

Each run publishes
`ghcr.io/yanko-7/vllm-torchtpu-deps:py312-<commit9>-r<RUN_NUMBER>` and moves
`latest`. The run summary records the immutable digest; reference the digest
in experiment configs.

On the first publish, set the GHCR package visibility to public if anonymous
pulls are required.

## Differences from the vllm-torchtpu pins

PyPI carries torch-tpu `0.2.0` but not the `0.2.0rc1` that vllm-torchtpu
pins, so the image installs `0.2.0`. Results measured with rc1 should be
rechecked on this image.

## Validation

The build runs `pip check` and imports torch, torch-tpu, vLLM and JAX on the
CPU backend. The runner has no TPU, so TPU execution is not validated here.
Installed versions are in `/opt/requirements-installed.txt`.
