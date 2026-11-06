FROM python:3.12
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

# Dependencies only: vllm-torchtpu source is mounted at run time and installed
# with `pip install --no-deps -e`, so experiments do not rebuild this image.
ARG VLLM_COMMIT=e37e51dd246cf421c554e7d6f53e178e3fd29085
ARG VLLM_TORCHTPU_COMMIT=8421301cf61f9e9ef2b08881cbd11832710136b1
ARG TORCH_TPU_VERSION=0.2.0
LABEL org.opencontainers.image.description="Dependency image for vllm-torchtpu on TPU v7x" \
      org.opencontainers.image.vllm.revision="${VLLM_COMMIT}" \
      org.opencontainers.image.revision="${VLLM_TORCHTPU_COMMIT}"

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    VLLM_TARGET_DEVICE=empty \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0

RUN python -m pip install --upgrade pip setuptools wheel setuptools_scm packaging \
        ninja cmake pybind11 \
    && python -m pip install torch==2.13.0 torchvision==0.28.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-deps "torch-tpu==${TORCH_TPU_VERSION}"

# vllm and vllm-torchtpu at pinned commits; their build requirements first,
# since vllm installs without build isolation against the torch above.
WORKDIR /opt/src
RUN for repo in vllm:${VLLM_COMMIT} vllm-torchtpu:${VLLM_TORCHTPU_COMMIT}; do \
        name=${repo%%:*}; commit=${repo##*:}; \
        git init -q "$name" \
        && git -C "$name" fetch -q --depth=1 "https://github.com/vllm-project/$name.git" "$commit" \
        && git -C "$name" checkout -q --detach FETCH_HEAD \
        && test "$(git -C "$name" rev-parse HEAD)" = "$commit"; \
    done

COPY requirements.py /opt/src/requirements.py
RUN python /opt/src/requirements.py build vllm vllm-torchtpu > /opt/src/build.txt \
    && python -m pip install -r /opt/src/build.txt \
    && VLLM_VERSION_OVERRIDE=0.1.dev1+g${VLLM_COMMIT:0:9}.empty \
        python -m pip install --no-build-isolation ./vllm \
    && python /opt/src/requirements.py runtime vllm-torchtpu "${TORCH_TPU_VERSION}" \
        > /opt/src/runtime.txt \
    && python -m pip install -r /opt/src/runtime.txt \
    && python -m pip check \
    && python -m pip freeze > /opt/requirements-installed.txt \
    && rm -rf /opt/src/vllm-torchtpu

# vllm itself is not imported: it needs the vllm-torchtpu platform plugin,
# which is installed at run time.
RUN JAX_PLATFORMS=cpu python -c "import importlib.metadata as m, jax, torch, torch_tpu; \
print({p: m.version(p) for p in ('torch', 'torch-tpu', 'vllm', 'jax', 'jaxlib', 'libtpu', 'tpu-sync-torch')}); \
print(jax.devices())"

ENV VLLM_TARGET_DEVICE=tpu \
    JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache
RUN mkdir -p /tmp/jit_cache
CMD ["bash"]
