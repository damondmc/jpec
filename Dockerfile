# The jpec command for GPU nodes: the package and JAX's CUDA 12 wheels, without the notebook
# extras. The CUDA libraries come with the wheels; the node supplies only the NVIDIA driver.
FROM python:3.13-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.13 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/jpec

WORKDIR /src/jpec
# the locked dependencies first, so a change to the code does not reinstall them
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --extra gpu
COPY src ./src
RUN uv sync --frozen --no-editable --extra gpu

ENV PATH=/opt/jpec/bin:$PATH
WORKDIR /work
CMD ["jpec", "--help"]
