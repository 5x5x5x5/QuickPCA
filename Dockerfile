# QuickPCA — CPU image
#
# Headless essential-dynamics PCA, cross-correlation and Free-Energy Landscapes
# for MD trajectories. This image installs the CPU build with the visualization
# (plotly/kaleido) and web (streamlit) extras.
#
# Build:
#   docker build -t quickpca:latest .
#
# Run (mount your own data — see note below):
#   docker run --rm -v "$PWD/data:/data" quickpca:latest \
#       run /data/reference.pdb /data/trajectory.nc -o /data/PCA_Report.png
#
# NOTE ON DATA: trajectory files can be large and are deliberately excluded from
# the image (see .dockerignore). Mount your topology/trajectory at runtime with
# `-v /path/to/your/data:/data` and reference them under /data.
#
# NOTE ON PERMISSIONS: the container runs as the non-root user `quickpca`
# (uid 1000). To let it write reports back into a host bind mount, the mounted
# directory must be writable by uid 1000 — either `chmod` it, `chown` it to
# 1000, or override the user with `--user "$(id -u):$(id -g)"`.

FROM python:3.12-slim

# Keep Python predictable and quiet inside the container.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# The CPU dependency set (numpy, scipy, scikit-learn, matplotlib, MDAnalysis,
# plotly, kaleido, streamlit) ships prebuilt manylinux wheels for cpython 3.12,
# so no compiler toolchain is required. If a future dependency needs to build
# from source, add build-essentials here, e.g.:
#   RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
#       && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project sources, then install the package with the CPU extras.
# Installing from the copied source keeps the build self-contained.
COPY . /app
RUN pip install ".[viz,web]"

# Run as a non-root user for safety. /data is a conventional mount point for
# user-supplied trajectories and report output.
RUN useradd --create-home --uid 1000 quickpca \
    && mkdir -p /data \
    && chown quickpca:quickpca /data
USER quickpca
WORKDIR /data

ENTRYPOINT ["quickpca"]
CMD ["--help"]

# -----------------------------------------------------------------------------
# GPU variant (optional, not built by default)
# -----------------------------------------------------------------------------
# For GPU-accelerated PCA via the JAX backend, build from a CUDA base image and
# install the JAX extra with a CUDA-enabled jaxlib instead of the steps above:
#
#   FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
#   RUN apt-get update && apt-get install -y --no-install-recommends \
#           python3 python3-pip \
#       && rm -rf /var/lib/apt/lists/*
#   WORKDIR /app
#   COPY . /app
#   # Install the CUDA build of jax/jaxlib (matching your CUDA version), then the
#   # quickpca jax extra:
#   RUN pip install --upgrade "jax[cuda12]" \
#       && pip install ".[jax,viz,web]"
#   # Run with: docker run --gpus all ... and pass `--backend jax` to quickpca.
#
# See https://jax.readthedocs.io/en/latest/installation.html for the jaxlib
# wheel matching your CUDA/cuDNN toolkit.
