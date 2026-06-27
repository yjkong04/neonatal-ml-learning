# syntax=docker/dockerfile:1

# Stage 1 — fetch the model checkpoint from the GitHub Release asset.
# curl and the apt cache never make it into the final image.
FROM python:3.11-slim AS model-fetcher
ARG MODEL_URL=https://github.com/yjkong04/neonatal-ml-learning/releases/download/kramer-classifier-deploy-v1/best_model.pt
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fL "$MODEL_URL" -o /best_model.pt

# Stage 2 — the actual app image.
FROM python:3.11-slim AS final
WORKDIR /app

COPY requirements-deploy.txt .
# CPU-only torch/torchvision wheels (see requirements-deploy.txt) keep this under ~500MB —
# the default PyPI wheels bundle CUDA and would roughly triple it for no benefit on Fly's CPU machines.
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY app ./app
COPY --from=model-fetcher /best_model.pt ./model/best_model.pt

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV MODEL_PATH=/app/model/best_model.pt
EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
