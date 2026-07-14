# Cloud Run (or any container host) image for the Gradio app in space/main.py.
# Reuses the same entry point HF Spaces would use -- see space/README.md for
# that path. Only needs app/recommenders (inference code) and space/ (entry
# point + its requirements); everything else in the repo (notebooks, data
# pipeline, tests) is irrelevant to the running container.

FROM python:3.12-slim

WORKDIR /app

COPY space/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/__init__.py app/__init__.py
COPY app/recommenders app/recommenders
COPY space/main.py main.py

# Set at deploy time (gcloud run deploy --set-env-vars ARTIFACTS_REPO=...),
# not baked in here -- keeps this image reusable without a rebuild if the
# artifacts repo changes.
ENV ARTIFACTS_REPO=""

EXPOSE 8080
CMD ["python", "main.py"]
