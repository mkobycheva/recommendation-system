---
title: Cross-domain Recommender
emoji: 📚
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.20.0
app_file: main.py
pinned: false
---

# Cross-domain Recommender (SVD / ALS / item2vec / SASRec / BERT4Rec)

Public deployment of the books+movies cross-domain recommender MVP. Single
Gradio process — recommenders run in-process (`app/recommenders/`), no
separate FastAPI backend (that split still exists for local dev, see
`../app/api/main.py` + `../app/frontend/app.py` in the main repo).

## How this Space is assembled

This Space's git repo is **not** the same as the `recommendation-system`
GitHub repo — it only needs a subset of it:

```
<space repo root>/
  main.py                  <- copied from space/main.py (entry point -- NOT
                               named app.py, that would collide with the
                               app/ package directory below)
  requirements.txt        <- copied from space/requirements.txt
  README.md                <- this file
  app/
    __init__.py            <- copied from app/__init__.py (empty)
    recommenders/
      __init__.py
      base.py
      svd_als.py
      item2vec.py
      sequential.py
```

To (re)deploy after code changes, from the `recommendation-system` repo root
(`app` branch):

```bash
rm -rf /tmp/hf-space-deploy && mkdir -p /tmp/hf-space-deploy/app
cp space/main.py /tmp/hf-space-deploy/main.py
cp space/requirements.txt /tmp/hf-space-deploy/requirements.txt
cp space/README.md /tmp/hf-space-deploy/README.md
cp app/__init__.py /tmp/hf-space-deploy/app/__init__.py
cp -r app/recommenders /tmp/hf-space-deploy/app/recommenders

cd /tmp/hf-space-deploy
git init -q
git remote add space https://huggingface.co/spaces/<your-hf-username>/<space-name>
git add .
git commit -m "Deploy"
git push --force space main
```

(First time: create the Space on huggingface.co first -- New Space, SDK:
Gradio -- so the `space` remote above exists to push to.)

## Model artifacts

Not stored in this repo (too large for the free-tier 1GB cap). Downloaded at
startup from a public HF Dataset repo via `huggingface_hub.snapshot_download`
-- see `ARTIFACTS_REPO` in `main.py` and the "Publish artifacts" cell at the
end of `notebooks/00_export_metadata.ipynb` in the main repo, which is what
populates that Dataset repo. Re-run that cell after retraining any model, then
restart this Space to pick up the refresh.
