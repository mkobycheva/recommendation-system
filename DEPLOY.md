# Deploy to Google Cloud Run

Public deployment of the MVP app (`space/main.py`) for course use -- low
traffic (~10 students), needed for about a month. Uses `min-instances=0`
(scales to zero, no cost while idle) rather than always-on, so the tradeoff
is a slow (1-3 min) cold start on the first request after a period of
inactivity, while staying free/near-free for this usage level.

Builds happen on Google Cloud Build (`--source .`), not locally -- no local
Docker needed, nothing touches your disk.

## One-time setup

```bash
# Install the gcloud CLI if you don't have it: https://cloud.google.com/sdk/docs/install

gcloud auth login
gcloud projects create <your-project-id> --name="recsys-mvp"   # or reuse an existing project
gcloud config set project <your-project-id>

# Billing must be enabled on the project for Cloud Run/Cloud Build, even to
# stay within the free tier -- Google Cloud Console will prompt for this if
# it's not already set up.

gcloud services enable run.googleapis.com cloudbuild.googleapis.com
```

## Publish artifacts first

The container downloads model artifacts from a public HF Dataset repo at
startup -- run the "Publish artifacts" cell at the end of
`notebooks/00_export_metadata.ipynb` (in Colab) before deploying, so
`ARTIFACTS_REPO` below actually has something to download.

## Deploy

From the repo root (`app` branch):

```bash
gcloud run deploy recsys-app \
  --source . \
  --region europe-central2 \
  --allow-unauthenticated \
  --set-env-vars ARTIFACTS_REPO=<your-hf-username>/recsys-artifacts \
  --memory 4Gi \
  --cpu 2 \
  --timeout 600 \
  --min-instances 0 \
  --max-instances 3
```

- `--allow-unauthenticated` -- required for students to reach it without a
  Google account/IAM permission. Without this flag, Cloud Run requires
  Google auth on every request.
- `--region` -- pick whatever's closest to you/students; `europe-central2`
  (Warsaw) is a reasonable default. Any Cloud Run region works the same way.
- `--memory 4Gi` / `--cpu 2` -- five models (torch + gensim + numpy) loaded
  at once need real headroom; scale down if cost is a concern and it still
  fits, scale up if the container gets OOM-killed.
- `--timeout 600` -- cold start includes downloading ~4GB before the first
  request can be served; the default 300s request timeout can be too tight
  for that. Bump further (Cloud Run's max is 3600) if deploys or first
  requests still time out.

`gcloud` prints a public Service URL (`https://recsys-app-xxxxx.a.run.app`)
when it finishes -- that's the link to share with students.

## Redeploying after retraining a model

1. Re-run the "Publish artifacts" cell in `00_export_metadata.ipynb` (uploads
   the refreshed `artifacts/` to the same HF Dataset repo).
2. Nothing to redeploy on the Cloud Run side unless `space/main.py` or
   `app/recommenders/*` changed -- existing running/future instances will
   pick up the new artifacts on their next cold start automatically (they
   always re-download at startup, nothing is cached across deploys).
3. If the code *did* change, rerun the same `gcloud run deploy` command
   above -- it rebuilds and replaces the running revision.

## Taking it down after the month

```bash
gcloud run services delete recsys-app --region europe-central2
```

With `min-instances=0` there's essentially no cost while nobody's using it,
but deleting the service is the clean way to be sure, and avoids leaving a
public URL up indefinitely.
