
Building `pelican` Google Cloud Container Builder step:
```bash
gcloud container builds submit --gcs-source-staging-dir=gs://djr-data/cloudbuild --tag gcr.io/${PROJECT_ID}/pelican:latest .
```

Building production blog with Google Cloud Container Builder:
```bash
gcloud container builds submit --gcs-source-staging-dir=gs://djr-data/cloudbuild --config cloudbuild.yaml .
```

Building test blog locally:
```bash
pelican content
gsutil -m rsync -r -c -d ./output gs://test.donaldrauscher.com
```
