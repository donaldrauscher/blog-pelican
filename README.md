
Building `pelican` Google Cloud Container Builder step:
```bash
gcloud container builds submit --gcs-source-staging-dir=gs://djr-data/cloudbuild --tag gcr.io/${PROJECT_ID}/pelican:latest .
```

Building blog:
```bash
gcloud container builds submit --gcs-source-staging-dir=gs://djr-data/cloudbuild --config cloudbuild.yaml .
```
