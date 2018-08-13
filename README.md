
Building `pelican` Google Cloud Container Builder step:
```bash
gcloud container builds submit --gcs-source-staging-dir=gs://djr-data/cloudbuild --tag gcr.io/${PROJECT_ID}/pelican:latest .
```

Building production blog with Google Cloud Container Builder:
```bash
source build.sh cloud prod
```

Building locally and pushing to test bucket:
```bash
source build.sh local test
```
