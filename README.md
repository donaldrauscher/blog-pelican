
Building production blog with Google Cloud Build:
```bash
source build.sh cloud prod
```

Building locally and pushing to test bucket:
```bash
source build.sh local test
```

Building `pelican` Google Cloud Build custom step:
```bash
gcloud builds submit --gcs-source-staging-dir=gs://djr-data/cloudbuild --tag gcr.io/${PROJECT_ID}/pelican:latest .
```
