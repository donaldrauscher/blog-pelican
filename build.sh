#!/usr/bin/env bash

function usage(){
  echo "usage: build.sh (cloud|local) (prod|test)"
  echo "This builds the blog locally or via Google Cloud Container Builder."
  echo "Pushes to gs://www.donaldrauscher.com for production and"
  echo "gs://test.donaldrauscher.com for test."
}

if [[ $# -ne 2 ]]; then
  usage
elif [[ $1 == "cloud" && $2 == "prod" ]]; then
  gcloud builds submit --gcs-source-staging-dir=gs://djr-data/cloudbuild \
    --config cloudbuild.yaml .
elif [[ $1 == "cloud" && $2 == "test" ]]; then
  gcloud builds submit --gcs-source-staging-dir=gs://djr-data/cloudbuild \
    --config cloudbuild.yaml --substitutions=_BRANCH=test,_SUB_DOMAIN=test,_ENV=test .
elif [[ $1 == "local" && $2 == "prod" ]]; then
  pelican content
  gsutil -m rsync -r -c -d ./output gs://www.donaldrauscher.com
elif [[ $1 == "local" && $2 == "test" ]]; then
  pelican content
  gsutil -m rsync -r -c -d ./output gs://test.donaldrauscher.com
else
  usage
fi