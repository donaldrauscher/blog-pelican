#!/bin/bash
gcloud beta functions deploy ml_predict --runtime python37 --trigger-http --stage-bucket=gs://djr_cloud_functions/
