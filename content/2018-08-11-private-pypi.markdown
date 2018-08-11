Title: Two Options for Hosting a Private PyPI Repository
Date: 2018-08-11
Tags: python pypi gcp cloud_build git_ops ci_cd
Slug: private-pypi

A few years back, I read [an interesting post](https://medium.com/airbnb-engineering/using-r-packages-and-education-to-scale-data-science-at-airbnb-906faa58e12d) about how Airbnb's data science team developed their own internal R package, Rbnb, to standardize solutions to common problems and reduce redundancy across projects.  I really like this idea and have implemented a similar solution for Python at places that I have worked.  This post details two options for hosting a private Python package, both of which leverage [Google Cloud Build](https://cloud.google.com/cloud-build/) for CI/CD.

### Option #1 - Gemfury

[Gemfury](https://gemfury.com/) is a cloud package respository that you can use to host both public and private packages for Python (and lots of other languages).  [Some useful instructions](https://gemfury.com/help/pypi-server/) for how to upload Python packages to Gemfury and install them with `pip`.  The following Cloud Build pipeline will, on tagged commits, download the package from Google Cloud Repositories, run tests, package, and curl to Gemfury:

```yaml
steps:
  - name: gcr.io/cloud-builders/gcloud
    args: ['source', 'repos', 'clone', '${_PACKAGE}', '--project=${PROJECT_ID}']
  - name: gcr.io/cloud-builders/git
    args: ['checkout', '${TAG_NAME}']
    dir: '/workspace/${_PACKAGE}'
  - name: gcr.io/${PROJECT_ID}/python-packager:latest
    entrypoint: 'bash'
    args: ['-c', 'pip3 install -e . && python3 -m pytest -s']
    dir: '/workspace/${_PACKAGE}'
  - name: gcr.io/${PROJECT_ID}/python-packager:latest
    args: ['setup.py', 'sdist']
    dir: '/workspace/${_PACKAGE}'
  - name: gcr.io/cloud-builders/curl
    entrypoint: 'bash'
    args: ['-c', 'curl -f -F package=@dist/${_PACKAGE}-${TAG_NAME}.tar.gz https://$${FURY_TOKEN}@push.fury.io/${_FURY_USER}/']
    secretEnv: ['FURY_TOKEN']
    dir: '/workspace/${_PACKAGE}'
secrets:
- kmsKeyName: projects/blog-180218/locations/global/keyRings/djr/cryptoKeys/fury
  secretEnv:
    FURY_TOKEN: CiQAUrbjD9VjSHPnmMvLV0Jv+duPGyuaIgS0C2u1LmcVRGHY/BwSPQCP7mNtRVGShanmgHUx5RHoohNDGWX4FnscAmbMBVplms0uOQfHLmLy/wkfaxAHYoK2pX/LKDxDIwQzAz0=
substitutions:
  _PACKAGE: djr-py
  _FURY_USER: donaldrauscher
```

NOTE: Need to create a KMS key/keyring, give Cloud Build access to it, and use that key to encrypt your Fury token. You can find additional instructions on how to do this [here](https://cloud.google.com/cloud-build/docs/securing-builds/use-encrypted-secrets-credentials).
```
echo -n ${FURY_TOKEN} | gcloud kms encrypt --plaintext-file=- --ciphertext-file=- --location=global --keyring=djr --key=fury | base64
```

### Option #2 - GCS Bucket

If you don't care about restricting which people can access your package (I clearly do not), then you can host a simple PyPI respository on a GCS bucket using [`dumb-pypi`](https://github.com/chriskuehl/dumb-pypi).  First, you will need to [set up a GCS bucket](https://cloud.google.com/storage/docs/hosting-static-website) where you can host a static site.  This Cloud Build pipeline uploads the package to GCS and triggers a [*second* Cloud Build pipeline](https://github.com/donaldrauscher/gcs-pypi) which rebuilds the PyPI repository on the specified GCS bucket.

```yaml
steps:
  - name: gcr.io/cloud-builders/git
    args: ['clone', '-b', '${TAG_NAME}', '--single-branch', '--depth', '1', 'https://github.com/${_GITHUB_USER}/${_PACKAGE}.git']
  - name: gcr.io/${PROJECT_ID}/python-packager:latest
    entrypoint: 'bash'
    args: ['-c', 'pip3 install -e . && python3 -m pytest -s']
    dir: '/workspace/${_PACKAGE}'
  - name: gcr.io/${PROJECT_ID}/python-packager:latest
    args: ['setup.py', 'sdist']
    dir: '/workspace/${_PACKAGE}'
  - name: gcr.io/cloud-builders/gsutil
    args: ['cp', 'dist/${_PACKAGE}-${TAG_NAME}.tar.gz', 'gs://${_BUCKET}/raw/']
    dir: '/workspace/${_PACKAGE}'
  - name: gcr.io/cloud-builders/git
    args: ['clone', 'https://github.com/donaldrauscher/gcs-pypi.git']
  - name: gcr.io/cloud-builders/gcloud
    args: ['builds', 'submit', '--config', 'cloudbuild.yaml', '--no-source', '--async', '--substitutions', '_BUCKET=${_BUCKET}']
    dir: '/workspace/gcs-pypi'
substitutions:
  _PACKAGE: djr-py
  _BUCKET: pypi.donaldrauscher.com
  _GITHUB_USER: donaldrauscher
```

===

NOTE: Both of these Cloud Build jobs require a `python-packager` [custom Cloud Build step](https://cloud.google.com/cloud-build/docs/create-custom-build-steps).  This is a simple Docker container with some Python utilities:
```Dockerfile
FROM gcr.io/cloud-builders/gcloud

RUN apt-get update \
  && apt-get install -y python3-pip \
  && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip setuptools wheel pylint pytest

ENTRYPOINT ["python3"]
```

I used option #2 to host my personal Python package ([djr-py](https://github.com/donaldrauscher/djr-py)) on [http://pypi.donaldrauscher.com/](http://pypi.donaldrauscher.com/).  Enjoy!