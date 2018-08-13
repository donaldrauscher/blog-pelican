Title: Quick and Easy BI: Setting up Redash on GKE
Date: 2017-12-31
Tags: business_intelligence, redash, gke, gcp
Slug: redash-gke

Professionally, I have worked quite a lot with BI platforms Looker and Tableau. They are great BI platforms for an organization, though probably too heavy (and too expensive) for a small project or a bootstrapping startup.  Sometimes you just need something where you can write queries and dump them into a visualization.  Recently, I was looking to implement a lightweight BI tool for a personal project.  I chose to use [Redash](https://redash.io/), which you can [self-host](https://redash.io/help-onpremise/setup/setting-up-redash-instance.html) on your own infrastructure.  This post documents how to set up Redash on Google Cloud using GKE.  Because I am using CloudSQL as the Postgres backend and a persistent drive for Redis, we can delete our cluster when we're not using it and spin it back up as needed, without losing any data!

## Infrastructure Setup

We will use the following Google Cloud components to set up Redash:

* Postgres DB (via CloudSQL)
* Persistent disk for Redis instance
* Kubernetes cluster for Redash Docker image

Here is a [Terraform](https://www.terraform.io) configuration which defines all the necessary infrastructure:
``` terraform
# infrastructure.tf

variable "project" {}

variable "postgres_user" {
  default = "redash"
}
variable "postgres_pw" {
  default = "hsader"
}

variable "region" {
  default = "us-central1"
}

variable "zone" {
  default = "us-central1-f"
}

provider "google" {
  version = "~> 1.4"
  project = "${var.project}"
  region = "${var.region}"
}

resource "google_compute_global_address" "redash-static-ip" {
  name = "redash-static-ip"
}

resource "google_compute_disk" "redash-redis-disk" {
  name  = "redash-redis-disk"
  type  = "pd-ssd"
  size = "200"
  zone  = "${var.zone}"
}

resource "google_sql_database_instance" "redash-db" {
  name = "redash-db"
  database_version = "POSTGRES_9_6"
  region = "${var.region}"
  settings {
    tier = "db-f1-micro"
  }
}

resource "google_sql_database" "redash-schema" {
  name = "redash"
  instance = "${google_sql_database_instance.redash-db.name}"
}

resource "google_sql_user" "proxyuser" {
  name = "${var.postgres_user}"
  password = "${var.postgres_pw}"
  instance = "${google_sql_database_instance.redash-db.name}"
  host = "cloudsqlproxy~%"
}

resource "google_container_cluster" "redash-cluster" {
  name = "redash-cluster"
  zone = "${var.zone}"
  initial_node_count = "1"
  node_config {
    machine_type = "n1-standard-4"
  }
}
```

Create our infrastructure with Terraform and install Helm Tiller on our Kubernetes cluster.  You will also need to create a service account that the CloudSQL proxy on Kubernetes will use.  Create that (Role = "Cloud SQL Client"), download the JSON key, and attach key as secret.

``` bash
export PROJECT_ID=$(gcloud config get-value project -q)
terraform apply -var project=${PROJECT_ID}

gcloud container clusters get-credentials redash-cluster
gcloud config set container/cluster redash-cluster

helm init

kubectl create secret generic cloudsql-instance-credentials \
    --from-file=credentials.json=[PROXY_KEY_FILE_PATH]
```

## Redash Deployment

Next, we need to deploy Redash on our Kubernetes cluster.  I packaged my Kubernetes resources in [a Helm chart](https://helm.sh/), which you can use to inject values / variables via template directives (e.g. \{\{ ... \}\}).  I used a [Helm hook](https://docs.helm.sh/developing_charts/#hooks) to set up the configuration and the database resources (CloudSQL proxy + Redis) and also run a job to initialize the Redash schema before deploying the app.

``` bash
helm install . --set projectId=${PROJECT_ID}
```

Redash resources:
``` yaml
# config.yaml
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: config
  annotations:
    "helm.sh/hook": pre-install
    "helm.sh/hook-weight": "1"
data:
  REDASH_DATABASE_URL: postgresql://{{ .Values.postgres.user }}:{{ .Values.postgres.pw }}@postgres:5432/redash
  REDASH_REDIS_URL: "redis://redis:6379/0"
  PYTHONUNBUFFERED: "0"
  REDASH_LOG_LEVEL: "INFO"
```
``` yaml
# db.yaml
---
kind: Service
apiVersion: v1
metadata:
  name: postgres
  annotations:
    "helm.sh/hook": pre-install
    "helm.sh/hook-weight": "2"
spec:
  type: ClusterIP
  selector:
    app: redash
    tier: postgres
  ports:
    - name: postgres
      port: 5432
---
kind: Service
apiVersion: v1
metadata:
  name: redis
  annotations:
    "helm.sh/hook": pre-install
    "helm.sh/hook-weight": "2"
spec:
  type: ClusterIP
  selector:
    app: redash
    tier: redis
  ports:
    - name: redis
      port: 6379
---
kind: Deployment
apiVersion: extensions/v1beta1
metadata:
  name: postgres
  annotations:
    "helm.sh/hook": pre-install
    "helm.sh/hook-weight": "2"
spec:
  replicas: 1
  template:
    metadata:
      labels:
        app: redash
        tier: postgres
    spec:
      containers:
        - name: cloudsql-proxy
          image: gcr.io/cloudsql-docker/gce-proxy:1.11
          command: ["/cloud_sql_proxy", "--dir=/cloudsql",
                    "-instances={{ .Values.projectId }}:us-central1:redash-db=tcp:0.0.0.0:5432",
                    "-credential_file=/secrets/cloudsql/credentials.json"]
          ports:
            - name: postgres
              containerPort: 5432
          volumeMounts:
            - name: cloudsql-instance-credentials
              mountPath: /secrets/cloudsql
              readOnly: true
            - name: ssl-certs
              mountPath: /etc/ssl/certs
            - name: cloudsql
              mountPath: /cloudsql
      volumes:
        - name: cloudsql-instance-credentials
          secret:
            secretName: cloudsql-instance-credentials
        - name: cloudsql
          emptyDir:
        - name: ssl-certs
          hostPath:
            path: /etc/ssl/certs
---
kind: Deployment
apiVersion: extensions/v1beta1
metadata:
  name: redis
  annotations:
    "helm.sh/hook": pre-install
    "helm.sh/hook-weight": "2"
spec:
  replicas: 1
  template:
    metadata:
      labels:
        app: redash
        tier: redis
    spec:
      containers:
        - name: redis
          image: redis:3.0-alpine
          ports:
            - name: redis
              containerPort: 6379
          volumeMounts:
            - name: redis-disk
              mountPath: /data/redis
      volumes:
        - name: redis-disk
          gcePersistentDisk:
            pdName: redash-redis-disk
            fsType: ext4
```

Redash DB initialization job:
``` yaml
# init.yaml
---
kind: Job
apiVersion: batch/v1
metadata:
  name: init-db
  annotations:
    "helm.sh/hook": pre-install
    "helm.sh/hook-weight": "3"
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: redash-init
          image: redash/redash:latest
          resources:
            requests:
              memory: 1Gi
          envFrom:
            - configMapRef:
                name: config
          args: ["create_db"]
```

Redash deployment:
``` yaml
# app.yaml
---
kind: Service
apiVersion: v1
metadata:
  name: redash
  annotations:
    kubernetes.io/ingress.global-static-ip-name: redash-static-ip
spec:
  type: LoadBalancer
  selector:
    app: redash
    tier: app
  ports:
    - port: 80
      targetPort: 5000
---
kind: Deployment
apiVersion: extensions/v1beta1
metadata:
  name: redash
spec:
  replicas: 1
  template:
    metadata:
      labels:
        app: redash
        tier: app
    spec:
      containers:
        - name: server
          image: redash/redash:latest
          resources:
            requests:
              memory: 1Gi
          ports:
            - containerPort: 5000
          envFrom:
            - configMapRef:
                name: config
          env:
            - name: REDASH_COOKIE_SECRET
              value: {{ .Values.cookieSecret }}
          args: ["server"]
        - name: workers
          image: redash/redash:latest
          resources:
            requests:
              memory: 1Gi
          envFrom:
            - configMapRef:
                name: config
          env:
            - name: WORKERS_COUNT
              value: "{{ .Values.numWorkers }}"
            - name: QUEUES
              value: "queries,scheduled_queries,celery"
          args: ["scheduler"]
```

\-\-\-

You can find all of my code up on my GitHub [here](https://github.com/donaldrauscher/redash-gke).  Cheers!

<img src="/images/redash-example.png" width="885px" style="display:block; margin-left:auto; margin-right:auto;">
