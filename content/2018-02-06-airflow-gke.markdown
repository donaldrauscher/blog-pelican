Title: Setting up Apache Airflow on GKE
Date: 2018-02-06
Tags: etl, pipelining, airflow, gke, gcp
Slug: airflow-gke

Historically, I have used [Luigi](https://luigi.readthedocs.io/en/latest/) for a lot of my data pipelining.  Recently, however, I have started experimenting with [Airflow](https://airflow.apache.org/) for [a variety of reasons](https://www.quora.com/Which-is-a-better-data-pipeline-scheduling-platform-Airflow-or-Luigi).  Some things I really like about Airflow:
- **Easier to parallize** - Luigi can only be scaled *locally*.  You can create multiple worker threads by passing `--workers N` when kicking off a job, but you cannot parallelize Luigi jobs across multiple machines!  Airflow parallelizes quite well.  For instance, you can use [Celery](https://airflow.apache.org/configuration.html#scaling-out-with-celery) to scale out your workers.
- **Superior scheduler** - The Luigi "central scheduler" is a bit of a misnomer; it doesn't actually schedule anything!  Its main purpose is to prevent worker threads from running the same task concurrently. That's it. You still need to initiate Luigi jobs with a cronjob.  The Airflow scheduler is *much* more useful. You can use it to set up a cronjob-like schedule for a DAG and even initiate retries following errors.  
- **Connection management** - Airflow has a nice mechanism for organizing [connections](https://airflow.apache.org/concepts.html#connections) to your resources.  This is really useful, especially in a multiuser environment.  It allows you to avoid storing secrets in .gitignore'd config files all over the place.
- **Better ongoing support** - Luigi, originally open sourced at Spotify, is currently maintained on a ["for fun basis"](https://github.com/tarrasch) by Arash Rouhani, who currently works at Google.  Meanwhile, Airflow, originally open sourced at Airbnb, is being incubated by Apache.  

Given that I have been on a Docker/Kubernetes kick of-late, I decided to spend some time setting up Airflow on GKE.  I leveraged [an awesome Docker image with Airflow](https://github.com/puckel/docker-airflow) from Matthieu Roisil.  I used a Postgres instance on CloudSQL for the Airflow meta database and Redis as the Celery backend. Also used [git-sync](https://github.com/kubernetes/git-sync) sidecar container to continuously sync DAGs and plugins on running cluster, so you only need to rebuild the Docker image when changing the Python environment!  Finally, I used Terraform for managing all my GCP infrastructure.

## Terraform Configuration

``` terraform
# infrastructure.tf

variable "project" {}

variable "postgres_user" {
  default = "airflow"
}
variable "postgres_pw" {
  default = "airflow"
}

variable "region" {
  default = "us-central1"
}

variable "zone" {
  default = "us-central1-f"
}

provider "google" {
  version = "~> 1.5"
  project = "${var.project}"
  region = "${var.region}"
}

resource "google_compute_global_address" "airflow-static-ip" {
  name = "airflow-static-ip"
}

resource "google_compute_disk" "airflow-redis-disk" {
  name  = "airflow-redis-disk"
  type  = "pd-ssd"
  size = "200"
  zone  = "${var.zone}"
}

resource "google_sql_database_instance" "airflow-db" {
  name = "airflow-db"
  database_version = "POSTGRES_9_6"
  region = "${var.region}"
  settings {
    tier = "db-g1-small"
  }
}

resource "google_sql_database" "airflow-schema" {
  name = "airflow"
  instance = "${google_sql_database_instance.airflow-db.name}"
}

resource "google_sql_user" "proxyuser" {
  name = "${var.postgres_user}"
  password = "${var.postgres_pw}"
  instance = "${google_sql_database_instance.airflow-db.name}"
  host = "cloudsqlproxy~%"
}

resource "google_container_cluster" "airflow-cluster" {
  name = "airflow-cluster"
  zone = "${var.zone}"
  initial_node_count = "1"
  node_config {
    machine_type = "n1-standard-4"
    oauth_scopes = ["https://www.googleapis.com/auth/devstorage.read_only"]
  }
}
```

## Kubernetes Manifest

Note: I packaged all Kubernetes resources in [a Helm chart](https://helm.sh/).  Helm has several features (e.g. named templates, value substitutions) that allow you write your Kubernetes manifests in a more DRY way.

``` yaml
# config.yaml
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: config-airflow
data:
  EXECUTOR: Celery
  POSTGRES_USER: airflow
  POSTGRES_DB: airflow
  POSTGRES_HOST: postgres
  POSTGRES_PORT: "5432"
  REDIS_HOST: redis
  REDIS_PORT: "6379"
  FLOWER_PORT: "5555"
  {{- if .Values.fernetKey }}
  FERNET_KEY: {{ .Values.fernetKey }}
  {{- end }}
  AIRFLOW__CORE__DAGS_FOLDER: "/git/git/dags/"
  AIRFLOW__CORE__PLUGINS_FOLDER: "/git/git/plugins/"
  AIRFLOW__CORE__LOAD_EXAMPLES: "0"
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: config-git-sync
data:
  GIT_SYNC_REPO: {{ .Values.dagRepo }}
  GIT_SYNC_DEST: git
```

``` yaml
# db.yaml
---
kind: Deployment
apiVersion: extensions/v1beta1
metadata:
  name: postgres
spec:
  replicas: 1
  template:
    metadata:
      labels:
        app: airflow
        tier: postgres
    spec:
      restartPolicy: Always
      containers:
        - name: cloudsql-proxy
          image: gcr.io/cloudsql-docker/gce-proxy:1.11
          command: ["/cloud_sql_proxy", "--dir=/cloudsql",
                    "-instances={{ .Values.projectId }}:us-central1:airflow-db=tcp:0.0.0.0:5432",
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
spec:
  replicas: 1
  template:
    metadata:
      labels:
        app: airflow
        tier: redis
    spec:
      restartPolicy: Always
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
            pdName: airflow-redis-disk
            fsType: ext4
```

``` yaml
# ingress.yaml
---
apiVersion: extensions/v1beta1
kind: Ingress
metadata:
  name: ingress
  annotations:
    kubernetes.io/ingress.global-static-ip-name: airflow-static-ip
    kubernetes.io/tls-acme: "true"
spec:
  tls:
  - secretName: airflow-tls
    hosts:
    - web.{{ .Values.domain }}
    - flower.{{ .Values.domain }}
  rules:
  - host: web.{{ .Values.domain }}
    http:
      paths:
      - backend:
          serviceName: web
          servicePort: 8080
  - host: flower.{{ .Values.domain }}
    http:
      paths:
      - backend:
          serviceName: flower
          servicePort: 5555
```

``` yaml
# service.yaml
---
  apiVersion: v1
  kind: Service
  metadata:
    name: web
  spec:
    type: NodePort
    selector:
      app: airflow
      tier: web
    ports:
      - name: web
        port: 8080
  ---
  apiVersion: v1
  kind: Service
  metadata:
    name: flower
  spec:
    type: NodePort
    selector:
      app: airflow
      tier: flower
    ports:
      - name: flower
        port: 5555
  ---
  kind: Service
  apiVersion: v1
  metadata:
    name: postgres
  spec:
    type: ClusterIP
    selector:
      app: airflow
      tier: postgres
    ports:
      - name: postgres
        port: 5432
        protocol: TCP
  ---
  kind: Service
  apiVersion: v1
  metadata:
    name: redis
  spec:
    type: ClusterIP
    selector:
      app: airflow
      tier: redis
    ports:
      - name: redis
        port: 6379
```

``` yaml
# deploy.yaml
{{- define "airflow" -}}
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  name: {{ .name }}
spec:
  replicas: {{ .replicas | default 1 }}
  template:
    metadata:
      labels:
        app: airflow
        tier: {{ .name }}
    spec:
      restartPolicy: Always
      containers:
        - name: web
          image: gcr.io/{{ .projectId }}/airflow-gke:latest
          ports:
            - name: web
              containerPort: 8080
          volumeMounts:
          - name: dagbag
            mountPath: /git
          envFrom:
          - configMapRef:
              name: config-airflow
          {{- if eq .name "web" }}
          livenessProbe:
            httpGet:
              path: /
              port: 8080
            initialDelaySeconds: 60
            timeoutSeconds: 30
          readinessProbe:
            httpGet:
              path: /
              port: 8080
            initialDelaySeconds: 60
            timeoutSeconds: 30
          {{- end }}
          command: ["/entrypoint.sh"]
          args:  {{ .commandArgs }}
        - name: git-sync
          image: gcr.io/google_containers/git-sync:v2.0.4
          volumeMounts:
          - name: dagbag
            mountPath: /git
          envFrom:
          - configMapRef:
              name: config-git-sync
      volumes:
        - name: dagbag
          emptyDir: {}
{{- end -}}

---
{{- $_ := set .Values.web "projectId" .Values.projectId }}
{{- template "airflow" .Values.web }}
---
{{- $_ := set .Values.scheduler "projectId" .Values.projectId }}
{{- template "airflow" .Values.scheduler }}
---
{{- $_ := set .Values.workers "projectId" .Values.projectId }}
{{- template "airflow" .Values.workers }}
---
{{- $_ := set .Values.flower "projectId" .Values.projectId }}
{{- template "airflow" .Values.flower }}
```

## Deploy Instructions

(1) Store project id and Fernet key as env variables; create SSL cert / key

``` bash
export PROJECT_ID=$(gcloud config get-value project -q)

if [ ! -f '.keys/fernet.key' ]; then
  export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
  echo $FERNET_KEY > .keys/fernet.key
else
  export FERNET_KEY=$(cat .keys/fernet.key)
fi
```

(2) Create Docker image and upload to Google Container Repository

``` bash
docker build -t airflow-gke:latest .
docker tag airflow-gke gcr.io/${PROJECT_ID}/airflow-gke:latest
gcloud docker -- push gcr.io/${PROJECT_ID}/airflow-gke
```

(3) Create infrastructure with Terraform

Note: You will also need to create a Service Account for the CloudSQL proxy in Kubernetes.  Create that (Role = "Cloud SQL Client"), download the JSON key, and attach as secret.  Stored in `.keys/airflow-cloudsql.json` in this example.

``` bash
terraform apply -var project=${PROJECT_ID}

gcloud container clusters get-credentials airflow-cluster
gcloud config set container/cluster airflow-cluster

kubectl create secret generic cloudsql-instance-credentials \
  --from-file=credentials.json=.keys/airflow-cloudsql.json
```

(4) Set up Helm / Kube-Lego for TLS

Note: You only need to set up [kube-lego](https://github.com/jetstack/kube-lego) if you want to set up TLS using [Let's Encrypt](https://letsencrypt.org/). I only set up HTTPS because I secured my instance with [Cloud IAP](https://cloud.google.com/iap/), which requires a HTTPS load balancer.  

``` bash
kubectl create serviceaccount -n kube-system tiller
kubectl create clusterrolebinding tiller-binding --clusterrole=cluster-admin --serviceaccount kube-system:tiller
helm init --service-account tiller

kubectl create namespace kube-lego

helm install \
  --namespace kube-lego \
  --set config.LEGO_EMAIL=donald.rauscher@gmail.com \
  --set config.LEGO_URL=https://acme-v01.api.letsencrypt.org/directory \
  --set config.LEGO_DEFAULT_INGRESS_CLASS=gce \
  stable/kube-lego
```

(5) Deploy with Kubernetes

``` bash
helm install . \
  --set projectId=${PROJECT_ID} \
  --set fernetKey=${FERNET_KEY}
```

## Test Pipeline

The example pipeline (`citibike.py`) streams data from [this Citibike API](https://gbfs.citibikenyc.com/gbfs/en/station_status.json) into Google BigQuery.  I had a lot of issues with the GCP contrib classes in Airflow ([BQ hook](https://github.com/apache/incubator-airflow/blob/master/airflow/contrib/hooks/bigquery_hook.py) did not support BQ streaming, [base GCP hook](https://github.com/apache/incubator-airflow/blob/master/airflow/contrib/hooks/gcp_api_base_hook.py) based on now-deprecated `oauth2client` library instead of `google-auth`) so I built my own plugin!

Note: To run Citibike example pipeline, will need to create a Service Account with BigQuery access and add to the `google_cloud_default` connection in the Airflow UI.

\-\-\-

Overall, I'm really excited to start using Airflow for more of my data pipelining.  Here is a [link](https://github.com/donaldrauscher/airflow-gke) to all my code on GitHub.  Cheers!
