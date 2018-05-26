Title: How to Deploy a Shiny App on Google Kubernetes Engine
Date: 2017-10-02
Tags: gcp, gke, docker, containers, kubernetes, shiny, farkle
Slug: shiny-on-docker
Resources: katex

[Shiny](https://shiny.rstudio.com/) is an awesome tool for building interactive apps powered by R. There are a couple options for [deploying](https://shiny.rstudio.com/deploy/) Shiny apps.  You can deploy to [Shinyapps.io](http://www.shinyapps.io/).  You can also deploy on your own machine using open source Shiny Server.  This tutorial shows how to setup a Docker container for a Shiny app and deploy on Google Kubernetes Engine.  And because deploying the "Hello World" example is entirely unsatisfying, I chose to build an app to help guide strategy for a game I recently played.

## Farkle - A Game of <span style='text-decoration:line-through;'>Guts & Luck</span> Probability

My wife and I recently stumbled upon this game in one of our relative's game closets.  The game is very simple and is played with just 6 dice (how they get people to buy a game that is comprised of only 6 dice is beyond me).  Different rolls are worth different amounts of points. 4 of a kind is worth 1000 points, 1-2-3-4-5-6 is worth 1500 points, three 6s is worth 600 points, etc.  1s and 5s _not_ used in other combinations count as 100 and 50 points respectively.  Detailed scoring rules [here](http://www.smartboxdesign.com/farklerules.html). For instance, a roll of 1-3-3-4-5-3 = 300 + 100 + 50 = 450 points.  

Any dice that don't count towards your score can be rolled again.  However, and this is the catch, if you score no points on a roll, you lose all points accumulated up to that point.  So in the above example, we could choose to roll 1 die (from the non-scoring 4), but we will lose the 450 points that we've banked if we don't roll a 1 or a 5 (the only scoring options with a single die).

Here is a summary of the expected number of points and the probability of scoring zero points on a single roll:
<table class="pretty">
<tr><th>Dice Remaining</th><th>P(Points = 0)</th><th>E[Points]</th></tr>
<tr><td>1</td><td>66.7%</td><td>25.0</td></tr>
<tr><td>2</td><td>44.4%</td><td>50.0</td></tr>
<tr><td>3</td><td>27.8%</td><td>83.6</td></tr>
<tr><td>4</td><td>15.7%</td><td>132.7</td></tr>
<tr><td>5</td><td>7.7%</td><td>203.3</td></tr>
<tr><td>6</td><td>2.3%</td><td>388.5</td></tr>
</table>

As expected, the more dice we roll, the more likely we are to _not_ get zero points.  Furthermore, since more high scoring options are available with each die, each incremental die gives us more expected points than the last.

If you manage to score using all 6 of the dice, you get to start rolling again with all 6 dice. A short-sighted player may observe that <span class="inline-equation" data-expr="\frac{2}{3}*0+\frac{1}{6}*500+\frac{1}{6}*550 = 175 < 450"></span> and thus not be willing to risk rolling that last die.  However, those 500 and 550 scenarios are too low!  The average 6 dice roll results in 0 points just 2.3% of the time and an average of <span class="inline-equation" data-expr="\frac{388.5}{1-0.023} = 397.7"></span> points if not zero.  Incorporating this into our expectation, the average number of points on the next two rolls is
actually <span class="inline-equation" data-expr="\frac{1}{6}*(1 - 0.023)(500+550+2*397.7) = 300.5"></span>.  And this is still conservative.  You have the option to roll a third time after that second roll, when prudent, which will increase the expected number of points further.  Though it still doesn't make sense to roll that last die, it's much closer than it appeared at face value.

In summary, at any point in time, we are making a decision about whether to continue rolling the dice or stop.  The key parameters are (1) how many points we have in the bank and (2) how many dice are remaining.  A good decision will be based not just on what might happen this roll but also on subsequent rolls.  We are going to make a Shiny app to help us make these decisions.

## Building & Deploying Our Shiny App

I started by doing a little up-front work to generate the state space of possible rolls.  This computation is not reactive and only needs to be performed once prior to app initialization.  Next, I created a recursive `play` function which determines the optional strategy (roll or stop) with parameters for how many points have been banked so far and how many dice are remaining.  I gave the function a max recursion depth to limit computation time.  I figured this is okay since (1) turns with a large number of rolls are quite improbable and thus contribute less to our decision making and (2) players become increasingly less likely to continue rolling as they accumulate more points since they have more to lose.  Finally, I made the Shiny app. Building Shiny apps involves laying out inputs, outputs, and the logic that ties them together.  This app is very simple.  Just 26 lines of R!

On GCP, one option for deploying our Shiny app is spinning up a Compute Instance, installing all the necessary software (R, Shiny server, and other necessary R packages), and downloading the code for our app.  A more organized approach is to instead use a container like Docker. Fundamentally, containers allow developers to separate applications from the environments in which they run.  Containers package an application and its dependencies into a single manifest (that can be version controlled) that runs directly on top of an OS kernel.

Firstly, we need to setup a Dockerfile for our Docker image.  I extended [this image](https://hub.docker.com/r/rocker/shiny/) which installs R and Shiny Server.  After that, I simply copy my app into the correct directory and do some initialization.
``` bash
# start with image with R and Shiny server installed
FROM rocker/shiny

# copy files into correct directories
COPY ./shiny/ /srv/shiny-server/farkle/
RUN mv /srv/shiny-server/farkle/shiny-server.conf /etc/shiny-server/shiny-server.conf

# initialize some inputs for the app
WORKDIR /srv/shiny-server/farkle/
RUN mkdir -p data && \
  R -e "install.packages(c('dplyr'), repos='http://cran.rstudio.com/')" && \
  Rscript init.R
```

Next, a few commands to make our Docker image and verify that it works:
``` bash
docker build -t farkle:latest .
docker run --rm -p 3838:3838 farkle:latest # test that it works locally
```

Then tag the image and push it to Google Container Repository:
``` bash
export PROJECT_ID=$(gcloud config get-value project -q)
docker tag farkle gcr.io/${PROJECT_ID}/shiny-farkle:latest
gcloud docker -- push gcr.io/${PROJECT_ID}/shiny-farkle
gcloud container images list-tags gcr.io/${PROJECT_ID}/shiny-farkle
```

Finally, we're going to deploy this image on Google Kubernetes Engine.  I used [Terraform](https://www.terraform.io/) to define and create the GCP infrastructure components for this project: a Kubernetes clusters and a global static IP.  Finally, we apply a Kubernetes manifest containing a deployment for our image and a service, connected to the static IP, to make the service externally accessible.  I packaged my Kubernetes resources in [a Helm chart](https://helm.sh/), which you can use to inject values / variables via template directives (e.g. \{\{ ... \}\}).

``` terraform
terraform apply -var project=${PROJECT_ID}

gcloud container clusters get-credentials shiny-cluster
gcloud config set container/cluster shiny-cluster

helm init
helm install . --set projectId=${PROJECT_ID}
```

Terraform configuration:
``` bash
variable "project" {}

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

resource "google_compute_global_address" "shiny-static-ip" {
  name = "shiny-static-ip"
}

resource "google_container_cluster" "shiny-cluster" {
  name = "shiny-cluster"
  zone = "${var.zone}"
  initial_node_count = "1"
  node_config {
    machine_type = "n1-standard-1"
    oauth_scopes = ["https://www.googleapis.com/auth/devstorage.read_only"]
  }
}
```

Kubernetes manifest:
``` yaml
---
apiVersion: v1
kind: Service
metadata:
  name: shiny-farkle-service
  annotations:
    kubernetes.io/ingress.global-static-ip-name: shiny-static-ip
  labels:
    app: shiny-farkle
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 3838
  selector:
    app: shiny-farkle
---
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  name: shiny-farkle-deploy
  labels:
    app: shiny-farkle
spec:
  replicas: 1
  template:
    metadata:
      labels:
        app: shiny-farkle
    spec:
      containers:
      - name: master
        imagePullPolicy: Always
        image: gcr.io/{{ .Values.projectId }}/shiny-farkle:latest
        ports:
        - containerPort: 3838
```

## Some Final Farkle Insights

The really important question is this: if I have X dice remaining, how many points must I have in the bank to NOT roll?  Using our Shiny app (3 roll look-forward), I estimated these numbers:
<table class="pretty">
<tr><th>Dice Remaining</th><th>Bank Threshold to Stop Rolling</th></tr>
<tr><td>1</td><td>262</td></tr>
<tr><td>2</td><td>225</td></tr>
<tr><td>3</td><td>386</td></tr>
<tr><td>4</td><td>944</td></tr>
<tr><td>5</td><td>2,766</td></tr>
<tr><td>6</td><td>16,785</td></tr>
</table>

The game is played to 10,000 points (with the lagging player getting a rebuttal opportunity). So there is virtually no scenario in which you would not roll 6 dice when given the opportunity!  You can find a link to all of my work [here](https://github.com/donaldrauscher/shiny-farkle) and a link to the app deployed using this methodology [here](https://shiny-farkle-ogzacojzsg.now.sh/)  Cheers!

Note #1: A big simplification that I make on game play is that _all dice that can be scored will be scored_.  In reality, players have the option not to score all dice.  For instance, if I roll three 1s, I can choose to bank one 1 and roll the 5 remaining dice, which, using our app, makes sense.  100 in the bank and 5 remaining dice has a 352.0 expectation; 200 in the bank and 4 dice remaining has a 329.0 expectation.

Note #2: Of course, these estimates are agnostic to the game situation.  In reality, you're trying to maximize your probability of winning, not your expected number of points.  If you are down by 5000 points, you're going to need to be a lot more aggressive.

 \-\-\-

 02-10-2018 Update: I moved hosting of this app to [Now](https://zeit.co) for purely financial reasons.  They provide serverless deployments for Node.js and Docker.  They also provide 3 instances for free!
