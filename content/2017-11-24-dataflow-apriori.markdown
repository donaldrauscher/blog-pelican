Title: Identifying Frequent Item Sets using Apache Beam/Dataflow
Date: 2017-11-24
Tags: gcp, dataflow, serverless, apache_beam, apriori, association_rules
Slug: dataflow-apriori
Resources: katex

I have used Google's **serverless DW service**, BigQuery, for several of my projects this past year.  I recently started familiarizing myself with with Google's **serverless data pipeline service**, DataFlow.  This post shows how to build a pipeline to identify frequently purchased item sets in [market basket data](https://www.kaggle.com/c/instacart-market-basket-analysis) from Instacart (3.2M orders, 32M total items purchased, 50K unique items purchased).

## What is Apache Beam?

Apache Beam is a programming model for building and executing data processing pipelines.  Pipelines are built using one of the Beam's supported SDKs (Java, Python) and executed using one of Beam's supported runners (Spark, Flink, Dataflow on GCP).  Pipelines themselves are typically just MapReduce operations: filtering, transforming, grouping + aggregating, etc.

Streaming data processing and batch data processing are often treated as distinctly different things.  Streaming = unbounded, low latency, low accuracy.  Batch = bounded, high latency, high accuracy.  Nathan Marz's popular [Lambda Architecture](http://nathanmarz.com/blog/how-to-beat-the-cap-theorem.html) calls for both, a "correct" batch layer with historical data and an "approximate" real-time layer with recent data.  However, the merits of this architecture are increasingly [being challenged](http://radar.oreilly.com/2014/07/questioning-the-lambda-architecture.html).  

Beam is built on the idea that streaming data processing is really a superset of batch data processing. Beam has [native features](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-102) (e.g. windowing, watermarking, triggering, accumulation rules) to handle one of the biggest challenges of streaming data sources, the skew between event time and processing time.  A common execution pattern in GCP is to use Pub/Sub to capture events and Dataflow to process those events and ingest into BigQuery, GCS, or back into Pub/Sub.

This post will build a batch pipeline on a bounded data source, and, as such, will not showcase a lot of the great features Beam has for streaming data processing!

## Apriori Algorithm

I wanted to do an analysis to identify association rules, e.g. when purchasing A, also purchase B.  I used the [Apriori algorithm](https://en.wikipedia.org/wiki/Apriori_algorithm) which is completed in two steps: (1) identify frequent item sets with specified support and (2) identify association rules with specified confidence.  Support and confidence definitions:
<div class="equation" data-expr="supp(X) = \frac{|t \in T; X \subseteq T|}{|T|}"></div>
<div class="equation" data-expr="conf(X \Rightarrow Y) = \frac{supp(X \cup Y)}{supp(X)}"></div>

Apriori looks for increasingly larger item sets comprised of items from smaller item sets. This works because of the monotonicity property, which states that if <span class="inline-equation" data-expr="X"></span> is frequent then any subset <span class="inline-equation" data-expr="X \subseteq Y"></span> is also frequent.

<img src="{filename}images/apriori.png" style="display:block; margin-left:auto; margin-right:auto;">

Source: [http://infolab.stanford.edu/~ullman/mmds/ch6.pdf](http://infolab.stanford.edu/~ullman/mmds/ch6.pdf)

## Market Basket Analysis Results

I used 5K orders (~0.1% of total orders) as my minimum support cutoff.  For simplicity, I only tested for frequent item sets of sizes 1-3.  Dataflow ran the pipeline in ~16 minutes, autoscaling up to 3 nodes for the majority of the job.  Here is a picture of my Dataflow pipeline, rotated for convenience:
<img src="{filename}images/dataflow_dag.png" style="display:block; margin-left:auto; margin-right:auto;">

I identified 1,094 frequent single items, 883 frequent item pairs, and 53 frequent item triples.  From that, I derived 45 association rules.  Here are the top 10 rules ranked in terms of confidence:
<table class="pretty">
<tr><th>LHS</th><th>LHS</th><th>Support</th><th>Confidence</th></tr>
<tr><td>Organic Raspberries, Organic Hass Avocado</td><td>Bag of Organic Bananas</td><td>11,409</td><td>44.2%</td></tr>
<tr><td>Non Fat Raspberry Yogurt</td><td>Icelandic Style Skyr Blueberry Non-fat Yogurt</td><td>7,224</td><td>44.1%</td></tr>
<tr><td>Organic Large Extra Fancy Fuji Apple, Organic Hass Avocado</td><td>Bag of Organic Bananas</td><td>5,804</td><td>43.5%</td></tr>
<tr><td>Apple Honeycrisp Organic, Organic Hass Avocado</td><td>Bag of Organic Bananas</td><td>6,650</td><td>42.9%</td></tr>
<tr><td>Organic Avocado, Cucumber Kirby</td><td>Banana</td><td>6,594</td><td>42.4%</td></tr>
<tr><td>Strawberries, Organic Avocado</td><td>Banana</td><td>5,290</td><td>41.5%</td></tr>
<tr><td>Total 2% Lowfat Greek Strained Yogurt with Peach</td><td>Total 2% with Strawberry Lowfat Greek Strained Yogurt</td><td>8,014</td><td>40.3%</td></tr>
<tr><td>Organic Whole Milk, Organic Avocado</td><td>Banana</td><td>5,190</td><td>39.7%</td></tr>
<tr><td>Bartlett Pears</td><td>Banana</td><td>13,682</td><td>38.6%</td></tr>
<tr><td>Organic Cucumber, Organic Hass Avocado</td><td>Bag of Organic Bananas</td><td>6,733</td><td>38.6%</td></tr>
</table>

You can find the code for these pipelines in my repo [here](https://github.com/donaldrauscher/instacart-apriori).  Cheers!
