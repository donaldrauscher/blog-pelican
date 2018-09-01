Title: Classifying Movie Reviews with TensorFlow
Date: 2018-08-29
Tags: tensorflow, cloud-ml, nlp
Slug: movie-reviews-tf
Resources: jquery

Recently, I've been having a lot of fun with Tensorflow!  Here I'm building a DNN classifier with TF for classifying movie reviews as positive or negative.  The data source is the [ACL 2011 IMDB dataset](http://ai.stanford.edu/~amaas/data/sentiment/).  I used a [custom estimator](https://www.tensorflow.org/guide/custom_estimators) so that I could implement cosine annealing for learning rate decay. 

I used [Cloud ML Engine](https://cloud.google.com/ml-engine/docs/tensorflow/deploying-models) to deploy the model on GCP.  I also used a Cloud Function to make the model accessible via a simple HTTP function.  Give it a try!

<form id="movie_reviews">
<textarea name="review" cols="50" rows="3" placeholder="action packed but ultimately devoid of substance"></textarea>
<br />
<input type="submit" value="Classify" />
<span id="review_class"></span>
</form>

<script type="text/javascript">
$(document).ready(function() {
    $('form#movie_reviews').submit(function(event) {
        var formData = {
            'model': 'movie_reviews',
            'version': 'v2',
            'instances': [$('textarea[name=review]').val()]
        };
        
        $("span#review_class").html('<img src="/theme/images/ajax-loader.gif" alt="Loading..." />');

        $.ajax({
            type: 'POST',
            url: 'https://us-central1-blog-180218.cloudfunctions.net/ml_predict',
            data: JSON.stringify(formData),
            dataType: 'json',
            contentType: 'application/json',
            crossDomain: true,
            success: function(data){
                max_class = data[0]['class'];
                if (max_class == 1) {
                    $("span#review_class").html("Positive").css('color', '#006400');
                } else {
                    $("span#review_class").html("Negative").css('color', '#B22222');
                }
            }
        })

        event.preventDefault();
    });
});
</script>

NOTE: I used a version of this dataset that had already been preprocessed into TFRecords.  As part of this preprocessing, the reviews were lowercases and split into words; punctuation, including "'", were treated as seperate words.  To properly serve this model, we need to replicate this preprocessing in the [serving input receiver](https://www.tensorflow.org/guide/saved_model#prepare_serving_inputs).  Unfortunately, I couldn't find a way to do this with native TF ops!  I was able to replicate it with `tf.py_func`.  However, [a documentated limitation](https://www.tensorflow.org/api_docs/python/tf/py_func) of `py_func` is that it is *not* serialized in the GraphDef, so it cannot be used for serving, which requires serializing the model and restoring in a different environment.  Here, I'm only doing a simple word split by spaces.


Here's a [link](https://github.com/donaldrauscher/movie-reviews-tf) to all the code for the model build and a [link](https://github.com/donaldrauscher/blog-pelican/tree/master/functions/ml_predict) to the Cloud Function for serving. Cheers!

===


## Build TF Estimator

```python
# set up feature columns
terms_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(key='terms', 
                                                                                 vocabulary_list=vocab)

terms_embedding_column = tf.feature_column.embedding_column(terms_feature_column, dimension=10)
feature_columns = [terms_embedding_column]

# create estimator spec
def make_model(features, labels, mode):

    # build graph
    net = tf.feature_column.input_layer(features, feature_columns)
    net = tf.layers.dense(net, units=10, activation=tf.nn.leaky_relu)
    net = tf.layers.dropout(net, rate=0.3, training=(mode == tf.estimator.ModeKeys.TRAIN))
    net = tf.layers.dense(net, units=10)
    logits = tf.layers.dense(net, 2)
    
    # compute predictions
    predicted_classes = tf.argmax(logits, 1)
    predicted_probs = tf.nn.softmax(logits)
    
    # generate predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class': predicted_classes,
            'prob': predicted_probs
        }
        
        export_outputs = {
          'predict': tf.estimator.export.PredictOutput(outputs=predictions)
        }
        
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    # compute loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # create training op with cosine annealing for learning rate
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        
        learning_rate = tf.train.cosine_decay(learning_rate=0.2, global_step=global_step, alpha=0.05, decay_steps=10000)
        
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
        
        train_op = optimizer.minimize(loss, global_step=global_step)
        
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    # compute evaluation metrics
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predicted_classes),
        'auc': tf.metrics.auc(labels=labels, predictions=predicted_probs[:, 1])
    }
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)


# create estimator
classifier = tf.estimator.Estimator(model_fn=make_model)

# train and evaluate
classifier.train(input_fn=lambda: input_fn([train_path], num_epochs=10))
test_stats = classifier.evaluate(input_fn=lambda: input_fn([test_path], num_epochs=1))

# export
def serving_input_receiver_fn():
    reviews = tf.placeholder(dtype=tf.string, shape=(None), name='reviews')
    terms = tf.sparse_tensor_to_dense(tf.string_split(reviews), default_value='')
    return tf.estimator.export.ServingInputReceiver({'terms': terms}, {'reviews': reviews})

export_path = classifier.export_savedmodel(export_dir_base='exports',
                                           serving_input_receiver_fn=serving_input_receiver_fn)

export_path = export_path.decode('utf-8')
```


<br />
## Model Results

<table class="pretty" style="margin-left: 0px;">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Train</th>
      <th>Test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Accuracy</th>
      <td>0.918600</td>
      <td>0.870160</td>
    </tr>
    <tr>
      <th>AUC</th>
      <td>0.971972</td>
      <td>0.942018</td>
    </tr>
    <tr>
      <th>Loss</th>
      <td>0.214713</td>
      <td>0.321892</td>
    </tr>
  </tbody>
</table>


<br />
## Upload to Cloud ML Engine

``` bash
#!/bin/bash

MODEL_NAME=movie_reviews
MODEL_VERSION=v2
MODEL_TIMESTAMP=$(ls -t exports/ | head -1)

DEPLOYMENT_SOURCE=gs://djr-data/movie-reviews

gsutil rsync -c -d -r exports/$MODEL_TIMESTAMP $DEPLOYMENT_SOURCE

#gcloud ml-engine models create $MODEL_NAME

gcloud ml-engine versions create $MODEL_VERSION --model $MODEL_NAME --origin $DEPLOYMENT_SOURCE \
    --python-version 3.5 --runtime-version 1.9
```

NOTE: Make sure the Python environment in which you build your model matches [the serving environment in Cloud ML](https://cloud.google.com/ml-engine/docs/tensorflow/runtime-version-list)!  


<br />
## Expose Model with a Cloud Function

```python
# gets predictions from cloud ml engine
def ml_predict(request):
    import flask
    import json
    import googleapiclient.discovery
    import google.auth

    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST',
        'Access-Control-Allow-Headers': 'Content-Type'
    }

    # handle pre-flight options request
    if request.method == 'OPTIONS':
        return flask.make_response(('', 204, headers))

    _, project = google.auth.default()

    request_json = request.get_json()

    model = request_json['model']
    version = request_json['version']
    instances = request_json['instances']

    service = googleapiclient.discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}/versions/{}'.format(project, model, version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return flask.make_response((
        json.dumps(response['predictions']),
        200,
        headers
    ))
```
