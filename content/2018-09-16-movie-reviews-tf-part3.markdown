Title: Building and Deploying a Deep Learning Model Part 3: Deploying a Serverless Microservice
Date: 2018-09-16
Tags: tensorflow, cloud-ml-engine, cloud-functions
Slug: movie-reviews-tf-part3
Resources: jquery

This is part 3 in a 3-part series ([part 1](./movie-reviews-tf-part2.html), [part 2](./movie-reviews-tf-part3.html)) on building and deploying a deep learning model for the popular [ACL 2011 IMDB dataset](http://ai.stanford.edu/~amaas/data/sentiment/).  In this part, I use [Cloud ML Engine](https://cloud.google.com/ml-engine/docs/tensorflow/deploying-models) to deploy the model on GCP.  I also use a Cloud Function to make the model accessible via a simple HTTP function.  Give it a try!


<form id="movie_reviews">
<textarea name="review" cols="50" rows="3">kept me on the edge of my seat</textarea>
<br />
<input type="submit" value="Classify" />
<span id="review_class"></span>
</form>

<script type="text/javascript">
$(document).ready(function() {
    $('form#movie_reviews').submit(function(event) {
        var formData = {
            'version': 'v5',
            'instances': [$('textarea[name=review]').val()]
        };
        
        $("span#review_class").html('<img src="/theme/images/ajax-loader.gif" alt="Loading..." />');

        $.ajax({
            type: 'POST',
            url: 'https://us-central1-blog-180218.cloudfunctions.net/classify_movie_reviews',
            data: JSON.stringify(formData),
            dataType: 'json',
            contentType: 'application/json',
            crossDomain: true,
            success: function(data){
                max_class = data[0]['class'];
                if (max_class == 1) {
                    $("span#review_class").html("Positive").css('color', '#006400');
                } else if (max_class == 0) {
                    $("span#review_class").html("Negative").css('color', '#B22222');
                } else {
                    $("span#review_class").html("Error").css('color', '#AAAAAA');
                }
            },
            error: function(data){
                $("span#review_class").html("Error").css('color', '#AAAAAA');
            }
        })

        event.preventDefault();
    });
});
</script>

===

### Upload Model to Cloud ML Engine

``` bash
#!/bin/bash

MODEL_NAME=movie_reviews
MODEL_VERSION=v1
MODEL_TIMESTAMP=$(ls -t exports/ | head -1)

DEPLOYMENT_SOURCE=gs://djr-data/movie-reviews

gsutil rsync -c -d -r exports/$MODEL_TIMESTAMP $DEPLOYMENT_SOURCE

gcloud ml-engine models create $MODEL_NAME

gcloud ml-engine versions create $MODEL_VERSION --model $MODEL_NAME --origin $DEPLOYMENT_SOURCE \
    --python-version 2.7 --runtime-version 1.9
```

NOTE: Make sure the Python environment in which you build your model matches [the serving environment in Cloud ML](https://cloud.google.com/ml-engine/docs/tensorflow/runtime-version-list)!  


### Expose Model with a Cloud Function

```python
# gets predictions from cloud ml engine
def classify_movie_reviews(request):
    import flask
    import json
    import re
    import math
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

    # this pulls out our proper nouns and treats them as single words
    def preprocessing(review):
        proper = r"([A-Z]([a-z]+|\.)(?:\s+[A-Z]([a-z]+|\.))*(?:\s+[a-z][a-z\-]+){0,2}\s+[A-Z]([a-z]+|\.)(?:\s+[0-9]+)?)"
        space_between_brackets = r"[\.\s]+(?=[^\[\]]*]])"
        brackets = r"(?:[\[]{2})(.*?)(?:[\]]{2})"

        review = re.sub(proper, '[[\\1]]', review)
        review = re.sub(space_between_brackets, '~', review)
        review = re.sub(brackets, '\\1', review)
        return review

    model = 'movie_reviews'
    version = request_json['version']
    instances = [preprocessing(i) for i in request_json['instances']]

    service = googleapiclient.discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}/versions/{}'.format(project, model, version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    # clear out nan if they exist
    for r in response['predictions']:
        if all([math.isnan(i) for i in r['prob']]):
            r['prob'] = []
            r['class'] = -1

    return flask.make_response((
        json.dumps(response['predictions']),
        200,
        headers
    ))
```

NOTE: Additional preprocessing for grouping movie names and proper nouns is replicated here since it could not be embedded in the TF input serving function.

Link to all code: [https://github.com/donaldrauscher/movie-reviews-tf](https://github.com/donaldrauscher/movie-reviews-tf)
