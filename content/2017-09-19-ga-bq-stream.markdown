Title: How to Stream Raw Google Analytics Data into BigQuery
Date: 2017-09-19
Tags: google_analytics, bigquery, cloud_functions, etl
Slug: ga-bq-stream

08-13-2018 Update: [As of 07-24-2018](https://cloud.google.com/functions/docs/release-notes), you can now write Google Cloud Functions in Python!  I re-wrote the Cloud Function in this post in Python.

I have been using Google Analytics for a while for my own projects. The Google Analytics interface is great for helping you track activity on your site at a high-level. However, there are some cases in which having access to raw GA events may be helpful. For instance, maybe you record a unique identifier in the user_id parameters and want to tie Google Analytics activity to data in another system, e.g. transactions.  So I set up a simple process to stream my GA events into BigQuery.

First, I created a Google Cloud Function to receive these events and ingest them into BigQuery:

```python
# streams google analytics data into bigquery
def ingest_ga(request):
    import datetime
    import flask
    from google.cloud import bigquery

    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET'
    }

    response = flask.make_response(('', 204, headers))

    # handle pre-flight options request
    if request.method == 'OPTIONS':
        return response

    mapping = {
        'version': 'v',
        'tracking_id': 'tid',
        'document_location': 'dl',
        'hit_type': 't',
        'user_id': 'uid',
        'client_id': 'cid',
        'user_language': 'ul',
        'event_category': 'ec',
        'event_action': 'ea',
        'event_label': 'el',
        'event_value': 'ev'
    }

    client = bigquery.Client()
    table_ref = client.dataset('google_analytics').table('events')
    table = client.get_table(table_ref)

    row = {k: request.args.get(v) for k, v in mapping.items()}
    row['timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    errors = client.insert_rows(table, [row])
    if len(errors) > 0:
        raise RuntimeError(errors[0]['errors'])

    return response
```

Next, I added some client-side JavaScript to also call my Cloud Function when uploading events to Google Analytics, effectively piggybacking the ['sendHitTask' task](https://developers.google.com/analytics/devguides/collection/analyticsjs/tasks):
``` javascript
function RouteGAData(tracker) {

    ga(function(tracker) {
        var originalSendHitTask = tracker.get('sendHitTask');
        tracker.set('sendHitTask', function(model) {
            var payLoad = model.get('hitPayload');
            originalSendHitTask(model);
            var routeRequest = new XMLHttpRequest();
            var routePath = "https://REGION-PROJECT.cloudfunctions.net/ingestGA";
            routeRequest.open('GET', routePath + "?" + payLoad, true);
            routeRequest.send();
        });
    });

}
ga('provide', 'ga_route_plugin', RouteGAData);
```

A few additional notes:

* Google Analytics has a lot of parameters that can be set!  They are detailed [here](https://developers.google.com/analytics/devguides/collection/protocol/v1/parameters).  My code is only syncing a specific subset of these that I care about.  You will need to edit this and the schema for your table in BigQuery if you want to track additional fields.
* To allow your domain to make requests to region-project.cloudfunctions.net/ingest_ga, I added a `Access-Control-Allow-Origin: *` header to the Cloud Function response, thus enabling Cross-Origin Resource Sharing (CORS).
* If loading client-side JS as a GA plug-in, `ga('require', 'ga_route_plugin')` must come *after* the `ga('create', ...)` command and *before* the `ga('send', 'pageview')` command.  Also make sure to update the REGION and PROJECT values.


You can check out all my entire code and more detailed set-up instructions [here](https://github.com/donaldrauscher/ga-bq-stream).  Cheers!


