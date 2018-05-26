Title: How to Stream Raw Google Analytics Data into BigQuery
Date: 2017-09-19
Tags: google_analytics, bigquery, cloud_functions, etl
Slug: ga-bq-stream

I have been using Google Analytics for a while for my own projects. The Google Analytics interface is great for helping you track activity on your site at a high-level. However, there are some cases in which having access to raw GA events may be helpful. For instance, maybe you record a unique identifier in the user_id parameters and want to tie Google Analytics activity to data in another system, e.g. transactions.

Google Analytics 360 offers easy export of GA events into BigQuery.  However, GA360 is comically expensive: [$150K / year](https://www.quora.com/What-is-the-cost-of-Google-Analytics-360-Suite)!  With this in mind, I sought to create my own solution, which was surprisingly easy.  First, I created a Google Cloud Function to receive these events and ingest them into BigQuery.  Second, I created some client-side JavaScript to also ping my Cloud Function when uploading events to Google Analytics (e.g. piggybacking the ['sendHitTask' task](https://developers.google.com/analytics/devguides/collection/analyticsjs/tasks)).

A few additional notes:
* Google Analytics has a lot of parameters that can be set!  They are detailed [here](https://developers.google.com/analytics/devguides/collection/protocol/v1/parameters).  My code is only syncing a subset of these, a specific subset that I care about.  You will need to edit this and the schema for your table in BigQuery if you want to track additional fields.
* The XMLHttpRequest function, which is what we're using to call our Cloud Function, can make cross-origin HTTP requests but these requests are [controlled by something called CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/Access_control_CORS) (Cross-Origin Resource Sharing). Since our request is a simple GET request, there isn't a pre-flight OPTIONS request; we can simply update our Cloud Function to add a Access-Control-Allow-Origin header which restricts access to the resource to requests from our designated site.
* Cloud Functions are super easy to build and deploy. I highly recommend the [Local Emulator](https://cloud.google.com/functions/docs/emulator) for local testing prior to deployment.  My only complaint is that they only work in Node.js, which I needed to familiarize myself with.  Amazon Lambda Functions, by contrast, can be written in [Java, Node.js, Python, or C#](https://aws.amazon.com/lambda/faqs/)!
* If loading client-side JS as a GA plug-in, ga('require', 'ga_route_plugin') must come *after* the ga('create', ...) command and *before* the ga('send', 'pageview') command.
* Make sure to update the REGION and PROJECT values in ga_route.js and the URL parameter in config.json

Props to Alexander Eroshkin, who created a [similar project](https://github.com/lnklnklnk/ga-bq) (I elected to use Cloud Functions rather than App Engine for simplicity).  You can check out all my entire code and more detailed set-up instructions [here](https://github.com/donaldrauscher/ga-bq-stream).  Cheers!

### Cloud function for ingesting BQ events
``` javascript
const config = require('./config.json');
const bigquery = require('@google-cloud/bigquery')();

function timestamp(){
  var now = new Date();
  now = now.toJSON();
  var regex = /([0-9]{4}-[0-9]{2}-[0-9]{2})T([0-9]{2}:[0-9]{2}:[0-9]{2})/g;
  var match = regex.exec(now);
  return match[1] + ' ' + match[2];
}

exports.ingestGA = function ingestGA (req, res) {
  origin = req.get("origin");
  if (origin != config.URL){
    res.header('Access-Control-Allow-Origin', '*');
    res.status(403).send(`Requests from ${origin} are not allowed!`);
  }
  else {
    res.header('Access-Control-Allow-Origin', config.URL);

    var dataset = bigquery.dataset(config.DATASET);
    var table = dataset.table(config.TABLE);
    var params = req.query;

    var row = {
      json: {
        version: params.v,
        tracking_id: params.tid,
        document_location: params.dl,
        hit_type: params.t,
        user_id: params.uid,
        client_id: params.cid,
        user_language: params.ul,
        event_category: params.ec,
        event_action: params.ea,
        event_label: params.el,
        event_value: params.ev,
        timestamp: timestamp()
      }
    };
    var options = {
      raw: true
    };

    function insertHandler(err, apiResponse){
      if (err){
        res.status(400).send(err);
      }
      else {
        res.status(200).send(apiResponse);
      }
    }

    table.insert(row, options, insertHandler);
  }
};
```

### Client-side JS to route GA events to newly-created cloud function
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
