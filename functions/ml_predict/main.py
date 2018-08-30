
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
