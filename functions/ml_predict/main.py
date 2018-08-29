
# gets predictions from cloud ml engine
def ml_predict(request):
    import flask
    import json
    import googleapiclient.discovery
    import google.auth

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

    response_obj = flask.make_response(json.dumps(response['predictions']))
    response_obj.headers.add('Access-Control-Allow-Origin', '*')

    return response_obj
