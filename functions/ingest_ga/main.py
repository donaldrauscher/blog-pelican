
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


