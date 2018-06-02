Title: Using Word2Vec for "Code Names"
Date: 2018-05-12
Tags: word2vec, nlp, dash, docker
Slug: w2v-code-names

["Code Names"](https://en.wikipedia.org/wiki/Codenames_(board_game)) Rules: People are divided into two teams.  The board is comprised of 25 words divided into 4 categories: blue team, red team, neutral, and the death word.  People are divided evenly into two teams (red and blue).  In each round, two people from either team take turns giving 1 word clues.  The goal is to get the other members of your team to guess your teams' words and NOT the other words, especially not the death word; if your team guesses the death word, you immediately lose.

It is a really fun game.  I also thought it might be an interesting application for Word2Vec. Word2Vec is a two-layer neural network which models the linguistic contexts of words.  There are two approaches to training Word2Vec: CBOW (continuous bag of words) and skip-gram.  CBOW predicts a word from a window of surrounding words.  Skip-gram uses a single word to predict words in the surrounding window.  This is a [nice summary](https://www.tensorflow.org/tutorials/word2vec).  Also cool, you don't need to train your own Word2Vec model! Lots of people/organizations provide pre-trained word vectors that you can easily implement, e.g. [Google News](https://code.google.com/archive/p/word2vec/) and [Facebook](https://fasttext.cc/docs/en/english-vectors.html).

I built a small app that uses Word2Vec to generate word hints for "Code Names".  I used Python's [`gensim`](https://radimrehurek.com/gensim/models/word2vec.html) package to measure word similarities / generate hints using pre-trained word vectors from [Stanford NLP's GloVe](https://nlp.stanford.edu/projects/glove/).  The app itself is built using [Plotly's Dash](https://plot.ly/products/dash/), which is analogous to Shiny for R. I packaged the entire thing in a Docker container.

Dash app (app.py)

    :::python
    import os
    
    import dash
    import dash_core_components as dcc
    import dash_html_components as html
    from dash.dependencies import Input, Output
    
    import pandas as pd
    import numpy as np
    
    from gensim.models import KeyedVectors
    
    import plotly.figure_factory as ff
    
    
    # initialize app
    app = dash.Dash()
    server = app.server
    
    # load model
    model = 'glove/w2v.{}.txt.gz'.format(os.getenv('GLOVE_MODEL', 'glove.6B.100d'))
    word_vectors = KeyedVectors.load_word2vec_format(model, binary=False)
    
    # precompute L2-normalized vectors (saves lots of memory)
    word_vectors.init_sims(replace=True)
    
    
    # pandas df to html
    def generate_table(df, max_rows=10):
        return html.Table(
            # header
            [html.Tr([html.Th(col) for col in df.columns])] +
    
            # body
            [html.Tr([
                html.Td(df.iloc[i][col]) for col in df.columns
            ]) for i in range(min(len(df), max_rows))]
        )
    
    
    # generate some clues
    def generate_hints(words):
        try:
            hints = word_vectors.most_similar(positive=words)
            hints = pd.DataFrame.from_records(hints, columns=['word','similarity'])
            return generate_table(hints)
        except KeyError as e:
            return html.Div(str(e))
    
    
    # generate dendrogram for word similarity
    def generate_dendro(words):
        try:
            similarities = np.array([word_vectors.distances(w, words) for w in words])
            figure = ff.create_dendrogram(similarities, labels=words)
            figure['layout'].update({'width': 800, 'height': 500})
            return figure
        except KeyError as e:
            pass
    
    
    # set up app layout
    app.layout = html.Div(children=[
        html.H1(children='Code Names Hints'),
        html.Table([
            html.Tr([html.Td("All Words:"), html.Td("Words for Hints:")]),
            html.Tr([html.Td(dcc.Textarea(id='words-all', value='god zeus bat ball mountain cold snow', style={'width': 500})),
                     html.Td(dcc.Input(id='words', value='bat ball', type='text'))]),
            html.Tr([html.Td(dcc.Graph(id='dendro')), html.Td(html.Div(id='hints'))])
        ])
    ])
    
    
    # set up app callbacks
    @app.callback(
        Output(component_id='dendro', component_property='figure'),
        [Input(component_id='words-all', component_property='value')]
    )
    def update_dendro(input_value):
        words = [w.lower() for w in input_value.strip().split(' ')]
        return generate_dendro(words)
    
    @app.callback(
        Output(component_id='hints', component_property='children'),
        [Input(component_id='words', component_property='value')]
    )
    def update_hints(input_value):
        words = [w.lower() for w in input_value.strip().split(' ')]
        return generate_hints(words)
    
    
    # run
    if __name__ == '__main__':
        app.run_server(debug=True)

Dockerfile

    :::docker
    FROM python:3.5-slim
    
    ENV PORT 8050
    ENV GLOVE_MODEL glove.6B.200d
    ENV GUNICORN_WORKERS 3
    ENV APP_DIR /app
    
    WORKDIR $APP_DIR
    
    RUN apt-get update \
      && apt-get install -y unzip gzip wget \
      && rm -rf /var/lib/apt/lists/*
    
    COPY requirements.txt app.py entrypoint.sh ./
    RUN chmod +x entrypoint.sh
    
    RUN pip install -r requirements.txt
    
    RUN wget -q http://nlp.stanford.edu/data/glove.6B.zip \
      && unzip glove.6B.zip -d glove \
      && rm glove.6B.zip \
      && python -m gensim.scripts.glove2word2vec --input glove/${GLOVE_MODEL}.txt --output glove/w2v.${GLOVE_MODEL}.txt \
      && gzip glove/w2v.${GLOVE_MODEL}.txt \
      && rm glove/*.txt
    
    EXPOSE $PORT
    
    ENTRYPOINT $APP_DIR/entrypoint.sh

entrypoint.sh

    :::bash
    #!/bin/bash
    echo Starting Gunicorn...
    gunicorn app:server \
        --name code-names \
        --bind 0.0.0.0:$PORT \
        --workers $GUNICORN_WORKERS \
        --preload \
        --worker-class gevent \
        --timeout 600 \
        --log-level info \
        "$@"

Example output:
<img src="{filename}images/code-names.png" width="885px" style="display:block; margin-left:auto; margin-right:auto;">

Overall, it does...okay haha.  In some cases, it does surprisingly well.  For instance, the app provides "published" as a top hint for "book" and "penguin".  However, the algorithm struggles to identify commonalities that may not be explicitly collocated in text.  For instance, for "dog" and "whale", "mammal" might be a good hint.  However, our app simply lists other animals, e.g. "cat" and "shark".

I'm hosting a version of the app [here](https://code-names-ypzseaxpnx.now.sh/) on Now.sh.  And a [link](https://github.com/donaldrauscher/code-names) to my repo on GH.  Cheers!
