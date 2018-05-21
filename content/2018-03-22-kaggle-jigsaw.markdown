Title: Doc2Vec + Dask + K8s for the Toxic Comment Classification Challenge
Date: 2018-03-22
Tags: kaggle, nlp, doc2vec, gke, dask, sklearn
Slug: kaggle-jigsaw

The goal of [this Kaggle challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) was to build a model to flag toxic Wikipedia comments.  The training dataset included 159,571 Wikipedia comments which were labeled by human raters.  Each comment was evaluated on 6 dimensions: toxic, severe toxic, obscene, threat, insult, and identity hate.

## Model Approach

This challenge is a great application for Doc2Vec, where we treat each of the toxicity dimensions as a label.  For Doc2Vec, I used the [gensim](https://radimrehurek.com/gensim/) package.  I also used gensim's Phraser for combining words into common phrases.  To put everything in a sklearn pipeline, I needed to create sklearn transformers/estimators for each step.  

My final model was a two model blend of Doc2Vec and TF-IDF + LR.  For the LR model, I used the nifty OneVsRestClassifier to build models for each of the 6 y-variables.

## Hyperparameter Tuning

I tuned each input model individually and subsequently the blend.  I used [Dask.distributed](https://distributed.readthedocs.io/en/latest/), specifically the [`dask-searchcv`](http://dask-searchcv.readthedocs.io/en/latest/) package, to parallelize my hyperparameter tuning step.  One of the big advantages of the `dask-searchcv` implementations of GridSearchCV and RandomizedSearchCV is that they avoid repeated work. Estimators with identical parameters and inputs will only be fit once!  In my example, I tested the following grid for my TF-IDF + LR model:
    
```python
param_grid = {
  'cv__lowercase': [True, False],
  'cv__ngram_range': [(1, 1), (1, 2)],
  'tfidf__norm': ['l1', 'l2', None],
  'lr__estimator__C': [0.01, 0.1],
  'lr__estimator__penalty': ['l1', 'l2']
}
```

Even though this parameter grid has 48 different combinations, GridSearchCV will only run the CountVectorizer step 4 times, the TF-IDF step 12 times, etc. Much more efficient!

Here's a snapshot of the Dask web UI during hyper parameter tuning:
<img src="/images/dash-web-ui.png" width="885px" style="display:block; margin-left:auto; margin-right:auto;">

## Dask Cluster

I set up my Dask cluster using Kubernetes.  And, of course, there was a very useful  for this already.  This Helm chart sets up a Dask scheduler + web UI, Dask worker(s), and a Jupyter Notebook instance.  When installing the Helm chart, you can use an accompanying `values.yaml` file to specify which Python packages you need to install.  I also used Terraform to create/scale my K8s cluster.

I created a modified version of [this Dask Helm chart](https://github.com/kubernetes/charts/tree/master/stable/dask) which adds a [`nodeSelector`](https://kubernetes.io/docs/concepts/configuration/assign-pod-node/#nodeselector) option for each of the deployments.  In K8s, we can create two node pools: one for the worker pods and one for the Jupyter/scheduler pods.  That way, when we want to add/remove workers, we can do so without taking down Jupyter!

I set up [three scripts](https://github.com/donaldrauscher/kaggle-jigsaw/tree/master/scripts) for initializing cluster, scaling up the number of nodes / workers, and destroying the cluster when we're done.

Note: The `helm init --wait` command will wait until the Tiller is running and ready to receive requests.  Very useful for CI/CD workflows.  You will need to be running [v2.8.2](https://github.com/kubernetes/helm/releases/tag/v2.8.2) (most recent as of the time of this post) to use this.  

## Notebook

```python
import pandas as pd
import numpy as np
import yaml, re

from google.cloud import storage
from io import BytesIO

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.phrases import Phrases, Phraser

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, strip_tags
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils.validation import check_is_fitted

import distributed
from dask_ml.model_selection import GridSearchCV as GridSearchCVBase
```

```python
# load the data
client_gcs = storage.Client()
bucket = client_gcs.get_bucket('djr-data')

def gcs_to_df(f):
    blob = bucket.blob(f)
    buf = BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    return pd.read_csv(buf, encoding = "utf-8")

df_train = gcs_to_df("kaggle-jigsaw/train.csv")
df_test = gcs_to_df("kaggle-jigsaw/test.csv")
yvar = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
```

```python
# initialize client for interacting with dask
# DASK_SCHEDULER_ADDRESS env variable specifies scheduler ip
client_dask = distributed.Client()
```

```python
# correlation matrix
df_train[yvar].corr()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>toxic</th>
      <th>severe_toxic</th>
      <th>obscene</th>
      <th>threat</th>
      <th>insult</th>
      <th>identity_hate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>toxic</th>
      <td>1.000000</td>
      <td>0.308619</td>
      <td>0.676515</td>
      <td>0.157058</td>
      <td>0.647518</td>
      <td>0.266009</td>
    </tr>
    <tr>
      <th>severe_toxic</th>
      <td>0.308619</td>
      <td>1.000000</td>
      <td>0.403014</td>
      <td>0.123601</td>
      <td>0.375807</td>
      <td>0.201600</td>
    </tr>
    <tr>
      <th>obscene</th>
      <td>0.676515</td>
      <td>0.403014</td>
      <td>1.000000</td>
      <td>0.141179</td>
      <td>0.741272</td>
      <td>0.286867</td>
    </tr>
    <tr>
      <th>threat</th>
      <td>0.157058</td>
      <td>0.123601</td>
      <td>0.141179</td>
      <td>1.000000</td>
      <td>0.150022</td>
      <td>0.115128</td>
    </tr>
    <tr>
      <th>insult</th>
      <td>0.647518</td>
      <td>0.375807</td>
      <td>0.741272</td>
      <td>0.150022</td>
      <td>1.000000</td>
      <td>0.337736</td>
    </tr>
    <tr>
      <th>identity_hate</th>
      <td>0.266009</td>
      <td>0.201600</td>
      <td>0.286867</td>
      <td>0.115128</td>
      <td>0.337736</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>

```python
df_train[yvar].apply(np.mean, axis=0)
```

```python
toxic            0.095844
severe_toxic     0.009996
obscene          0.052948
threat           0.002996
insult           0.049364
identity_hate    0.008805
dtype: float64
```

```python
# train/test split
xdata = df_train.comment_text
ydata = df_train[yvar]
xdata_train, xdata_eval, ydata_train, ydata_eval = train_test_split(xdata, ydata, test_size = 0.2, random_state = 1)
```

```python
# return words from corpus
# TODO: also try r"([\w][\w']*\w)"
def tokenize(doc, token=r"(?u)\b\w\w+\b"):
    doc = strip_tags(doc.lower())
    doc = re.compile(r"\s\s+").sub(" ", doc)
    words = re.compile(token).findall(doc)
    return words


# remove stop words
def remove_stop_words(x, stop_words=ENGLISH_STOP_WORDS):
    return [i for i in x if i not in stop_words]
```

```python
# wrapper for gensim Phraser
COMMON_TERMS = ["of", "with", "without", "and", "or", "the", "a"]
class PhraseTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, common_terms=COMMON_TERMS):
        self.phraser = None
        self.common_terms = common_terms

    def fit(self, X, y=None):
        phrases = Phrases(X, common_terms=self.common_terms)
        self.phraser = Phraser(phrases)
        return self

    def transform(self, X):
        return X.apply(lambda x: self.phraser[x])
```

```python
# for making tagged documents
# NOTE: can't use FunctionTransformer since TransformerMixin doesn't pass y to fit_transform anymore
class MakeTaggedDocuments(BaseEstimator):

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        if y is not None:
            yvar = list(y.columns)
            tags = y.apply(lambda row: [i for i,j in zip(yvar, row) if j == 1], axis=1)
            return [TaggedDocument(words=w, tags=t) for w,t in zip(X, tags)]
        else:
            return [TaggedDocument(words=w, tags=[]) for w in X]

    def fit_transform(self, X, y):
        return self.transform(X, y)
```

```python
# wrapper for gensim Doc2Vec
class D2VEstimator(BaseEstimator):

    def __init__(self, min_count=10, alpha=0.025, min_alpha=0.0001, vector_size=200, dm=0, epochs=20):
        self.min_count = min_count
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.vector_size = vector_size
        self.dm = dm
        self.epochs = epochs
        self.yvar = None
        self.model = Doc2Vec(seed=1, hs=1, negative=0, dbow_words=0,
                             min_count=self.min_count, alpha=self.alpha, min_alpha=self.min_alpha,
                             vector_size=self.vector_size, dm=self.dm, epochs=self.epochs)

    def get_tags(self, doc):
        vec = self.model.infer_vector(doc.words, self.model.alpha, self.model.min_alpha, self.model.epochs)
        return dict(self.model.docvecs.most_similar([vec]))

    def fit(self, X, y=None):
        self.model.build_vocab(X)
        self.model.train(X, epochs=self.model.epochs, total_examples=self.model.corpus_count)
        self.model.delete_temporary_training_data()
        self.yvar = list(y.columns)
        return self

    def predict_proba(self, X):
        pred = [self.get_tags(d) for d in X]
        pred = pd.DataFrame.from_records(data=pred)
        return pred[self.yvar]
```

```python
# blend predictions from multiple models
class Blender(FeatureUnion):

    def __init__(self, transformer_list, n_jobs=1, transformer_weights=None):
        self.transformer_list = transformer_list
        self.scaler_list = [(t, StandardScaler()) for t, _ in transformer_list]
        self.n_jobs = n_jobs
        default_transformer_weights = list(np.ones(len(transformer_list)) / len(transformer_list))
        self.transformer_weights = transformer_weights if transformer_weights else default_transformer_weights

    @property
    def transformer_weights(self):
        return self._transformer_weights

    @transformer_weights.setter
    def transformer_weights(self, values):
        self._transformer_weights = {t[0]:v for t,v in zip(self.transformer_list, values)}

    # don't need to check for fit and transform
    def _validate_transformers(self):
        pass

    # iterator with scalers
    def _iter_ss(self):
        get_weight = (self.transformer_weights or {}).get
        return [(t[0], t[1], s[1], get_weight(t[0])) for t, s in zip(self.transformer_list, self.scaler_list)]

    # also fit scalers
    def fit(self, X, y):
        super(Blender, self).fit(X, y)
        self.scaler_list = [(name, ss.fit(trans.predict_proba(X))) for name, trans, ss, _ in self._iter_ss()]
        return self

    # generate probabilities
    def predict_proba(self, X):
        Xs = [ss.transform(trans.predict_proba(X))*weight for name, trans, ss, weight in self._iter_ss()]
        return np.sum(Xs, axis=0)
```

```python
# create pipeline
d2v_pipeline = Pipeline(steps=[
    ('tk', FunctionTransformer(func=lambda x: x.apply(tokenize), validate=False)),
    ('ph', PhraseTransformer()),
    ('sw', FunctionTransformer(func=lambda x: x.apply(remove_stop_words), validate=False)),
    ('doc', MakeTaggedDocuments()),
    ('d2v', D2VEstimator())
])

lr_pipeline = Pipeline(steps=[
    ('cv', CountVectorizer(min_df=5, max_features=50000, strip_accents='unicode',
                           stop_words='english', analyzer='word')),
    ('tfidf', TfidfTransformer(sublinear_tf=True, use_idf=True)),
    ('lr', OneVsRestClassifier(LogisticRegression(class_weight="balanced")))
])

pipeline = Blender(transformer_list=[('d2v', d2v_pipeline), ('lr', lr_pipeline)])
```

```python
# for non-multimetric, don't require refit = True for best_params_ / best_score_
class GridSearchCV(GridSearchCVBase):

    # For multiple metric evaluation, refit is a string denoting the scorer that should be
    # used to find the best parameters for refitting the estimator
    @property
    def scorer_key(self):
        return self.refit if self.multimetric_ else 'score'

    @property
    def best_index(self):
        check_is_fitted(self, 'cv_results_')
        return np.flatnonzero(self.cv_results_['rank_test_{}'.format(self.scorer_key)] == 1)[0]

    @property
    def best_params_(self):
        return self.cv_results_['params'][self.best_index]

    @property
    def best_score_(self):
        return self.cv_results_['mean_test_{}'.format(self.scorer_key)][self.best_index]
```

```python
# some functions for dealing with parameter grids
def add_prefix(prefix, x):
    return {'{}__{}'.format(prefix, k):v for k,v in x.items()}

def flatten_dict(x):
    temp = {}
    for k,v in x.items():
        if isinstance(v, dict):
            temp.update(add_prefix(k, flatten_dict(v.copy())))
        else:
            temp.update({k: v})
    return temp
```

```python
# hyperparameter tuning
param_grid = {
    'd2v': {
        'd2v__min_count': [10, 25],
        'd2v__alpha': [0.025, 0.05],
        'd2v__epochs': [10, 20, 30],
        'd2v__vector_size': [200, 300]        
    },
    'lr': {
        'cv__lowercase': [True, False],
        'cv__ngram_range': [(1, 1), (1, 2)],
        'tfidf__norm': ['l1', 'l2', None],
        'lr__estimator__C': [0.01, 0.1],
        'lr__estimator__penalty': ['l1', 'l2']        
    },
    'blender': {
        'transformer_weights': [(0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3)]        
    }
}

# wrapper for hyperparameter tuning
def hyperparameter_tune(pipeline, param_grid):
    # create tuner
    tuner = GridSearchCV(pipeline, param_grid, scheduler=client_dask, scoring='roc_auc',
                         cv=3, refit=False, return_train_score=False)

    # determine optimal hyperparameters
    tuner.fit(xdata_train, ydata_train)
    print('Best params: %s' % (str(tuner.best_params_)))
    print('Best params score: %s' % (str(tuner.best_score_)))

    return tuner.best_params_

# load saved hyperparameters if available; o.w. tune
try:
    with open('model_param_d2v.yaml', 'r') as f:
        param_optimal = yaml.load(f)

except IOError:
    param_optimal = {}

    # tune each model
    param_optimal['d2v'] = hyperparameter_tune(d2v_pipeline, param_grid['d2v'])
    param_optimal['lr'] = hyperparameter_tune(lr_pipeline, param_grid['lr'])

    # tune blender
    d2v_pipeline.set_params(**param_optimal['d2v'])
    lr_pipeline.set_params(**param_optimal['lr'])
    param_optimal.update(hyperparameter_tune(pipeline, param_grid['blender']))

    # flatten
    param_optimal = flatten_dict(param_optimal)

    # save best params
    with open('model_param_d2v.yaml', 'w') as f:
        yaml.dump(param_optimal, f)
```

<pre>Best params: {'d2v__alpha': 0.025, 'd2v__epochs': 30, 'd2v__min_count': 10, 'd2v__vector_size': 200}
Best params score: 0.9520673206887134
Best params: {'cv__lowercase': True, 'cv__ngram_range': (1, 1), 'lr__estimator__C': 0.1, 'lr__estimator__penalty': 'l2', 'tfidf__norm': 'l2'}
Best params score: 0.9764642394949188
Best params: {'transformer_weights': (0.3, 0.7)}
Best params score: 0.9774035665175447</pre>

```python
# build model with optimal param
pipeline.set_params(**param_optimal)
pipeline.fit(xdata_train, ydata_train)
```

<pre>Blender(n_jobs=1,
    transformer_list=[('d2v', Pipeline(memory=None,
     steps=[('tk', FunctionTransformer(accept_sparse=False,
          func=<function <lambda> at 0x7f39416d12f0>, inv_kw_args=None,
          inverse_func=None, kw_args=None, pass_y='deprecated',
          validate=False)), ('ph', PhraseTransformer(com...ne,
          solver='liblinear', tol=0.0001, verbose=0, warm_start=False),
          n_jobs=1))]))],
    transformer_weights={'d2v': 0.3, 'lr': 0.7})</pre>

```python
# apply to eval set
ydata_eval_pred = pipeline.predict_proba(xdata_eval)
```

```python
# calculate auc
auc = [roc_auc_score(ydata_eval[y], ydata_eval_pred[:,i]) for i,y in enumerate(yvar)]
print('Model AUCs: %s' % auc)
print('Avg AUC: %s' % np.mean(auc))
```

<pre>Model AUCs: [0.9662283198414882, 0.9857095145804597, 0.982421955124849, 0.9849362663053255, 0.9757783792333873, 0.9768901227451926]
Avg AUC: 0.9786607596384505</pre>

```python
# generate final model
pipeline_final = clone(pipeline)
pipeline_final.set_params(**param_optimal)
pipeline_final.fit(xdata, ydata)
```

<pre>Blender(n_jobs=1,
    transformer_list=[('d2v', Pipeline(memory=None,
     steps=[('tk', FunctionTransformer(accept_sparse=False,
          func=<function <lambda> at 0x7f39416d12f0>, inv_kw_args=None,
          inverse_func=None, kw_args=None, pass_y='deprecated',
          validate=False)), ('ph', PhraseTransformer(com...ne,
          solver='liblinear', tol=0.0001, verbose=0, warm_start=False),
          n_jobs=1))]))],
    transformer_weights={'d2v': 0.3, 'lr': 0.7})</pre>

```python
# generate output
xdata_test = df_test.comment_text
ydata_test_pred = pipeline_final.predict_proba(xdata_test)
ydata_test_pred = pd.DataFrame(data=ydata_test_pred, columns=yvar)
ydata_test_pred['id'] = df_test.id
ydata_test_pred.to_csv('submission.csv', index=False)
```

===

Pretty good!  With more time, I definitely would have focused on adding more models to the stack, e.g. Naive Bayes and RF/XGBoost.  A [link](https://github.com/donaldrauscher/kaggle-jigsaw) to my repo on GH.
