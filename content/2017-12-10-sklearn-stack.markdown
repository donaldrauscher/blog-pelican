Title: Model Stacking with Sklearn
Date: 2017-12-10
Tags: python, sklearn, machine_learning, model_stacking
Slug: sklearn-stack

[Stacking](https://rd.springer.com/content/pdf/10.1007%2FBF00117832.pdf), also called meta ensembling, is a technique used to boost predictive accuracy by blending the predictions of multiple models.  This technique is most effective when you have multiple, well-performing models which are _not_ overly similar.  Participants in Kaggle competitions will observe that winning solutions are often blends of multiple models, sometimes even models available in public notebooks!  A [nice write-up](https://mlwave.com/kaggle-ensembling-guide/) from Kaggle grand master [Triskelion](https://www.kaggle.com/triskelion) on using stacking in Kaggle competitions.  Throw back: the winning solution to the NetFlix challenge, from team BellKor's Pragmatic Chaos, used [a blend of hundreds of different models](https://www.netflixprize.com/assets/GrandPrize2009_BPC_BigChaos.pdf).

<img src="/images/stacking.png" style="display:block; margin-left:auto; margin-right:auto;">
Source: [https://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/](https://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/)

I recently sought to implement a simple model stack in sklearn.  The mlxtend package has a [`StackingClassifier`](https://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/) for this.  However, there was one big problem with this class: it does not allow you to use out-of-sample predictions from input models to train the meta classifier.  This is a huge problem!  Otherwise, overfitting models will dominate the weights.  I created my own class, leveraging the native [`FeatureUnion`](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html) class to house the input models and `cross_val_predict` to generate out-of-sample predictions.  For the meta classifier itself, I applied the logit function to the probabilities from the input models and fed them into a simple logistic regression.

``` python
import numpy as np

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression

from scipy.special import logit


# method for linking `predict_proba` to `transform`
def chop_col0(function):
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs)[:,1:]
    return wrapper


def add_transform(classifiers):
    for key, classifier in classifiers:
        if isinstance(classifier, Pipeline):
            classifier = classifier.steps[-1][-1]
        classifier.transform = chop_col0(classifier.predict_proba)
        classifier.__class__.transform = chop_col0(classifier.__class__.predict_proba)
        # NOTE: need to add to class so `clone` in `cross_val_predict` works


# default function applies logit to probabilies and applies logistic regression
def default_meta_classifier():
    return Pipeline([
        ('logit', FunctionTransformer(lambda x: logit(np.clip(x, 0.001, 0.999)))),
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression())
    ])


# stacking classifier
class StackingClassifier(Pipeline):

    def __init__(self, classifiers, meta_classifier=None, cv=3):
        add_transform(classifiers)
        self.classifiers = classifiers
        self.meta_classifier = meta_classifier if meta_classifier else default_meta_classifier()
        self.cv = cv
        self.steps = [('stack', FeatureUnion(self.classifiers)), ('meta', self.meta_classifier)]
        self.memory = None

    @staticmethod
    def add_dict_prefix(x, px):
        return {'%s__%s' % (px, k) : v for k,v in x.items()}

    def set_params(self, **kwargs):
        return super(StackingClassifier, self).set_params(**self.add_dict_prefix(kwargs, 'stack'))

    def fit(self, X, y):
        meta_features = cross_val_predict(FeatureUnion(self.classifiers), X, y, cv=self.cv, method="transform")
        self.meta_classifier.fit(meta_features, y)
        for name, classifier in self.classifiers:
            classifier.fit(X, y)
        return self
```

You can see this code implemented [here](https://github.com/donaldrauscher/hospital-readmissions/blob/master/model.py).  I built a model to predict which recently hospitalized diabetic patients will be re-hospitalized within 30 days, using [this dataset](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008) from UCI.  My model stack contained a logistic regression with regularization, a random forest, and a gradient boosting (xgboost) model.  Here is a summary of model performance:

<table class="pretty">
<tr><th>Model</th><th>AUC</th></tr>
<tr><td>LR+RF+XGB Model Stack</td><td>0.6990824552912449</td></tr>
<tr><td>LR+RF+XGB Average</td><td>0.6981398497127431</td></tr>
<tr><td>XGBoost</td><td>0.6956653497449965</td></tr>
<tr><td>Random Forest</td><td>0.6952079165690574</td></tr>
<tr><td>Logistic Regression</td><td>0.684611003872049</td></tr>
</table>

As you can see, a simple average of the models outperforms any one model.  And our model stack outperforms the simple average.

Another technique that I'd like to explore is [feature-weighted linear stacking](https://arxiv.org/pdf/0911.0460.pdf): tuning a meta model using interactions of meta features and input model predictions, the idea being that we can identify pockets of samples in which certain models perform best.  More on this later!
