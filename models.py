from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, GridSearchCV

def linear_model(X, y, n_splits=12, **kwargs):
  np.random.seed(666420)
  model = Pipeline([
    ('pca', PCA(whiten=True)),
    ('lr', LogisticRegressionCV(
      Cs=np.geomspace(1e-5, 1e-4, 5),
      cv=StratifiedKFold(n_splits=n_splits),
      scoring='neg_log_loss',
      solver='sag',
      tol=1e-3,
    ))
  ])
  search = GridSearchCV(model,
    param_grid={'pca__n_components': np.arange(10, 20)},
    cv=StratifiedKFold(n_splits=n_splits),
    scoring='neg_log_loss',
    refit=True,
    n_jobs=-1,
    verbose=2,
  ).fit(X, y)
  print('n_components: '+str(search.best_params_['pca__n_components']))
  print('C: '+str(search.best_estimator_.steps[1][1].C_[0]))
  print('CV loss: '+str(-search.best_score_))
  return search.best_estimator_

def adaboost(X, y, n_splits=12, **kwargs):
  np.random.seed(666420)
  model = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1, min_samples_leaf=1))
  search = GridSearchCV(model,
    param_grid={
      'n_estimators': np.arange(2, 12),
      'learning_rate': [0.1],
    },
    cv=StratifiedKFold(n_splits=n_splits),
    scoring='neg_log_loss',
    refit=True,
    n_jobs=-1,
    verbose=2,
  ).fit(X, y)
  print(search.best_params_)
  print(-search.best_score_)
  return search.best_estimator_

classifiers = {
  'linear': linear_model,
  'adaboost': adaboost,
}
