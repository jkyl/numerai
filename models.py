from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from sklearn.metrics import log_loss
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from xgboost.sklearn import XGBClassifier
from mlxtend.classifier import EnsembleVoteClassifier

def grid_search(
    model, param_grid,
    n_splits=3,
    n_jobs=-1,
    verbose=2,
    **kwargs
  ):
  return GridSearchCV(
    model, param_grid,
    cv=StratifiedKFold(n_splits),
    scoring='neg_log_loss',
    verbose=verbose,
    n_jobs=n_jobs,
    refit=True,
  ).best_estimator_

def linear(X, y, **kwargs):
  model = LogisticRegressionCV(
    Cs=np.geomspace(1e-4, 1e-2, 9),
    cv=StratifiedKFold(kwargs['n_splits']),
    verbose=kwargs['verbose'],
    n_jobs=kwargs['n_jobs'],
    scoring='neg_log_loss',
    solver='lbfgs',
    max_iter=1000,
    tol=1e-4,
  ).fit(X, y)
  for v in model.scores_.values():
    print('mean: '+str(-v.mean(0)))
    print('std:  '+str(v.std(0)))
  print('C: '+str(model.C_[0]))
  return model

def pca_linear(X, y, **kwargs):
  model = Pipeline([
    ('pca', PCA(whiten=True)),
    ('lr', LogisticRegressionCV(
      Cs=np.geomspace(1e-5, 1e-4, 3),
      cv=StratifiedKFold(n_splits=3),
      scoring='neg_log_loss',
      solver='sag',
      tol=1e-3,
    ))
  ])
  param_grid = {'pca__n_components': np.arange(1, 51)}
  return grid_search(model, param_grid, **kwargs).fit(X, y)

def adaboost(X, y, **kwargs):
  model = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1, min_samples_leaf=1),
    learning_rate=0.1,
  )
  param_grid={'n_estimators': np.arange(2, 12)}
  return grid_search(model, param_grid, **kwargs).fit(X, y)

def xgboost(X, y, **kwargs):
  model = XGBClassifier(
    gamma=1,
    max_depth=1,
    subsample=0.8,
    reg_lambda=0.1,
    learning_rate=0.1,
    min_child_weight=1,
    colsample_bytree=1,
    objective='binary:logistic'
  )
  param_grid={'n_estimators': [80, 100, 120]}
  return grid_search(model, param_grid, **kwargs).fit(X, y)

def train_base_learner(X, y):
  return LogisticRegression(
    C=1e-1, solver='sag', tol=1e-4, max_iter=1000
  ).fit(X, y)

def make_voting_ensemble(learners, k):
  return EnsembleVoteClassifier(
    [learners[i] for i in range(len(learners)) if i != k],
    refit=False).fit([[0], [1]], [0, 1])

def validate_ensemble(ensemble, X, y):
  probs = ensemble.predict_proba(X)[:, 1]
  return log_loss(y, probs)

def voting(X, y, eras, **kwargs):
  memory = joblib.Memory('/tmp/joblib', verbose=0)
  @memory.cache(ignore=['eras'])
  def mask(eras, era):
    return eras == era
  ueras = np.unique(eras)
  executor = joblib.Parallel(
    n_jobs=kwargs['n_jobs'],
    verbose=kwargs['verbose'])
  learners = executor(joblib.delayed(train_base_learner)(
    X[mask(eras, train_era)], y[mask(eras, train_era)])
      for train_era in ueras)
  ensembles = executor(joblib.delayed(make_voting_ensemble)(
    learners, k) for k in range(len(ueras)))
  losses = executor(joblib.delayed(validate_ensemble)(
    ensembles[i], X[mask(eras, val_era)], y[mask(eras, val_era)])
      for i, val_era in enumerate(ueras))
  print('cv loss (unweighted): '+str(np.mean(losses)))
  memory.clear(warn=False)
  return make_voting_ensemble(learners, -1)

classifiers = {
  'linear': linear,
  'pca_linear': pca_linear,
  'adaboost': adaboost,
  'xgboost': xgboost,
  'voting': voting,
}
