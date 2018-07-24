from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from sklearn.externals import joblib
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neural_network import MLPClassifier

from xgboost.sklearn import XGBClassifier
from mlxtend.classifier import EnsembleVoteClassifier


def era_split(eras):
  splits = []
  for val_era in np.unique(eras):
    val_mask = eras==val_era
    splits.append((np.where(~val_mask)[0], np.where(val_mask)[0]))
  return splits


def weight_samples_by_era(eras):
  ueras = np.unique(eras)
  weights = np.ones(eras.size)
  for era in ueras:
    mask = eras==era
    weights[mask] = eras.size / float(mask.sum() * len(ueras))
  return weights


def grid_search(model, param_grid, eras, n_jobs=-1, verbose=2):
  return GridSearchCV(
    model, param_grid,
    cv=era_split(eras),
    scoring='neg_log_loss',
    verbose=verbose,
    n_jobs=n_jobs,
    refit=True)


def linear(X, y, eras, verbose=2, n_jobs=-1):
  sample_weight = weight_samples_by_era(eras)
  model = LogisticRegressionCV(
    solver='lbfgs',
    tol=1e-4,
    max_iter=1000,
    Cs=[1e-3],
    cv=era_split(eras),
    scoring='neg_log_loss',
    verbose=verbose,
    n_jobs=n_jobs,
  ).fit(X, y, sample_weight=sample_weight)
  print('cv loss: '+str(min(-model.scores_[True].mean(0))))
  print('C: '+str(model.C_[0]))
  return model


def xgboost(X, y, eras, verbose=2, n_jobs=-1):
  sample_weight = weight_samples_by_era(eras)
  model = XGBClassifier(
    gamma=0.1,
    max_depth=1,
    subsample=1,
    reg_lambda=1,
    learning_rate=0.1,
    min_child_weight=1,
    colsample_bytree=1,
    n_estimators=100,
    objective='binary:logistic')
  param_grid={}
  search = grid_search(model, param_grid, eras, verbose=verbose, n_jobs=n_jobs)
  search.fit(X, y, sample_weight=sample_weight)
  print(search.best_score_)
  return search.best_estimator_


def train_base_learner(X, y):
  return LogisticRegression(
    C=0.1, solver='sag', tol=1e-4, max_iter=1000).fit(X, y)


def make_voting_ensemble(learners, k):
  return EnsembleVoteClassifier(
    [l for i, l in enumerate(learners) if i != k],
      refit=False, voting='soft').fit([[0], [1]], [0, 1])


def validate_ensemble(ensemble, X, y):
  probs = ensemble.predict_proba(X)[:, 1]
  return log_loss(y, probs)


def voting(X, y, eras, verbose=2, n_jobs=-1):
  def mask(eras, era):
    return np.where(eras==era)[0]
  ueras = np.unique(eras)
  executor = joblib.Parallel(n_jobs=n_jobs, verbose=verbose)
  learners = executor(joblib.delayed(train_base_learner)(
    X[mask(eras, train_era)], y[mask(eras, train_era)])
      for train_era in ueras)
  ensembles = [make_voting_ensemble(learners, k) for k in range(len(ueras))]
  losses = executor(joblib.delayed(validate_ensemble)(
    ensembles[i], X[mask(eras, val_era)], y[mask(eras, val_era)])
      for i, val_era in enumerate(ueras))
  print('cv loss: '+str(np.mean(losses)))
  return make_voting_ensemble(learners, -1)


classifiers = {
  'linear': linear,
  'xgboost': xgboost,
  'voting': voting,
}
