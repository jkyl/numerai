from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import joblib

from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from xgboost.sklearn import XGBClassifier
from mlxtend.classifier import EnsembleVoteClassifier


__author__ = 'Jonathan Kyl'


def era_split(eras):
  '''K-Fold split using `eras` as predetermined folds.
  args:
    eras: array of eras corresponding to training data
  returns:
    splits: list of (train, val) indices, same length as unique eras
  '''
  splits = []
  for val_era in np.unique(eras):
    val_mask = eras==val_era
    splits.append((np.where(~val_mask)[0], np.where(val_mask)[0]))
  return splits


def weight_samples_by_era(eras):
  '''Weight samples inversely to the size of the era that contains them. 
  args:
    eras: array of eras corresponding to training data
  returns:
    weights: array of weights corresponding to training data, computed as 
      $ W_{eras==era} = |W| / (|eras==era| \times |set(eras)|) $
  '''
  ueras = np.unique(eras)
  weights = np.ones(eras.size)
  for era in ueras:
    mask = eras==era
    weights[mask] = eras.size / float(mask.sum() * len(ueras))
  return weights


def grid_search(model, param_grid, eras, n_jobs=-1, verbose=2):
  '''Builds a grid search estimator for finding values of `param_grid` that 
  maximize `model`s generalization to unseen eras. 
  args:
    model: sklearn `Estimator` object with `fit` and `predict_proba` methods
    param_grid: dict whose keys are params of `model` and whose values are 
      lists of values for those params, over which to exhautively search
    eras: array of eras corresponding to training data
  returns:
    GridSearchCV: unfit meta `Estimator` object that will find values of 
      `param_grid` that minimize the mean log loss across eras, then will 
      refit on the entire dataset with the optimal parameters
  '''
  return GridSearchCV(
    model, param_grid,
    cv=era_split(eras),
    scoring='neg_log_loss',
    verbose=verbose,
    n_jobs=n_jobs,
    refit=True)


def linear(X, y, eras, verbose=2, n_jobs=-1):
  '''Fits and validates a logistic regression model with l2 penalty across eras,
  with samples weighted inversely to the size of their containing eras. 
  args:
    X: 2D array of observations
    y: 1D array of target values
    eras: 1D array of eras 
  returns:
    model: model fit to the full data using optimal `C` hyperparameter
  '''
  sample_weight = weight_samples_by_era(eras)
  model = LogisticRegressionCV(
    solver='lbfgs',
    tol=1e-4,
    max_iter=1000,
    Cs=[1e-4, 1e-3, 1e-2],
    cv=era_split(eras),
    scoring='neg_log_loss',
    verbose=verbose,
    n_jobs=n_jobs,
  ).fit(X, y, sample_weight=sample_weight)
  print('cv loss: '+str(min(-model.scores_[True].mean(0))))
  print('C: '+str(model.C_[0]))
  return model


def xgboost(X, y, eras, verbose=2, n_jobs=-1):
  '''Fits and validates an `eXtreme Gradient Boosting` model across eras,
  with optional hyperparameter tuning.
  args:
    X: 2D array of observations
    y: 1D array of target values
    eras: 1D array of eras 
  returns:
    model: model fit to the full data
  '''
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
  '''Fits a single logistic regression model.
  args:
    X: 2D array of observations
    y: 1D array of target values
  returns:
    LogisticRegression: model fit to (X, y)
  '''
  return LogisticRegression(
    C=0.1, solver='sag', tol=1e-4, max_iter=1000).fit(X, y)


def make_voting_ensemble(learners, k=-1):
  '''Constructs an `EnsembleVoteClassifier` meta-estimator from a 
  list of pre-fit models
  args: 
    learners: list of pre-fit sklearn models
    k: index of learner to exclude for validation (-1 == include all)
  returns:
    EnsembleVoteClassifier: meta-estimator whose predicted probabilities 
      are the average of the probabilities predicted by each sub-learner
  '''
  return EnsembleVoteClassifier(
    [l for i, l in enumerate(learners) if i != k],
      refit=False, voting='soft').fit([[0], [1]], [0, 1])


def validate_ensemble(ensemble, X, y):
  '''Evaluate the log loss of a model on a given dataset.
  args:
    ensemble: pre-fit sklearn `Estimator` object
    X: 2D array of observations
    y: 1D array of target values
  returns:
    log_loss: mean cross-entropy loss on the provided data
  '''
  probs = ensemble.predict_proba(X)[:, 1]
  return log_loss(y, probs)


def voting(X, y, eras, verbose=2, n_jobs=-1):
  '''Train and validate a soft voting ensemble of logistic regression 
  models, each fit to a disjoint subset of the data specified by `eras`. 
  args:
    X: 2D array of observations
    y: 1D array of target values
    eras: 1D array of eras 
  returns:
    model: model fit to the full data
  '''
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
