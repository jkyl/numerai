from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import glob
import argparse

import numerapi
import credentials

import numpy as np
import pandas as pd

from models import classifiers


def preprocess_data(train_df, test_df, target):
  features = [i for i in train_df.columns if i.startswith('feature')]
  target = 'target_'+target
  assert target in train_df.columns
  live_mask = np.isnan(test_df[target].values)
  train = (train_df[features].values, train_df[target].values.astype(bool))
  test = (test_df[features].values, test_df[target].values.astype(bool))
  full = (
    np.concatenate([train[0], test[0][~live_mask]]),
    np.concatenate([train[1], test[1][~live_mask]]),
    np.array([int(s[3:]) for s in np.concatenate(
      [train_df['era'].values, test_df['era'][~live_mask].values])]))
  return full, test[0]


def postprocess_results(ids, probs, target):
  zipped = np.stack([ids, probs], axis=1)
  prepend = np.concatenate([[['id', 'probability_' + target]], zipped])
  return prepend


def download(api, round, datadir):
  round_dir = os.path.join(datadir, str(round))
  if not os.path.exists(round_dir):
    os.makedirs(round_dir)
    api.download_current_dataset(round_dir, unzip=True)
    for csv_file in glob.glob(os.path.join(round_dir, '*/*.csv')):
      os.rename(csv_file,
        os.path.join(round_dir, os.path.basename(csv_file)))
  return round_dir


def predict(round_dir, target, model, verbose, n_jobs, **kwargs):
  np.random.seed(666420)
  train_csv = os.path.join(round_dir, 'numerai_training_data.csv')
  test_csv = os.path.join(round_dir, 'numerai_tournament_data.csv')
  train_df, test_df = pd.read_csv(train_csv), pd.read_csv(test_csv)
  (X, y, eras), X_test = preprocess_data(train_df, test_df, target)
  model = classifiers[model](X, y, eras=eras, verbose=verbose, n_jobs=n_jobs)
  probs = model.predict_proba(X_test)[:, 1]
  results = postprocess_results(test_df['id'], probs, target)
  return results


def main(args):
  api = numerapi.NumerAPI(
    verbosity='debug',
    secret_key=credentials.key,
    public_id=credentials.id)
  current_round = api.get_current_round()
  if args.round == 'latest':
    args.round = current_round
  elif int(args.round) > current_round:
    raise ValueError('round {} does not exist (latest == {})'
       .format(args.round, current_round))
  round_dir = download(api, args.round, args.datadir)
  results = predict(round_dir, **args.__dict__)
  output_file = os.path.join(round_dir,
    'results_target-{}_model-{}.csv'.format(args.target, args.model))
  np.savetxt(output_file, results, delimiter=',', fmt='%s')
  if args.upload:
    tournament = dict(zip(
      ['bernie', 'elizabeth', 'jordan', 'ken', 'charles'],
      range(1, 6)))[args.target]
    api.upload_predictions(output_file, tournament)


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('-d', '--datadir', type=str, default='/Volumes/4TB/numerai')
  p.add_argument('-m', '--model', type=str, default='voting')
  p.add_argument('-r', '--round', type=str, default='latest')
  p.add_argument('-t', '--target', type=str, default='bernie')
  p.add_argument('-u', '--upload', action='store_true', default=False)
  p.add_argument('-v', '--verbose', type=int, default=2)
  p.add_argument('-j', '--n_jobs', type=int, default=-1)
  args = p.parse_args()
  main(args)
