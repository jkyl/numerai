from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import argparse
import numpy as np
import pandas as pd

from models import classifiers

def preprocess_data(train_df, test_df, target):
  features = [i for i in train_df.columns if i.startswith('feature')]
  target = 'target_'+target
  assert target in train_df.columns
  live_mask = np.isnan(test_df[target].values)
  train = (train_df[features].values, train_df[target].values)
  test = (test_df[features].values, test_df[target].values)
  full = (
    np.concatenate([train[0], test[0][~live_mask]]),
    np.concatenate([train[1], test[1][~live_mask]]),
    np.concatenate([train_df['era'].values,
                    test_df['era'][~live_mask].values]))
  return full, test[0]

def postprocess_results(test_df, probs, target):
  zipped = np.stack([test_df['id'], probs], axis=1)
  prepend = np.concatenate([[['id', 'probability_' + target]], zipped])
  return prepend

def main(train_csv, test_csv, target='bernie', model='linear', **kwargs):
  np.random.seed(666420)
  train_df, test_df = pd.read_csv(train_csv), pd.read_csv(test_csv)
  (X, y, eras), X_test = preprocess_data(train_df, test_df, target)
  model = classifiers[model](X, y, eras=eras, **kwargs)
  probs = model.predict_proba(X_test)[:, 1]
  results = postprocess_results(test_df, probs, target)
  return results

if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('train_csv', type=str)
  p.add_argument('test_csv', type=str)
  p.add_argument('out_csv', type=str)
  p.add_argument('-m', '--model', type=str, default='linear')
  p.add_argument('-t', '--target', type=str, default='bernie')
  p.add_argument('-v', '--verbose', type=int, default=2)
  p.add_argument('-j', '--n_jobs', type=int, default=-1)
  args = p.parse_args()
  results = main(**args.__dict__)
  np.savetxt(args.out_csv, results, delimiter=',', fmt='%s')
