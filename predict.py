from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import argparse
import numpy as np
import pandas as pd

from models import classifiers

def process_data(train_df, test_df, target):
  features = [i for i in train_df.columns if 'feature' in i]
  target = [i for i in train_df.columns if target in i]
  target = target[0]
  live_mask = np.isnan(test_df[target].values)
  train = (train_df[features].values, train_df[target].values)
  test = (test_df[features].values, test_df[target].values)
  full = (
    np.concatenate([train[0], test[0][~live_mask]], axis=0),
    np.concatenate([train[1], test[1][~live_mask]], axis=0))
  return full, test

def main(args):
  train_df, test_df = pd.read_csv(args.train_csv), pd.read_csv(args.test_csv)
  (X, y), (X_test, _) = process_data(train_df, test_df, args.target)
  model = classifiers[args.model](X, y, n_splits=args.n_splits)
  probs = model.predict_proba(X_test)[:, 1]
  zipped = np.stack([test_df['id'], probs], axis=1)
  return np.concatenate([[['id', 'probability_'+args.target]], zipped], axis=0)

if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('train_csv', type=str)
  p.add_argument('test_csv', type=str)
  p.add_argument('out_csv', type=str)
  p.add_argument('-m', '--model', type=str, default='linear')
  p.add_argument('-t', '--target', type=str, default='bernie')
  p.add_argument('-n', '--n_splits', type=int, default=3)
  args = p.parse_args()
  results = main(args)
  np.savetxt(args.out_csv, results, delimiter=',', fmt='%s')
