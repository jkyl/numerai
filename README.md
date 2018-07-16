# numerai
Classify encrypted financial data for the crowdsourced hedge fund [numer.ai](https://numer.ai)

### Requirements:
* numpy
* pandas
* scikit-learn
* mlxtend
* xgboost

### Algorithms:
* logistic regression
* adaboost
* gradient-boosted trees
* voting ensemble (current leader)

All rely on cross-validation across eras for hyperparameter optimization, and are parallelized using joblib. 
