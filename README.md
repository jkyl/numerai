# numerai
Classify encrypted financial data for the crowdsourced hedge fund [numer.ai](https://numer.ai)

### Requirements:
* numpy
* pandas
* scikit-learn
* mlxtend
* xgboost

### Algorithms:
##### (`*` indicates current leader)
* logistic regression
* gradient-boosted trees
* voting ensemble
* genetic hyperparameter search
* gradient descent hyperparameter search *

All rely on cross-validation across eras for hyperparameter optimization, and are parallelized using joblib. 
