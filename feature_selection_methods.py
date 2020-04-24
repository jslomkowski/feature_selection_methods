
"""
This is a quick demonstration of different features selection techniques.
Particularly how they affect training results and how much does it take to apply
them.
"""

import time

import numpy as np
import pandas as pd
from category_encoders import OrdinalEncoder
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor


def mape(y_pred, y):
    mape = round(mean_absolute_error(y, y_pred) / np.mean(y), 4)
    return mape


def make_prediction(X_train, X_test, y_train, y_test):
    model = XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(mape(y_pred, y_test))
    return mape(y_pred, y_test)


# this is simple custom scorer
mape_score = make_scorer(mape, greater_is_better=False)

# lets import data and print out missing values
df = pd.read_csv('datasets/ames.csv')
nuls = df.isnull().mean().sort_values()
print(nuls[nuls > 0])

# Everything that's above 40% missing values needs to go
df = df.drop(['FireplaceQu', 'Fence', 'Alley',
              'MiscFeature', 'PoolQC'], axis=1)

# separate target variable from features
X = df.loc[:, df.columns != 'SalePrice']
y = df['SalePrice']

# Cut into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=100)

# mark columns that are datatype object and numerical
num = X_train.select_dtypes(exclude='object').columns
cat = X_train.select_dtypes(include='object').columns

# create pipeline to handle categorical variables
cat_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='constant', fill_value='missing')),
    ('conv', OrdinalEncoder(drop_invariant=True, handle_unknown='missing'))])

# create pipeline to handle numerical variables
num_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='median'))])

# transform data using above pipelines for categorical data and numerical
trans = ColumnTransformer([
    ('cat', cat_pipe, cat),
    ('num', num_pipe, num)])

# create final pipeline (this is not necessary but in the future if we decide
# we can fit algorithm here as the last step)
reg = Pipeline([
    ('trans', trans)])

# fit-transform data and change output to pandas dataframe
X_cols = X_train.columns
X_train = reg.fit_transform(X_train)
X_train = pd.DataFrame(X_train, columns=X_cols)
X_test = reg.transform(X_test)
X_test = pd.DataFrame(X_test, columns=X_cols)

# lets create first prediction without any feature selection
make_prediction(X_train, X_test, y_train, y_test)
# score is 0.0938%

# Let's test sequential feature selection from mlxtend package
sfs1 = SFS(XGBRegressor(),
           k_features=len(X_train.columns),  # k_features=2,
           forward=True,  # floating=False,
           verbose=1,
           scoring=mape_score,
           cv=0,
           n_jobs=-1)
time_s = time.time()
sfs1.fit(X_train, y_train)
print('sfs time:', time.time() - time_s)
sfs_results = pd.DataFrame.from_dict(sfs1.subsets_).T.reset_index()

test_mape = pd.DataFrame([], columns=['index', 'test_score'])
for i in range(1, len(sfs_results) + 1):
    print(i)
    _mape = make_prediction(X_train[list(sfs1.subsets_[i]['feature_names'])],
                            X_test[list(sfs1.subsets_[i]['feature_names'])],
                            y_train,
                            y_test)
    result = pd.DataFrame([i, _mape], index=['index', 'test_score'])
    result = result.T
    test_mape = test_mape.append(result)
sfs_results = pd.merge(sfs_results, test_mape, how='left', on='index')
sfs_results['train_test_abs'] = np.abs(
    np.abs(sfs_results['avg_score']) - sfs_results['test_score'])
sfs_results = sfs_results.sort_values('train_test_abs', ascending=True)

# and the winner is... configuration 72 with 9% mae on test data
# train_test_abs confirms that algorithm overfits 8% so more research is needed

sfs_results.to_csv('reports/sfs_results.csv', index=False)

# now, time for ExhaustiveFeatureSelector. This will take some time so use with
# caution
efs1 = EFS(XGBRegressor(),
           min_features=1,
           max_features=len(X_train.columns),
           print_progress=True,
           scoring=mape_score,
           cv=0,
           n_jobs=-1)
time_s = time.time()
# efs1.fit(X_train, y_train)
print('efs time:', time.time() - time_s)
efs_results = pd.DataFrame.from_dict(efs1.subsets_).T
efs_results = efs_results.sort_values('avg_score', ascending=False)
efs_results.to_csv('reports/efs_results.csv', index=False)
make_prediction(X_train[list(efs1.best_feature_names_)],
                X_test[list(efs1.best_feature_names_)],
                y_train,
                y_test)

# fit on ExhaustiveFeatureSelector is switched off by default, it would simply
# take to much time to execute the script. Because of that, recommendation is to
# use SequentialFeatureSelector as a tool to filter important data attributes
