

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score,cross_validate
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn import metrics
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 300)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# amsterdam_prep.py dosyamızdaki amsterdam_data_prep fonksiyonunu çağırıyoruz
from amsterdam_prep import amsterdam_data_prep

#######################################################################################################
#                   ~~~~~~      BASE MODELS     ~~~~~
########################################################################################################

def cal_metric_for_regression(model,X, y, scoring, name):
    model = model
    cv_results = cross_validate(model, X, y, cv=10, scoring=scoring, return_train_score=True)

    train_rmse = np.sqrt(-cv_results['train_neg_mean_squared_error'].mean())
    test_rmse = np.sqrt(-cv_results['test_neg_mean_squared_error'].mean())
    train_r2 = cv_results['train_r2'].mean()
    test_r2 = cv_results['test_r2'].mean()

    print(f"############## {name} #################")
    print("Train RMSE: ", round(train_rmse, 4))
    print("Test RMSE: ", round(test_rmse, 4))
    print("Train R2: ", round(train_r2, 4))
    print("Test R2: ", round(test_r2, 4))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    return train_rmse, test_rmse


def plot_importance(model, features, num, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.show()


def main():
    final_model = pd.read_csv('datasets/final_model.csv')

    y = final_model["price"]
    X = final_model.drop(["price"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)


    models = {"LGBM": LGBMRegressor(),
              "XGBoost": XGBRegressor(),
              "RF": RandomForestRegressor(),
              "CatBoost": CatBoostRegressor(verbose=False)}

    for name, model in models.items():
        cal_metric_for_regression(model, X, y, scoring=['neg_mean_squared_error', 'r2'], name=name)

        model.fit(X, y)
        # Özellik önem derecelerini çizdirir
        plot_importance(model, X, len(X.columns))



    ######################################################################################################

if __name__ == "__main__":
    print("İşlem başladı")
    main()


# yeni değişkenleri eklemeden önce sonuçlar:

############## LGBM #################
# Train RMSE:  61.3814
# Test RMSE:  90.6858
# Train R2:  0.8139
# Test R2:  0.582
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ############## XGBoost #################
# Train RMSE:  31.4309
# Test RMSE:  96.5647
# Train R2:  0.9512
# Test R2:  0.5247
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ############## RF #################
# Train RMSE:  35.4517
# Test RMSE:  96.548
# Train R2:  0.9379
# Test R2:  0.526
# ############## CatBoost #################
# Train RMSE:  52.6971
# Test RMSE:  89.6305
# Train R2:  0.8629
# Test R2:  0.5912

# yeni değişkenleri ekledikten sonra sonuçlar:

# ############## LGBM #################
# Train RMSE:  7.7093
# Test RMSE:  13.7463
# Train R2:  0.9971
# Test R2:  0.9905
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ############## XGBoost #################
# Train RMSE:  2.506
# Test RMSE:  13.3758
# Train R2:  0.9997
# Test R2:  0.9909
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ############## RF #################
# Train RMSE:  5.106
# Test RMSE:  14.0407
# Train R2:  0.9987
# Test R2:  0.9901
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
############## CatBoost #################
# Train RMSE:  4.7457
# Test RMSE:  11.0792
# Train R2:  0.9989
# Test R2:  0.9937

