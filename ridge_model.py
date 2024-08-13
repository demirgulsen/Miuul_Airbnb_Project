

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
#                   ~~~~~~      MODEL  - RIDGE    ~~~~~
########################################################################################################

def preprocess_data(df):
    # Özellikler ve hedef değişkenleri ayırma
    X = df.drop('price', axis=1)
    y = df['price']

    # Kategorik ve sayısal değişkenleri ayırma
    cat_features = [col for col in X.columns if X[col].dtype == 'object']
    num_features = [col for col in X.columns if X[col].dtype != 'object']

    # Özellikler için ön işleme
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), cat_features)
        ]
    )
    return preprocessor.fit_transform(X), y


def main():
    final_model = pd.read_csv('datasets/final_model.csv')

    X_preprocessed, y = preprocess_data(final_model)

    model = Ridge()
    param_grid = {'alpha': [0.1, 1, 10, 100]}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_preprocessed, y)

    # En iyi heiparametre ve performansı yazdırma
    best_model = grid_search.best_estimator_
    print("En iyi hiperparametreler: ", grid_search.best_params_)
    print("En iyi modelin MSE: ", -grid_search.best_score_)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    # Test verisi ile tahmin yapma
    y_pred = best_model.predict(X_preprocessed)

    # Hata metriklerini hesaplama
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print("Ortalama Kare Hatası (MSE): ", mse)
    print("MAE: ", mae)
    print("R^2 Skoru: ", r2)

    ######################################################################################################
    dfLinReg = pd.DataFrame({'Gerçek Fiyat': y.values, 'Tahmin Edilen Fiyat': y_pred.flatten()})
    dfLinReg.head(30)

    # gerçek fiyat ile tahmin edilen fiyat arasındaki farkı görselleştirelim
    first20preds = dfLinReg.head(20)
    c = 'darkgreen', 'steelblue'
    first20preds.plot(kind='bar', figsize=(9, 6), color=c)
    plt.grid(which='major', linestyle='-', linewidth='0.3', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()
    ######################################################################################################


if __name__ == "__main__":
    print("İşlem başladı")
    main()


# yeni değişkenleri eklemeden önce sonuçlar:

# En iyi hiperparametreler:  {'alpha': 100}
# En iyi modelin MSE:  9322.884250983889
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Ortalama Kare Hatası (MSE):  8787.02004890192
# MAE:  67.52288471819149
# R^2 Skoru:  0.5661857592731614


# yeni değişkenleri ekledikten sonra sonuçlar:

# En iyi hiperparametreler:  {'alpha': 0.1}
# En iyi modelin MSE:  2248.958468586733
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Ortalama Kare Hatası (MSE):  2147.116072295093
# MAE:  34.6112177075679
# R^2 Skoru:  0.8939971089776348