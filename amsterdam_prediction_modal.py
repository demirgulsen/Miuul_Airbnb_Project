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

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 300)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# amsterdam_prep.py dosyamızdaki amsterdam_data_prep fonksiyonunu çağırıyoruz
from amsterdam_prep import amsterdam_data_prep


#######################################################################################################
#                     ~~~~~~      MODEL     ~~~~~
########################################################################################################

def deneme_base_models(X, y, scoring="neg_mean_squared_error"):
    print("Base Models....")
    models = [
        ("LightGBM", LGBMRegressor()),
        ('LR', LinearRegression()),
        ("Ridge", Ridge()),
        ("Lasso", Lasso()),
        ('RF', RandomForestRegressor()),
        ('GBM', GradientBoostingRegressor()),
        ("XGBoost", XGBRegressor(objective='reg:squarederror')),
        ]

    for name, regressor in models:
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
        print(f"RMSE: {round(rmse, 4)} ({name}) ")


########################################################################################################




xgboost_params = {"learning_rate": [0.01,0.05,0.1,0.3],
                  "max_depth": [3,5, 8],
                  "n_estimators": [50,100, 200],
                  "colsample_bytree": [0.2,0.5, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500,1000],
                   "colsample_bytree": [0.5,0.7, 1]}

regressors = [('XGBoost', XGBRegressor(eval_metric='logloss'), xgboost_params),
              ('LightGBM', LGBMRegressor(), lightgbm_params)]

def hyperparameter_optimization(X, y, cv=3, scoring="neg_mean_squared_error"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, regressor, params in regressors:
        print(f"########## {name} ##########")
        cv_results = cross_validate(regressor, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(regressor, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = regressor.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

####################################################################################################
from sklearn.ensemble import VotingRegressor

def voting_regressor(best_models, X, y):
    print("Voting Regressor...")
    voting_reg = VotingRegressor(estimators=[('KNN', best_models["KNN"]),
                                             ('RF', best_models["RF"]),
                                             ('LightGBM', best_models["LightGBM"])])
    voting_reg.fit(X, y)

    # Perform cross-validation
    cv_results = cross_validate(voting_reg, X, y, cv=3, scoring=['neg_mean_squared_error', 'r2'])

    print(f"Mean Squared Error: {-cv_results['test_neg_mean_squared_error'].mean()}")
    print(f"R2 Score: {cv_results['test_r2'].mean()}")

    return voting_reg


####################################################################################################
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

    return train_rmse, test_rmse

def base_models(X,y):
    models = {"LGBM": LGBMRegressor(),
              "XGBoost": XGBRegressor(),
              "RF": RandomForestRegressor(),
              "CatBoost": CatBoostRegressor(verbose=False)}

    for name, model in models.items():
        print(cal_metric_for_regression(model,X, y, scoring=['neg_mean_squared_error', 'r2'], name=name))

#####################################################################################
def gen_model(model,X_train,X_test,y_train, y_test):
    # Modeli test seti ile değerlendirin
    y_pred = model.predict(X_test)

    dfLinReg = pd.DataFrame({'Gerçek Fiyat': y_test.values, 'Tahmin Edilen Fiyat': y_pred.flatten()})
    print(dfLinReg.head(30))

    # comparison
    first20preds = dfLinReg.head(20)
    c = 'darkgreen', 'steelblue'
    first20preds.plot(kind='bar', figsize=(9, 6), color=c)
    plt.grid(which='major', linestyle='-', linewidth='0.3', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()

#####################################################################################
# Feature Önem Düzeylerini kontrol etme
def plot_importance(model, features, num, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.show()
    # if save:
    #     plt.savefig('importances.png')

#####################################################################################

def main():
    final_model = pd.read_csv('datasets/final_model_deneme.csv')

    #############################################################################
    y = final_model["price"]
    X = final_model.drop(["price"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


    # Herhangi bir model seçilebilir
    model = RandomForestRegressor()    # LGBMRegressor (*)  ve LinearRegression  en iyi sonuç
    model.fit(X_train, y_train)

    #############################################################################
    # Gerçek değerler ile tahmin edilen değerler arasındaki faekı görelim
    print("******** Seçilen Modele Göre Gerçek Fiyat ve Tahmin Edilen Fiyat Değerleri Sonuçları ******************")
    gen_model(model,X_train,X_test,y_train, y_test)

    # Özellik önem derecelerini görelim
    plot_importance(model, X_train, len(X.columns))

    print("******************** 1.Base Model Sonuçları *********************************************")
    base_models(X, y)

    print("******************** 2.Deneme Base Models Sonuçları *********************************************")
    deneme_base_models(X, y)



    best_models = hyperparameter_optimization(X, y)
    voting_clf = voting_regressor(best_models, X, y)
    #joblib.dump(voting_clf, "voting_clf.pkl")
    return voting_clf


if __name__ == "__main__":
    print("İşlem başladı")
    main()
