import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score,cross_validate
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 300)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# amsterdam_prep.py dosyamızdaki amsterdam_data_prep fonksiyonunu çağırıyoruz
from amsterdam_prep import amsterdam_data_prep

#######################################################################################################
#                   ~~~~~~      MODEL  - RIDGE    ~~~~~
########################################################################################################

def main():
    final_model = pd.read_csv('datasets/final_model.csv')

    y = final_model["price"]
    X = final_model.drop(["price"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

    model = RandomForestRegressor(random_state=42)  # LGBMRegressor (*)  ve LinearRegression  en iyi sonuç
    model.fit(X_train, y_train)

    # Modeli test seti ile değerlendirelim
    y_pred = model.predict(X_test)


    # Performans
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("R^2 Score:", r2_score(y_test, y_pred))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")





    ######################################################################################################

    dfLinReg = pd.DataFrame({'Gerçek Fiyat': y_test.values, 'Tahmin Edilen Fiyat': y_pred.flatten()})
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

# Mean Squared Error: 9406.090176573089
# Mean Absolute Error: 66.17640851887707
# R^2 Score: 0.5718534445493221

# yeni değişkenleri ekledikten sonra sonuçlar:

# Mean Squared Error: 238.56414387705712
# Mean Absolute Error: 5.079511132623426
# R^2 Score: 0.9891410336773728




