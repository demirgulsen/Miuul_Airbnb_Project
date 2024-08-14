import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score,cross_validate
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 300)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

amsterdam_all_listing = pd.read_csv('datasets/all_listings.csv')

awd_df = amsterdam_all_listing.copy()
awd_df.head()
awd_df.shape
awd_df.isnull().sum()
awd_df.columns
awd_df.info()
awd_df.describe().T


rev_detail = pd.read_csv('datasets/reviews_comment.csv')
rev_detail.head()
rev_detail.shape
rev_detail.isnull().sum()

##############################################################
# VERİ ÖNİŞLEME
##############################################################
# Herbir değişkenin eşsiz değer sayılarını ve veri tiplerini birlikte inceleyelim

def control_value_counts(df):
    data = []
    for col in awd_df.columns:
        column_info = {
            'dtype': awd_df[col].dtype,
            'column_name': col,
            'unique_count': awd_df[col].nunique()
        }
        data.append(column_info)
    return pd.DataFrame(data)
result_df = control_value_counts(awd_df)
result_df.head(50)

#####################################################################################################################################
# bu daha kısa :)
# awd_df = awd_df[['id', 'host_id', 'host_name', 'host_since', 'neighbourhood_cleansed', 'latitude', 'longitude', 'property_type', \
#  'room_type', 'accommodates', 'bathrooms_text', 'bedrooms', 'beds', 'amenities', 'price', 'minimum_nights', 'maximum_nights',\
#  'availability_30', 'availability_365', 'calendar_last_scraped', 'number_of_reviews', 'review_scores_rating',\
#  'review_scores_communication', 'review_scores_location', 'instant_bookable', 'reviews_per_month']]
#####################################################################################################################################

delete_columns = ['listing_url','name','description','calendar_updated','last_scraped','scrape_id','source', \
                  'host_url','host_location','host_about','host_thumbnail_url','host_neighbourhood','host_verifications', \
                  'host_identity_verified','host_picture_url','host_has_profile_pic','license','picture_url','first_review','last_review',\
                  'neighbourhood_group_cleansed','neighborhood_overview','neighbourhood','bathrooms', 'has_availability', \
                  'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_is_superhost', 'minimum_nights_avg_ntm','maximum_nights_avg_ntm',\
                  'minimum_minimum_nights','maximum_minimum_nights','minimum_maximum_nights','maximum_maximum_nights', \
                  'availability_60','availability_90','number_of_reviews_ltm','number_of_reviews_l30d','host_listings_count',\
                  'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms','calculated_host_listings_count_shared_rooms', \
                  'review_scores_accuracy','calculated_host_listings_count','host_total_listings_count','review_scores_cleanliness',\
                  'review_scores_checkin','review_scores_value','instant_bookable'
                  ]
#,'reviews_per_month', 'amenities'
awd_df = awd_df.drop(delete_columns, axis=1)

###################################################################################################
# Tüm amenities sütunlarını ekle
awd_df = awd_df.assign(
    has_private_entrance=awd_df['amenities'].apply(lambda x: x.find('Private entrance') != -1),
    has_self_checkin=awd_df['amenities'].apply(lambda x: x.find('Self check-in') != -1),
    has_kitchen=awd_df['amenities'].apply(lambda x: x.find('Kitchen') != -1),
    has_bathtub=awd_df['amenities'].apply(lambda x: x.find('Bathtub') != -1),
    has_host_greeting=awd_df['amenities'].apply(lambda x: x.find('Host greets you') != -1),
    has_dishwasher=awd_df['amenities'].apply(lambda x: x.find('Dishwasher') != -1)
    # has_longterm=awd_df['amenities'].apply(lambda x: x.find('Long term stays allowed') != -1),
    # has_fireplace=awd_df['amenities'].apply(lambda x: x.find('Indoor fireplace') != -1),
    # has_parking=awd_df['amenities'].apply(lambda x: x.find('Free parking on premises') != -1)
)

awd_df = awd_df.drop(['amenities'],axis=1)

##########################################################################################
# AYKIRI DEĞERLERİ KONTROL ETME VE TEMİZLEME
##########################################################################################
def outlier_thresholds(dataframe, col_name, q1=0.15, q3=0.85):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if (dataframe[col_name] > up_limit).any() or (dataframe[col_name] < low_limit).any():

        return True
    else:
        return False

#########################################################
# Değişknelerin tipini verir
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
        Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
        Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

        Parameters
        ------
            dataframe: dataframe
                    Değişken isimleri alınmak istenilen dataframe
            cat_th: int, optional
                    numerik fakat kategorik olan değişkenler için sınıf eşik değeri
            car_th: int, optinal
                    kategorik fakat kardinal değişkenler için sınıf eşik değeri

        Returns
        ------
            cat_cols: list
                    Kategorik değişken listesi
            num_cols: list
                    Numerik değişken listesi
            cat_but_car: list
                    Kategorik görünümlü kardinal değişken listesi

        Examples
        ------
            import seaborn as sns
            df = sns.load_dataset("iris")
            print(grab_col_names(df))


        Notes
        ------
            cat_cols + num_cols + cat_but_car = toplam değişken sayısı
            num_but_cat cat_cols'un içerisinde.
            Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

        """

    # cat_cols, cat_but_car
    # num_cols = [col for col in df.columns if df[col].dtype != "O"]
    # num_but_cat = [col for col in num_cols if df[col].nunique() < 10]

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(awd_df)

num_cols = [col for col in num_cols if col not in ['id','host_id','calendar_last_scraped']]

for col in num_cols:
    print(col, check_outlier(awd_df, col))

#############################################################################
# Aykırı Değerlerin Kendilerine Erişmek
#############################################################################

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

for col in num_cols:
    grab_outliers(awd_df, col,True)


#############################################################################
# Baskılama Yöntemi ile aykırı değerleri temizleyelim

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    print(col, check_outlier(awd_df, col))

for col in num_cols:
    replace_with_thresholds(awd_df,col)

for col in num_cols:
    print(col, check_outlier(awd_df, col))

##########################################################################################
# EKSİK DEĞERLERİ KONTROL ETME VE TEMİZLEME
##########################################################################################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(awd_df)

#null_columns = missing_values_table(awd_df, True)

###################################################################################################
# awd_df['bedrooms'] = awd_df['bedrooms'].fillna(0)
# awd_df['beds'] = awd_df['beds'].fillna(awd_df['bedrooms'])

del_null = ['beds','bedrooms', 'bathrooms_text']
awd_df = awd_df.dropna(subset=del_null)

awd_df['combine_beds_bedrooms_acommodates'] = awd_df['beds'] + awd_df['bedrooms'] + awd_df['accommodates']

awd_df.drop(['beds','bedrooms','accommodates'], axis=1, inplace=True)

###################################################################################################

awd_df['price'] = awd_df['price'].replace({r'^\$': '', r',': ''}, regex=True).astype(float)

awd_df['calendar_last_scraped'] = pd.to_datetime(awd_df['calendar_last_scraped'], errors='coerce')

##########################################################################################

# property_type' ı kategorilere göre gruplama

property_type_counts = awd_df['property_type'].value_counts()
property_type_counts[property_type_counts < 11].index

ent_rent_unit_types = set(list(set(col for col in awd_df['property_type'].values if 'Entire rental unit' in col)))
ent_condo_types = set(list(set(col for col in awd_df['property_type'].values if 'Entire condo' in col)))
ent_home_types = set(list(set(col for col in awd_df['property_type'].values if 'Entire home' in col)))

# 'Entire' içeren türler, belirtilen türler hariç
entire_types = set(col for col in awd_df['property_type'].values if 'Entire' in col)
entire_types = entire_types - (ent_rent_unit_types | ent_condo_types | ent_home_types)

private_types = list(set(col for col in awd_df['property_type'].values if 'Private' in col))
shared_types = list(set(col for col in awd_df['property_type'].values if 'Shared' in col))
room_in_types = list(set(col for col in awd_df['property_type'].values if 'Room in' in col))
boat_types= list(set(col for col in awd_df['property_type'].values if 'Boat' in col))
entire_villa_types= list(set(col for col in awd_df['property_type'].values if 'Entire villa' in col))
room_in_apot_types= list(set(col for col in awd_df['property_type'].values if 'Room in aparthotel' in col))
entire_vachom_types= list(set(col for col in awd_df['property_type'].values if 'Entire vacation home' in col))
sha_room_in_housboat_types= list(set(col for col in awd_df['property_type'].values if 'Shared room in houseboat' in col))

other_types = list(set([col for col in awd_df['property_type'].values if (col not in ent_rent_unit_types) and (col not in ent_condo_types) and (col not in ent_home_types) and\
                        (col not in entire_types) and (col not in private_types) and (col not in shared_types) and (col not in room_in_types) and \
                       (col not in boat_types) and (col not in entire_villa_types) and (col not in room_in_apot_types) and (col not in entire_vachom_types) and \
                       (col not in entire_vachom_types)]))
categories = {
    'Entire rental unit': ent_rent_unit_types,
    'Entire condo':ent_condo_types,
    'Entire home' : ent_home_types,
    'Entire Units': entire_types,
    'Private Rooms': private_types,
    'Shared Rooms': shared_types,
    'Room in': room_in_types,
    'Other': other_types
}
########################3
def categorize(property_type):
    for category, values in categories.items():
        if property_type in values:
            return category
    return 'Unknown'

# `property_type`'ı kategorilere ayırma
awd_df['property_type_category'] = awd_df['property_type'].apply(categorize)

awd_df = awd_df.drop('property_type', axis=1)

############################################################################################################
# Eksik Değer Problemini Çözme
############################################################################################################
# Tüm tarihler 06/2024' e ait. O yüzden tarihe göre sıralamaya gerek yok

awd_df['price'] = (awd_df.groupby(['neighbourhood_cleansed', 'room_type', 'property_type_category'])['price']   # model başarısı için property_type buna göre de denenebilir
                   .transform(lambda x: x.replace(0, np.nan).fillna(x.mean()).replace(np.nan, 0)))

####################################################################################
def calculate_price_mean(df):
    # Öncelikle gruplamayı deneyelim
    grouped = df.groupby(['neighbourhood_cleansed', 'room_type', 'property_type_category'])

    # Eğer gruplama sonucu veri varsa, fiyatların ortalamasını hesaplayalım
    if not grouped.size().eq(0).all():
        # Fiyatların ortalamasını hesaplayalım
        result = grouped['price'].mean()
    else:
        # Eğer veri yoksa, bir üst gruplama düzeyine geçelim
        grouped = df.groupby(['neighbourhood_cleansed', 'room_type'])
        if not grouped.size().eq(0).all():
            result = grouped['price'].mean()
        else:
            # Eğer bu gruplamada da veri yoksa, son düzey olan 'neighbourhood_cleansed' grubuna geçelim
            grouped = df.groupby(['neighbourhood_cleansed'])
            result = grouped['price'].mean()

    return result

# Fonksiyonu çağırarak sonuçları elde edelim
price_mean = calculate_price_mean(awd_df)
print(price_mean)


# Ya da    ####################################################################################

def fill_missing_prices(df):
    # Grup düzeylerine göre eksik fiyatları doldur
    for group_level in [['neighbourhood_cleansed', 'room_type', 'property_type_category'],
                        ['neighbourhood_cleansed', 'room_type'],
                        ['neighbourhood_cleansed']]:
        # Grup düzeyinde eksik değerleri doldur
        df['price'] = df.groupby(group_level)['price'].transform(
            lambda x: x.replace(0, np.nan).fillna(x.mean()).replace(np.nan, 0))

        # Eğer eksik değer kalmadıysa, döngüyü kır
        if df['price'].isna().sum() == 0:
            break

    return df

# Fonksiyonu çağırarak eksik değerleri dolduralım
awd_df = fill_missing_prices(awd_df)
print(awd_df)

# değişkenlerin price ile ilişkisini görselleştirelim
# grouped = awd_df.groupby(['neighbourhood_cleansed', 'room_type', 'property_type_category'])['price'].mean().reset_index()
# pivot_df = grouped.pivot_table(index=['neighbourhood_cleansed', 'room_type'], columns='property_type_category', values='price')
#
# plt.figure(figsize=(12, 8))
# pivot_df.plot(kind='bar', stacked=True, figsize=(14, 7))
# plt.title('Average Price by Neighbourhood, Room Type, and Property Type')
# plt.xlabel('Neighbourhood and Room Type')
# plt.ylabel('Average Price')
# plt.legend(title='Property Type Category')
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.show()
######################################################

##########################################################################################
# bathrooms_text

awd_df.dropna(subset=['bathrooms_text'], inplace=True)

# Sayısal değeri çıkaran fonksiyon
def extract_bathroom_number(bathroom_text):
    # "0 shared baths" gibi ifadeleri 0 olarak değerlendirmek
    if '0' in bathroom_text:
        return 0

    # Regex ile sayısal değeri bul
    match = re.search(r'(\d+(\.\d+)?)', bathroom_text)
    if match:
        return float(match.group(1))
    return 0


# Fonksiyonu uygulama
awd_df['bathroom_count'] = awd_df['bathrooms_text'].apply(extract_bathroom_number)

awd_df = awd_df.drop('bathrooms_text', axis=1)

#######################################################

# half_baths = {'Half-bath', 'Private half-bath', 'Shared half-bath'}
#
# #private_baths = ['private bath']
# private_baths = {'private bath'}
#
# shared_baths = set(col for col in awd_df['bathrooms_text'].values if 'shared baths' in col)
# shared_baths.add('1 shared bath')
#
# baths = set(col for col in awd_df['bathrooms_text'].values if ('baths' in col))
# baths = baths - (private_baths | shared_baths)
# baths.add('1 bath')
#
# # Kategorilere göre etiketleme fonksiyonu
# def categorize_bathrooms(text):
#     if text in half_baths:
#         return 'Half Bath'
#     elif text in private_baths:
#         return 'Private Bath'
#     elif text in shared_baths:
#         return 'Shared Bath'
#     elif text in baths:
#         return 'Bath'
#     else:
#         return 'Other'
#
# # Fonksiyonu uygulama
# awd_df['bathroom_category'] = awd_df['bathrooms_text'].apply(categorize_bathrooms)
#
# # Sonuçları göster
# print(awd_df[['bathrooms_text', 'bathroom_category']])
#
# awd_df.head(30)
############################################################################################################
# review içeren ve null değer sayısı 0' dan büyük olan sütunları silelim

rev_cols = [col for col in awd_df.columns if awd_df[col].isnull().sum() > 0 and 'review' in col]
awd_df = awd_df.dropna(subset=rev_cols)

missing_values_table(awd_df)

# ------------------

weights = {
    'review_scores_rating': 0.30,
    'review_scores_communication': 0.30,
    'review_scores_location': 0.40,
}

# Bu ağırlıklarla genel skoru hesaplayalım
def calculate_weighted_score(df, weights):
    score = 0
    for col, weight in weights.items():
        if col in df.columns:
            score += df[col] * weight
    return score

# Genel skoru dataframe'e ekleyin
awd_df['general_score'] = calculate_weighted_score(awd_df, weights)


rev_del = ['review_scores_rating', 'review_scores_communication','review_scores_location']
awd_df = awd_df.drop(columns=rev_del)

# Genel skoru ve fiyatı karşılaştırmak için bir scatter plot oluşturalım
plt.figure(figsize=(10, 6))
sns.scatterplot(x='general_score', y='price', data=awd_df)
plt.title('General Score vs Price')
plt.xlabel('General Score')
plt.ylabel('Price')
plt.show()


# aykırı değer oluşuyor onları silelim
cat_cols, num_cols,  cat_but_car = grab_col_names(awd_df)
for col in num_cols:
    replace_with_thresholds(awd_df,col)

for col in num_cols:
        print(col, check_outlier(awd_df, col))

##########################################################################################
# Numerik ve kategorik değişkenleri getirelim

cat_cols, num_cols,  cat_but_car = grab_col_names(awd_df)

def inf_value_counts(df, col_list):
    data = []
    for col in col_list:
        column_info = {
            'dtype': df[col].dtype,
            'column_name': col,
            'unique_count': df[col].nunique()
        }
        data.append(column_info)
    return pd.DataFrame(data)


inf_cat_cols = inf_value_counts(awd_df, cat_cols)
inf_num_cols = inf_value_counts(awd_df, num_cols)


#####################################################
# Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.

#Numerik
num_df= awd_df[num_cols]
correlation_matrix=num_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Korelasyon Matrisi')
plt.show()


#Kategorik
for var in cat_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=awd_df[var], hue=awd_df['price'])
    plt.title(f'{var} ve price arasındaki ilişki')
    plt.xlabel(var)
    plt.ylabel('Count')
    plt.show()
#######################################################################
# Feature Engineering

awd_df['price_range'] = pd.cut(awd_df['price'], bins=[0, 50, 100, 200, 500, np.inf],
                               labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
awd_df['nightly_price'] = awd_df['price'] / awd_df['minimum_nights']
awd_df['host_active_years'] = (pd.to_datetime(date.today()) - pd.to_datetime(awd_df['host_since'])).dt.days // 365
awd_df['review_count'] = awd_df['number_of_reviews'] + awd_df['reviews_per_month'] * 12

#######################################################################
# Konum bilgisini kullanarak kümeleme yapalım

coords = awd_df[['latitude', 'longitude']]
kmeans = KMeans(n_clusters=10).fit(coords)
awd_df['location_cluster'] = kmeans.labels_

# Encoder işlemlerinden önce silinecek diğer değişkenleri silelim
awd_df = awd_df.drop(['id','host_id','host_name','calendar_last_scraped','host_since','latitude', 'longitude','neighbourhood_cleansed'], axis=1)

#############################################################################

final_df = awd_df.copy()

###############################################################################
 #  ENCODER İŞLEMLERİNİ UYGULAYALIM
###############################################################################
cat_cols, num_cols,  cat_but_car = grab_col_names(final_df)

# 2 SINIFLI DEĞİŞKENLERE LABEL ENCODER UYGULAYALIM
le = LabelEncoder()

def label_encoder(dataframe, binary_col):
    dataframe[binary_col] = le.fit_transform(dataframe[binary_col])
    return dataframe

# 2 sınıflı kategorik değişkenleri seçmeliyiz
binary_cols = [col for col in final_df.columns if final_df[col].dtype not in ['int64', 'float']
               and final_df[col].nunique() == 2]  # len(df[col].unique()) == 2 şeklinde de yazılabilir fakat unique nan değerleri de saydığı için kullanılması önerilmez

for col in binary_cols:
    label_encoder(final_df, col)

###########################################################
# room_type' a özel olarak label encoder uygulayaalım
leb = LabelEncoder()
room_type_mapping = {
    'Entire home/apt': 0,
    'Private room': 1,
    'Hotel room': 2,
    'Shared room': 3
}
final_df['room_type'] = final_df['room_type'].map(room_type_mapping)
final_df['room_type'] = leb.fit_transform(final_df['room_type'])

###########################################################
# Diğer kategorik değişkenlere one-hot encoder uygulayalım
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in final_df.columns if 10 >= final_df[col].nunique() > 2]

final_df = one_hot_encoder(final_df, ohe_cols)

##########################################################################################
# tekrar eden kayıt var mı kontrol edelim, varsa silelim
final_df.duplicated().sum()
# final_df = final_df.drop_duplicates()

final_df.head()

##########################################################################################
# Standart Scaler Uygulayalım
##########################################################################################
cat_cols, num_cols, cat_but_car = grab_col_names(final_df)
num_cols = [col for col in num_cols if "price" not in col]

scaler = StandardScaler()
final_df[num_cols] = scaler.fit_transform(final_df[num_cols])


final_df[num_cols].head()

#final_df.to_csv('datasets/Amsterdam_Datasets/final_model.csv', index=False)

#######################################################################################################
#######################################################################################################
# Çok iyi sonuçlar veriyor

# Veriyi bölme ve ön işleme
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

X_preprocessed, y = preprocess_data(final_df)

# Ridge Regression Modeli ve Hiperparametre Optimizasyonu
model = Ridge()
param_grid = {'alpha': [0.1, 1, 10, 100]}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_preprocessed, y)

# En iyi modeli ve performansı yazdırma
best_model = grid_search.best_estimator_
print("En iyi hiperparametreler: ", grid_search.best_params_)
print("En iyi modelin MSE: ", -grid_search.best_score_)



# Test verisi ile tahmin yapma
y_pred = best_model.predict(X_preprocessed)

# Hata metriklerini hesaplama
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("Ortalama Kare Hatası (MSE): ", mse)
print("MAE: ", mae)
print("R^2 Skoru: ", r2)

dfLinReg = pd.DataFrame({'Gerçek Fiyat': y.values, 'Tahmin Edilen Fiyat': y_pred.flatten()})
dfLinReg.head(30)

#gerçek fiyat ile tahmin edilen fiyat arasındaki farkı görselleştirelim
first20preds=dfLinReg.head(20)
c='darkgreen', 'steelblue'
first20preds.plot(kind='bar',figsize=(9,6), color=c)
plt.grid(which='major', linestyle='-', linewidth='0.3', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# Ortalama Kare Hatası (MSE):  2149.8616897176457
# MAE:  34.604772912969366
# R^2 Skoru:  0.8938615581389134

# Feature Importence
#######################################################################################################
#######################################################################################################

y = final_df["price"]
X = final_df.drop(["price"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

#######################################################################################################
#                     ~~~~~~      MODEL     ~~~~~
########################################################################################################
# Herhangi bir modele göre denenebilir
model = RandomForestRegressor(random_state=42)    # LGBMRegressor (*)  ve LinearRegression  en iyi sonuç
model.fit(X_train, y_train)

# Modeli test seti ile değerlendirelim
y_pred = model.predict(X_test)

dfLinReg = pd.DataFrame({'Gerçek Fiyat': y_test.values, 'Tahmin Edilen Fiyat': y_pred.flatten()})
dfLinReg.head(30)

#  Gerçek Fiyat  Tahmin Edilen Fiyat
# 0        179.000              169.820
# 1        495.000              479.350
# 2         79.000               81.830
# 3        390.000              389.550
# 4        280.000              280.070
# 5        300.000              300.000
# 6        175.000              175.980
# 7         75.000               73.920
# 8        220.000              220.510
# 9        205.000              204.690
# 10       450.000              450.000
# 11       390.000              390.200
# 12        61.000               73.500
# 13       321.000              321.500
# 14       379.000              377.680
# 15       225.000              225.060
# 16       224.000              223.800
# 17       314.000              314.050
# 18       300.000              299.990
# 19       100.000               88.290
# 20       178.000              176.350
# 21       500.000              499.760
# 22       138.000              137.700
# 23       184.000              184.310
# 24        66.000               76.900
# 25       400.000              400.000
# 26       325.000              325.420
# 27       350.000              349.150
# 28       125.000              127.440
# 29       275.000              274.790

# Performans
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Mean Squared Error: 233.2204779525653
# Mean Absolute Error: 4.981437560503388
# R^2 Score: 0.9893842667440458

#gerçek fiyat ile tahmin edilen fiyat arasındaki farkı görselleştirelim
first20preds=dfLinReg.head(20)
c='darkgreen', 'steelblue'
first20preds.plot(kind='bar',figsize=(9,6), color=c)
plt.grid(which='major', linestyle='-', linewidth='0.3', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

########################################################################################################

models = {"LGBM": LGBMRegressor(),
          "XGBoost": XGBRegressor(),
          "RF": RandomForestRegressor(),
          "CatBoost": CatBoostRegressor(verbose=False)}

def cal_metric_for_regression(model, scoring, name):
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


for name, model in models.items():
    cal_metric_for_regression(model, scoring=['neg_mean_squared_error', 'r2'], name=name)


# En son sonuçlar...
# ############## LGBM #################
# Train RMSE:  61.9029
# Test RMSE:  91.8755
# Train R2:  0.8108
# Test R2:  0.5712
# ############## XGBoost #################
# Train RMSE:  32.5033
# Test RMSE:  96.0655
# Train R2:  0.9478
# Test R2:  0.5306
# ############## RF #################
# Train RMSE:  35.5432
# Test RMSE:  97.571
# Train R2:  0.9376
# Test R2:  0.5165
# ############## CatBoost #################
# Train RMSE:  52.8706
# Test RMSE:  90.6397
# Train R2:  0.8619
# Test R2:  0.5827


# ############## LGBM #################
# Train RMSE:  7.6325
# Test RMSE:  13.7632
# Train R2:  0.9971
# Test R2:  0.9905
# ############## XGBoost #################
# Train RMSE:  2.4522
# Test RMSE:  12.9582
# Train R2:  0.9997
# Test R2:  0.9915
# ############## RF #################
# Train RMSE:  5.0544
# Test RMSE:  13.9827
# Train R2:  0.9987
# Test R2:  0.9902
# ############## CatBoost #################
# Train RMSE:  4.748
# Test RMSE:  11.0457
# Train R2:  0.9989
# Test R2:  0.9938

########################################################################################################################
# Feature Önem Düzeylerini kontrol edelim

model = RandomForestRegressor()
model.fit(X, y)
def plot_importance(model, features, num=len(X.columns), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.show()
    # if save:
    #     plt.savefig('importances.png')

# Özellik önem derecelerini çizdirir
plot_importance(model, X)

########################################################################################################################################################
# train_df = final_df[final_df['price'].notnull()]
# test_df = final_df[final_df['price'].isnull()]

# y = train_df['SalePrice']  # np.log1p(df['SalePrice'])
# X = train_df.drop(["Id", "SalePrice"], axis=1)
#
# # Train verisi ile model kurup, model başarısını değerlendiriniz.
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),

          ('KNN', KNeighborsRegressor()),
          ('RF', RandomForestRegressor()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")



# MAE
from sklearn.metrics import mean_absolute_error
for name, regressor in models:
    model = regressor.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    print(f"MAE: {mae} ({name}) ")


final_df['price'].mean()
final_df['price'].std()
final_df["price"].hist(bins=100)
plt.show(block=True)
########################################################################################################
##################
# BONUS : Log dönüşümü yaparak model kurunuz ve rmse sonuçlarını gözlemleyiniz.
# Not: Log'un tersini (inverse) almayı unutmayınız.
##################

# Log dönüşümünün gerçekleştirilmesi
train_df= final_df["price"]
test_df = final_df.drop(["price"], axis=1)

# plt.hist(np.log1p(train_df['SalePrice']), bins=100)
y = np.log1p(train_df['price'])
X = train_df.drop(["price"], axis=1)

# Verinin eğitim ve tet verisi olarak bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)


lgbm = LGBMRegressor().fit(X_train, y_train)
y_pred = lgbm.predict(X_test)

y_pred
# Yapılan LOG dönüşümünün tersinin (inverse'nin) alınması
new_y = np.expm1(y_pred)
new_y
new_y_test = np.expm1(y_test)
new_y_test

np.sqrt(mean_squared_error(new_y_test, new_y))

# RMSE : 22866.43915128612

###############################################################
# hiperparametre optimizasyonlarını gerçekleştiriniz.
###############################################################

lgbm_model = LGBMRegressor(random_state=46)

rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model, X, y, cv=5, scoring="neg_mean_squared_error")))


lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500],
               "colsample_bytree": [0.5, 0.7, 1]
             }

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=5,
                            n_jobs=-1,
                            verbose=-1).fit(X, y)


lgbm_gs_best.best_params_
final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

print(f"İlk RMSE: {rmse}")
rmse_new = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))
print(f"Yeni RMSE: {rmse_new}")







































'''
amsterdam_all_listing = pd.read_csv('datasets/Amsterdam_Datasets/all_listings.csv')
ve
rev_detail = pd.read_csv('datasets/Amsterdam_Datasets/reviews_comment.csv')

veri setlerini okutup ikisini birleştirelim ve kullanacağımız değişkenleri belirleyip veri setini eda için hazır hale getirelim
Daha sonra
comments değişkenindeki null kayıtları silelim
son olarak ta 
amsterdam_eda sayfasında yazdığımız prep fonksiyonuna gönderelim
Ordan hazır hale gelen final_model.csv veri setini kullanarak recommend işlemlerini gerçekleştirelim

1. Description
2. Amenities
3. Coments

# Filtreleme için;
4. host_name
5. neighbourhood_cleansed
6. property_type
7. room_type
8. price
'''

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv('datasets/Amsterdam_Datasets/all_listings.csv')

df = df[
    ['id', 'name', 'description', 'host_id', 'host_name', 'host_since', 'neighbourhood_cleansed',
     'property_type', 'room_type', 'bedrooms', 'price', 'minimum_nights', 'maximum_nights',
     'has_availability', 'first_review', 'last_review', 'availability_30', 'availability_365',
     'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
     'review_scores_communication', 'review_scores_location', 'review_scores_checkin', 'instant_bookable',
     'reviews_per_month']]

df['price'] = df['price'].replace({r'^\$': '', r',': ''}, regex=True).astype(float)
df['price'] = (df.groupby(['neighbourhood_cleansed'])['price']
               .transform(lambda x: x.replace(0, np.nan).fillna(x.mean()).replace(np.nan, 0)))

df.loc[:, 'description'] = df['description'].fillna('name')
df.loc[:, 'has_availability'] = df['has_availability'].fillna('f')
df = df[df['has_availability'] == 't']  # aktif olanları gösterelim

#######################################################################
tfidf = TfidfVectorizer(stop_words='english')
df['description'] = df['description'].fillna('')
tfidf_matrix = tfidf.fit_transform(df['description'])

#######################################################################
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
#######################################################################
indices = pd.Series(df.index, index=df['name'])
indices = indices[~indices.index.duplicated(keep='last')]
user_input = 'Comfortable double room'
movie_index = indices[user_input]
# if movie_index == []:
#    return []
similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
movie_indices = similarity_scores.sort_values(by='score', ascending=False)[0:11].index
df['name'].iloc[movie_indices]
# #######################################################################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)





def content_based_recommender(title, cosine_sim, dataframe):
    # index'leri olusturma
    indices = pd.Series(dataframe.index, index=dataframe['name'])
    indices = indices[~indices.index.duplicated(keep='last')]
    # title'ın index'ini yakalama
    movie_index = indices[title]
    # title'a gore benzerlik skorlarını hesapalama
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # kendisi haric ilk 10 filmi getirme
    movie_indices = similarity_scores.sort_values("score", ascending=False)[0:11].index
    return dataframe['name'].iloc[movie_indices]

def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe.loc[:, 'description'] = dataframe['description'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataframe['description'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

#'accommodates', 'beds', 'amenities',
# def data_prep(neighbourhood, selected_price):
def data_prep(df):


    df['price'] = df['price'].replace({r'^\$': '', r',': ''}, regex=True).astype(float)
    df['price'] = (df.groupby(['neighbourhood_cleansed'])['price']
                   .transform(lambda x: x.replace(0, np.nan).fillna(x.mean()).replace(np.nan, 0)))

    df.loc[:, 'description'] = df['description'].fillna('name')

    df = df[df['has_availability'] == 't']  # aktif olanları gösterelim



    # df['description'] = df['description'].fillna(df['name'])
    # null_count_desc = df['description'].isnull().sum()
    # if null_count_desc != 0:
    #     df.loc[:, 'description'] = df['description'].fillna(' ')


    # bins = [0, 10, 50, 150, 500, 1000, np.inf]
    # labels = ['0-10', '10-50', '50-150', '150-500', '500-1000', '1000+']
    # df['price_range'] = pd.cut(df['price'], bins=bins, labels=labels, right=False)

    # selected_price = 200
    # # Kullanıcının fiyatının hangi kategoriye düştüğünü belirle
    # user_label = pd.cut([selected_price], bins=bins, labels=labels)[0]
    # # Kategoriye göre filtreleme yap
    # df = df[df['price_range'] == user_label]
    # if df.shape[0] != 0:
    #     neighbourhood = 'Spacious home w roof garden'
    #     df = df[df['neighbourhood_cleansed'] == neighbourhood]
    #     return df
    # else:
    #     return df

    return df


def main():

    df = pd.read_csv('datasets/Amsterdam_Datasets/all_listings.csv')

    df = df[
        ['id', 'name', 'description', 'host_id', 'host_name', 'host_since', 'neighbourhood_cleansed',
         'property_type', 'room_type', 'bedrooms', 'price', 'minimum_nights','maximum_nights',
         'has_availability', 'first_review', 'last_review', 'availability_30', 'availability_365',
         'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
         'review_scores_communication', 'review_scores_location', 'review_scores_checkin', 'instant_bookable',
         'reviews_per_month']]

    # df = data_prep(neighbourhood, selected_price)
    prep_df = data_prep(df)
    # if (prep_df.shape[0] == 0):
    #     return print('otur evinde be yaaaa')


    # selected_price = 300
    # neighbourhood = 'De Baarsjes - Oud-West'
    user_input_name = 'Room Charlotte, not for parties'
    cosine_sim = calculate_cosine_sim(prep_df)
    recommend_names = content_based_recommender(user_input_name, cosine_sim, prep_df)
    return print(recommend_names)



if __name__ == "__main__":
    print("İşlem başladı")
    main()



    # main(kullanıcının seçtiği isim gelmeli)  # ekranda name değişkenini kullanıcının seçmesi için gösterelim. Oradan gelecek ismi fonksiyona gönderelim




    #
    # num_vars = ['number_of_reviews', 'review_scores_rating', 'review_scores_location','review_scores_value',
    #             'review_scores_accuracy', 'review_scores_cleanliness','review_scores_communication','review_scores_checkin', 'reviews_per_month']
    #
    # # df_plot = df_model_merged.select_dtypes(include=[np.number])
    # df_plot = df[num_vars]
    #
    # matrix = np.triu(df_plot.corr())
    #
    # plt.figure(figsize=(12, 9))
    # sns.heatmap(df_plot.corr(), annot=True, mask=matrix,cmap='coolwarm', linewidths=.5, fmt='.1f',vmin=-1, vmax=1)

#
# 6170                     Luxueus appartement met uitzicht
# 6163                 Appartement in Stadsdeel Zuid — Pijp
# 6164      Stylish & light appartment alongside Westerpark
# 6165             Big and cosy apartment with amazing view
# 6167                          Houseboat-Amsterdam-classic
# 6168    Luxurious & Stylish 2floor + sunny big roofter...
# 6169                             Apartment in trendy East
# 6171                Stylish apartment | 4P | Canal views!
# 6161    Hip stylish one-bedroom apartment Amsterdam South
# 6172                          Apartment in Amsterdam West
#           Comfortable double room
# 2443                               Comfortable double room
# 16296                    Charming Studio with Roof Terrace
# 3000                               Comfortable double room
# 12549     Modern Houseboat | Jordaan area | A unique stay!
# 17955    Central, big window, rear room with private bath.
# 17411    Central, big window, rear room with private bath.
# 2514                Amstel Nest - an urban retreat for two
# 17049             Centre, canal view with private bathroom
# 14335    Sonnenberg - Canal side & view - Private & Cen...




# Kullanıcıdan name girişi al (örnek olarak 'Beautiful apartment in canal house' kullanılmıştır)
# user_input_name = 'Beautiful apartment in canal house'
#
# user_input_name = 'Comfortable double room'
# # Kullanıcının girdiği name'e göre veri setini filtreleme
# filtered_df = df[df['name'].str.contains(user_input_name, case=False, na=False)]
#
# # Benzersiz `name` değerlerini seçme
# unique_filtered_df = filtered_df.drop_duplicates(subset=['name'])
#
# # `number_of_reviews` sütununa göre sıralama
# sorted_unique_df = unique_filtered_df.sort_values(by='number_of_reviews', ascending=False)
#
# print(sorted_unique_df[['name', 'number_of_reviews']])





