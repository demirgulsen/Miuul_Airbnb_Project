import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import re

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 300)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


##########################################################################################
# AYKIRI DEĞERLER
def outlier_thresholds(dataframe, col_name, q1=0.15, q3=0.85):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


##########################################################################################
# Değişken Tipleri
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

##########################################################################################
# Aykırı Değerlerin Kendilerine Erişmek
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

# Aykırı Değerleri  Alt-Üst Sınırlarına Göre Değiştirmek
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

##########################################################################################
le = LabelEncoder()
def label_encoder(dataframe, binary_col):
    dataframe[binary_col] = le.fit_transform(dataframe[binary_col])
    return dataframe
##########################################################################################
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
##########################################################################################

def amsterdam_data_prep(awd_df):

    # En önemli amenities sütunlarını ekleyelim
    awd_df = awd_df.assign(
        has_private_entrance=awd_df['amenities'].apply(lambda x: x.find('Private entrance') != -1),
        has_self_checkin=awd_df['amenities'].apply(lambda x: x.find('Self check-in') != -1),
        has_kitchen=awd_df['amenities'].apply(lambda x: x.find('Kitchen') != -1),
        has_bathtub=awd_df['amenities'].apply(lambda x: x.find('Bathtub') != -1),
        has_host_greeting=awd_df['amenities'].apply(lambda x: x.find('Host greets you') != -1),
        has_dishwasher=awd_df['amenities'].apply(lambda x: x.find('Dishwasher') != -1),
        has_longterm=awd_df['amenities'].apply(lambda x: x.find('Long term stays allowed') != -1),
        has_fireplace=awd_df['amenities'].apply(lambda x: x.find('Indoor fireplace') != -1),
        has_parking=awd_df['amenities'].apply(lambda x: x.find('Free parking on premises') != -1)
    )
    awd_df = awd_df.drop(['amenities'], axis=1)

    # Değişknelerin tipini getirelim
    cat_cols, num_cols, cat_but_car = grab_col_names(awd_df)
    num_cols = [col for col in num_cols if col not in ['id','host_id','calendar_last_scraped']]

    # Aykırı değerleri temizleyelim
    for col in num_cols:
        replace_with_thresholds(awd_df, col)

    del_null = ['beds','bedrooms', 'bathrooms_text']
    awd_df = awd_df.dropna(subset=del_null)

    awd_df['combine_beds_bedrooms_acommodates'] = awd_df['beds'] + awd_df['bedrooms'] + awd_df['accommodates']
    awd_df.drop(['beds', 'bedrooms', 'accommodates'], axis=1, inplace=True)

    awd_df['price'] = awd_df['price'].replace({r'^\$': '', r',': ''}, regex=True).astype(float)

    awd_df['calendar_last_scraped'] = pd.to_datetime(awd_df['calendar_last_scraped'], errors='coerce')

    # property_type' ı kategorilere göre gruplama
    property_type_counts = awd_df['property_type'].value_counts()
    property_type_counts[property_type_counts < 11].index

    ent_rent_unit_types = set(list(set(col for col in awd_df['property_type'].values if 'Entire rental unit' in col)))
    ent_condo_types = set(list(set(col for col in awd_df['property_type'].values if 'Entire condo' in col)))
    ent_home_types = set(list(set(col for col in awd_df['property_type'].values if 'Entire home' in col)))

    # 'Entire' içeren türler (belirtilen türler hariç)
    entire_types = set(col for col in awd_df['property_type'].values if 'Entire' in col)
    entire_types = entire_types - (ent_rent_unit_types | ent_condo_types | ent_home_types)

    private_types = list(set(col for col in awd_df['property_type'].values if 'Private' in col))
    shared_types = list(set(col for col in awd_df['property_type'].values if 'Shared' in col))
    room_in_types = list(set(col for col in awd_df['property_type'].values if 'Room in' in col))
    other_types = list(set([col for col in awd_df['property_type'].values if (col not in ent_rent_unit_types) and (col not in ent_condo_types) and (col not in ent_home_types) and\
                            (col not in entire_types)and (col not in private_types) and (col not in shared_types) and (col not in room_in_types)]))
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

    def categorize(property_type):
        for category, values in categories.items():
            if property_type in values:
                return category
        return 'Unknown'

    # `property_type`'ı kategorilere ayırma
    awd_df['property_type_category'] = awd_df['property_type'].apply(categorize)

    awd_df = awd_df.drop('property_type', axis=1)

    # price' ı ortalama değerlerle
    awd_df['price'] = (awd_df.groupby(['neighbourhood_cleansed', 'room_type', 'property_type_category'])[
                           'price']  # model başarısı için property_type buna göre de denenebilir
                       .transform(lambda x: x.replace(0, np.nan).fillna(x.mean()).replace(np.nan, 0)))



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

    # bathrooms_text'in sayısal değerlerini çıkarıp kendisine atalım
    awd_df['bathroom_count'] = awd_df['bathrooms_text'].apply(extract_bathroom_number)
    awd_df = awd_df.drop('bathrooms_text', axis=1)

    # review sütunlarındaki null değer sayısı 0' dan büyük olan kayıtları silelim
    rev_cols = [col for col in awd_df.columns if awd_df[col].isnull().sum() > 0 and 'review' in col]
    awd_df = awd_df.dropna(subset=rev_cols)

    #Önem düzeyine sahip review değişkenlerini ağırlıklandırıp genel bir score değişkeni oluşturalım
    weights = {
        'review_scores_rating': 0.20,
        'review_scores_communication': 0.20,
        'review_scores_location': 0.50,
    }
    # Bu ağırlıklarla genel skoru hesaplayın
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

    cat_cols, num_cols,  cat_but_car = grab_col_names(awd_df)
    for col in num_cols:
        replace_with_thresholds(awd_df,col)

    # Yeni Özellikler Ekleyelim
    awd_df['price_range'] = pd.cut(awd_df['price'], bins=[0, 50, 100, 200, 500, np.inf],
                                   labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    awd_df['nightly_price'] = awd_df['price'] / awd_df['minimum_nights']
    awd_df['host_active_years'] = (pd.to_datetime(date.today()) - pd.to_datetime(awd_df['host_since'])).dt.days // 365
    awd_df['review_count'] = awd_df['number_of_reviews'] + awd_df['reviews_per_month'] * 12

    # latitude ve longitude değişkenlerini kullanarak kümeleme yapalım
    coords = awd_df[['latitude', 'longitude']]
    kmeans = KMeans(n_clusters=10).fit(coords)
    awd_df['location_cluster'] = kmeans.labels_

    awd_df = awd_df.drop(['latitude', 'longitude'], axis=1)

    # Encoder işlemlerinden önce silinecek diğer değişkenleri silelim
    awd_df = awd_df.drop(['id', 'host_id', 'host_name', 'calendar_last_scraped', 'host_since', 'neighbourhood_cleansed'], axis=1)

    ##################################
    final_df = awd_df.copy()
    ##################################

    # Encoder işlemlerini uygulayalım
    cat_cols, num_cols,  cat_but_car = grab_col_names(final_df)

    # 2 SINIFLI DEĞİŞKENLERE LABEL ENCODER UYGULAYALIM


    # 2 sınıflı kategorik değişkenleri seçmeliyiz
    binary_cols = [col for col in final_df.columns if final_df[col].dtype not in ['int64', 'float']
                   and final_df[col].nunique() == 2]

    for col in binary_cols:
        label_encoder(final_df, col)

    # room_type' a özel olarak label encoding uygulayalım
    room_type_mapping = {
        'Entire home/apt': 0,
        'Private room': 1,
        'Hotel room': 2,
        'Shared room': 3
    }
    final_df['room_type'] = final_df['room_type'].map(room_type_mapping)

    final_df['room_type'] = le.fit_transform(final_df['room_type'])

    # Diğer kategorik değişkenlere one-hot encoder uygulayalım
    ohe_cols = [col for col in final_df.columns if 10 >= final_df[col].nunique() > 2]

    final_df = one_hot_encoder(final_df, ohe_cols)

    # Standart Scaler Uygulayalım
    cat_cols, num_cols, cat_but_car = grab_col_names(final_df)
    num_cols = [col for col in num_cols if "price" not in col]

    scaler = StandardScaler()
    final_df[num_cols] = scaler.fit_transform(final_df[num_cols])

    return final_df


def main():
    df = pd.read_csv('datasets/all_listings.csv')

    df = df[
        ['id', 'host_id', 'host_name', 'host_since', 'neighbourhood_cleansed', 'latitude', 'longitude', 'property_type', \
         'room_type', 'accommodates', 'bathrooms_text', 'bedrooms', 'beds', 'amenities', 'price', 'minimum_nights',
         'maximum_nights', \
         'availability_30', 'availability_365', 'calendar_last_scraped', 'number_of_reviews', 'review_scores_rating', \
         'review_scores_communication', 'review_scores_location', 'instant_bookable', 'reviews_per_month']]

    ###################################################################################################

    final_model = amsterdam_data_prep(df)
    final_model.to_csv('datasets/final_model.csv', index=False)
    return final_model



if __name__ == "__main__":
    print("İşlem başladı")
    main()

