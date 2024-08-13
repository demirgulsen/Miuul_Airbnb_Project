
'''
İlk aşamada;
rev_detail = pd.read_csv('datasets/reviews_comment.csv') dosyasından yorumları çekip
SentimentIntensityAnalyzer yöntemi kullanılarak yorumlardaki kelimeler üzerinde duygu analizi yapıldı

(Textblob ve SentimentIntensityAnalyzer yöntemlerinin her ikiside karşılaştırıldı ve en iyi sonucu veren yöntemin SentimentIntensityAnalyzer
olduğuna karar verilerek işlemler bu yöntem üzerinden yapıldı)
Daha sonra;
amsterdam_all_listing = pd.read_csv('datasets/all_listings.csv')
ve
rev_detail = pd.read_csv('datasets/reviews_comment.csv')

veri setleri birleştirildi ve kullanılacak değişkenler belirlenip veri seti EDA aşamsından geçirilerek hazır hale getirildi
Daha sonra;
en çok yorum alan ve rating score değerine sahip evler hesaplanarak elde edildi.
Buradan top 20 listesi çıkarılarak kullanıcıya önermek üzere hazır hale getirildi ve final_model.csv dosyası olarak kaydedildi
Son olarak final_model.csv veri seti kullanılarak recommend işlemleri gerçekleştirildi

'''

import numpy as np
import pandas as pd
import joblib

pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_colwidth', None)  # URL'lerin tamamını göster

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

##########################################################################################################
weights = {
    'review_scores_value': 0.2,
    'review_scores_rating': 0.2,
    'review_scores_accuracy': 0.15,
    'review_scores_cleanliness': 0.15,
    'review_scores_communication': 0.15,
    'review_scores_location': 0.10,
    'review_scores_checkin': 0.05
}

def get_top_recommendations(df, weights, min_reviews=10, min_reviews_per_month=0.1, top_n=1000):
    """
    Verilen dataframe üzerinde ağırlıklı genel skoru hesaplar ve en iyi önerileri getirir.

    :param df: İnceleme verilerini içeren dataframe
    :param weights: İnceleme puanları için ağırlıklar sözlüğü
    :param min_reviews: Filtreleme için minimum yorum sayısı
    :param min_reviews_per_month: Filtreleme için minimum aylık yorum sayısı
    :param top_n: Gösterilecek öneri sayısı
    :return: Ağırlıklı skora göre sıralanmış en iyi öneriler
    """
    # Kopya oluştur
    df_copy = df.copy()

    # Genel skoru hesapla
    df_copy['weighted_score'] = (
        df_copy['review_scores_value'] * weights.get('review_scores_value', 0) +
        df_copy['review_scores_rating'] * weights.get('review_scores_rating', 0) +
        df_copy['review_scores_accuracy'] * weights.get('review_scores_accuracy', 0) +
        df_copy['review_scores_cleanliness'] * weights.get('review_scores_cleanliness', 0) +
        df_copy['review_scores_communication'] * weights.get('review_scores_communication', 0) +
        df_copy['review_scores_location'] * weights.get('review_scores_location', 0) +
        df_copy['review_scores_checkin'] * weights.get('review_scores_checkin', 0)
    )

    # Filtreleme
    filtered_df = df_copy[
        (df_copy['number_of_reviews'] > min_reviews) &
        (df_copy['reviews_per_month'] > min_reviews_per_month)
    ]

    # Skora göre sıralama yap
    top_recommendations = filtered_df.sort_values(
        by=['weighted_score', 'number_of_reviews', 'reviews_per_month'],
        ascending=[False, False, False]
    ).head(top_n)

    return top_recommendations
##########################################################################################################
analyzer = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    return analyzer.polarity_scores(text)['compound']  # -1 (negatif) ile 1 (pozitif) arasında bir değer


def best_comments(df):
    df['vader_sentiment'] = df['comments'].apply(get_vader_sentiment)
    vader_sentiments = df['vader_sentiment'].tolist()
    avg_sentiment2 = df.groupby('id')['sentiment2'].mean().reset_index()

    top_comments = avg_sentiment2.sort_values(by='sentiment2', ascending=False).head(10)

##########################################################################################################
def data_prep(df):
    df['description'] = df['description'].fillna(df['name'])

    df['price'] = df['price'].replace({r'^\$': '', r',': ''}, regex=True).astype(float)
    df['price'] = (df.groupby(['neighbourhood_cleansed'])['price']
                   .transform(lambda x: x.replace(0, np.nan).fillna(x.mean()).replace(np.nan, 0)))

    rev_cols = [col for col in df.columns if 'review' in col]
    for col in rev_cols:
        df.loc[:, col] = df[col].fillna(0)

    df = df.dropna()
    df = df[df['has_availability'] == 't']

    return df


def main(name):

    listing_detail = pd.read_csv('datasets/all_listings.csv')
    reviews_comment = pd.read_csv('datasets/reviews_comment.csv')

    listing_detail = listing_detail.rename(columns={'id': 'listing_id'})
    df = reviews_comment.merge(listing_detail, how="left", on="listing_id")

    df = df[
        ['listing_id', 'id', 'name', 'description', 'date', 'comments', 'listing_url','picture_url','host_id', 'host_name', 'host_since', 'neighbourhood_cleansed',
         'property_type', 'room_type', 'bedrooms', 'price', 'minimum_nights', 'comments','maximum_nights','price',
         'has_availability', 'first_review', 'last_review', 'availability_30', 'availability_365',
         'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
         'review_scores_communication', 'review_scores_location', 'review_scores_checkin', 'instant_bookable',
         'reviews_per_month','review_scores_value']]

    prep_df = data_prep(df)

#######################################################################################################################
    # En çok yorum alan ve en iyi ratinge sahip listeyi getirir
    new_df = get_top_recommendations(prep_df, weights)
    # Skora göre sıralama yap
    top_recommendations = new_df.sort_values(by=['weighted_score', 'number_of_reviews', 'reviews_per_month'],
                                             ascending=False)
    # En yüksek genel skorları listeleme
    top_scores = top_recommendations[['listing_id', 'name', 'listing_url', 'picture_url','number_of_reviews', 'weighted_score']].sort_values(
        by='weighted_score', ascending=False)
##################################################################################################################3
    # rev_df = reviews_comment.copy()
    # rev_df.dropna(inplace=True)
    #
    # # İki veri çerçevesini 'listing_id' sütunu üzerinden birleştirme
    # merged_df = pd.merge(top_scores, rev_df, on='listing_id', how='inner').sort_values(by='weighted_score', ascending=False)

###########################################################################################################################3
###########################################################################################################################3
    # Normalizasyon işlemi
    top_scores['normalized_reviews'] = top_scores['number_of_reviews'] / top_scores['number_of_reviews'].max()
    top_scores['normalized_score'] = top_scores['weighted_score'] / top_scores['weighted_score'].max()

    # Ağırlıklı puanı hesapla (örneğin: %50 yorum sayısı + %50 skor)
    top_scores['weighted_value'] = 0.5 * top_scores['normalized_reviews'] + 0.5 * top_scores['normalized_score']

    # Sonuçları sıralayıp en yüksekten düşüğe doğru 20 sonuç al
    result_df = top_scores.sort_values(by='weighted_value', ascending=False).drop_duplicates(subset='listing_id')
    top_20_weighted = result_df[['listing_id', 'name', 'listing_url', 'number_of_reviews', 'weighted_score', 'weighted_value','picture_url']].head(21)

    # top_20_comments.to_csv('datasets/Amsterdam_Datasets/Top_20_comments.csv', index=False)
    joblib.dump(top_20_weighted, "top_20_comment.pkl")
    # print(top_20_weighted)
    return top_20_weighted




if __name__ == "__main__":
    print("İşlem başladı")
    name = 'Comfortable double room'
    main(name)




#                listing_id                                                name                                      listing_url  number_of_reviews  weighted_score  weighted_value
# 59250  572146163845162871  Light Travelin,Private Accommodation Near Van Gogh  https://www.airbnb.com/rooms/572146163845162871                224           4.983           0.998
# 28876            22529423       Former Rembrandt Workshop 2 Br Canal View Apt            https://www.airbnb.com/rooms/22529423                101           4.987           0.724
# 5448             39918326  Clean cozy canalview room + bathroom for 1 person❤            https://www.airbnb.com/rooms/39918326                 96           4.991           0.713
# 16312            40363985     Clean private Amsterdam loft in Museumdistrict.            https://www.airbnb.com/rooms/40363985                 64           4.988           0.642
# 31962            50564592         Ideal for families, 1 double, 2 single beds            https://www.airbnb.com/rooms/50564592                 35           4.983           0.576
# 13841            23819110           Spacious studio with terrace on waterside            https://www.airbnb.com/rooms/23819110                 32           4.991           0.571
# 3135   936814096393617140    Spacious and characteristic loft with canal view  https://www.airbnb.com/rooms/936814096393617140                 31           4.994           0.569
# 1670             30905786        City Centre Boutique Deluxe Room in Monument            https://www.airbnb.com/rooms/30905786                 29           4.994           0.564
# 14748  660577560760890684     Beautiful Canal View Loft - City Center Jordaan  https://www.airbnb.com/rooms/660577560760890684                 21           4.990           0.546
# 1196              6077424                    Charming House in Museum Quarter             https://www.airbnb.com/rooms/6077424                 19           4.997           0.542
# 15525            13491924        Jordaan Penthouse Royal blue on top location            https://www.airbnb.com/rooms/13491924                 17           4.988           0.537
# 3584   637673199251857238           Bright Amsterdam apartment in lovely area  https://www.airbnb.com/rooms/637673199251857238                 15           4.993           0.533
# 0      667315149461706557                          Cozy apartment in the Pijp  https://www.airbnb.com/rooms/667315149461706557                 14           5.000           0.531
# 14900            43904607                       Lovely apartment in Amsterdam            https://www.airbnb.com/rooms/43904607                 14           4.989           0.530
# 29905  600555326893149601  Chic 2-bedroom apartment in the heart of Amsterdam  https://www.airbnb.com/rooms/600555326893149601                 14           4.986           0.530
# 30224  915725944348188008                Cosy living&bedroom with nice energy  https://www.airbnb.com/rooms/915725944348188008                 14           4.986           0.530
# 1446             53041583  Charming 1 bedroom Apartment in Amsterdam Oud West            https://www.airbnb.com/rooms/53041583                 13           4.996           0.529
# 3777              9078341                         Modern Apartment in De Pijp             https://www.airbnb.com/rooms/9078341                 13           4.992           0.528
# 4075   633354846526546363     Rustig appartement in levendige buurt “De Pijp”  https://www.airbnb.com/rooms/633354846526546363                 13           4.992           0.528
# 31240            25021801                                     B&B Urban Oasis            https://www.airbnb.com/rooms/25021801                 13           4.984           0.527
