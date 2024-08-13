import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px

pd.set_option('display.max_colwidth', None)  # URL'lerin tamamını göster

# python -m streamlit run main.py

from amsterdam_content_based_recommender import content_based_recommender_Word2Vec, calculate_cosine_sim_Word2Vec, data_prep

st.set_page_config(layout='wide', page_title='Miuulbnb', page_icon='🏘️')

@st.cache_data
def get_data():
    dataframe = pd.read_csv('datasets/all_listings.csv')
    return dataframe

@st.cache_data
def get_comment():
    comment = joblib.load('top_20_comment.pkl')
    return comment

@st.cache_data
def get_pipeline():
    pipeline = joblib.load('final_model.pkl')
    return pipeline


@st.cache_data
def get_superhost():
    superhost = joblib.load('top_superhosts.pkl')
    return superhost


@st.cache_data
def get_top():
    top = joblib.load('top_10_scores.pkl')
    return top

df = get_data()
model = get_pipeline()
superhost = get_superhost()
comment = get_comment()
superhost = get_superhost()




st.title('🏘️:rainbow[Miuul]:blue[B]n:violet[G]')

home_tab,  recommendation_tab, random_tab, predict_tab, super_host, comment_tab = st.tabs(["Ana Sayfa", "Öneri Sistemi", "Rastgele", "Ev Fiyatları", "Superhostlar", "Yorumlar"])

# home tab
col1, col2 = home_tab.columns([1, 1])
col1.image("https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExNnFvYzRoc3RwN2wyYm4waTFib2o2YTlvdmEwcXpoc2x5amdibmJzdiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/oDXJPqRpJTf7qsxc3l/giphy.gif")
col1.subheader("Aradığınız Ev İşte Burada")
col1.markdown('*MiuulBng şirketi olarak sizlere istediğiniz özelliklere sahip evleri sunuyoruz. Ayrıca evini kiraya vermek isteyenlere evleri için en uygun kira tutarını belirliyoruz*')
col2.subheader("Tam Aradığınız Yerdesiniz!")
col2.markdown("Bizi tercih eden müşterilerimizin ve ev sahiplerimizin mutluluğuna ortak oluyoruz ")
col2.image("https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExMnl6dGdkNHNqNXF2ajJ5bGZta2VnajNhYzUycDRvdHg3ZmJhMjNtcCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/d2d3Tx0Ltc3HIZJgua/giphy.gif")
col1.image("https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExNDd4OHJ0Zm5ubW9kcGVqODNzYjZoaG1zMHIyMGp1ZXV6emJoN2p0NiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/WoFPRKUTbXMClPCSjl/giphy.gif")
col1.subheader("En İyi Ev Sahiplerine mi Sahibiz")
col1.markdown("Aylık olarak güncellediğimiz en iyi 10 ev sahibi listesini gördünüz mü? En tatlı ve yardımsever ev sahipleri bizimle çalışıyor.")


#######################################################################################################################
# recommender tab  ####################################################################################################

col4, col5, col6 = recommendation_tab.columns([1, 1, 1])
name = col4.selectbox(
    "Ev ismi seçiniz",
    ("Clean, cozy and beautiful home in Amsterdam!", "Groep Accommodation Sailing Ship", "45m2 studio in historic citycenter (free bikes)", "Green oases in the city of Amsterdam", "Dubbel kamer, ontbijt is inbegrepen",
     "Cozy apartment in Amsterdam West", "Groep Accommodation Sailing Ship", "Super location & Baby-friendly apartment", "Spacious Family Home in Centre With Roof Terrace", "Room Aris (not for parties)",
     "Vondelpark apartment", "Prestige room downtown", "Quiet room with free car parking", "Five Person Room", "Intimate studio", "Room in Amsterdam + free parking", "Appartement in oud west",
     "Cozy house with green garden near Amsterdam centre", "The Little House in the garden",'Sunny Canal Apartment'))


prep_df = data_prep(df)
price = col5.number_input("Fiyat", min_value=0, max_value=int(prep_df.price.max()))

#####################################################################
# ikon ekleme
warning_icon = "👉🏻 🧘🏻‍♀️🧘🏻‍♂️️"

#####################################################################
cosine, model_w2v = calculate_cosine_sim_Word2Vec(prep_df)

col6.write(' ')
col6.write(' ')
if col6.button('Öner'):
    recommendations = content_based_recommender_Word2Vec(name, price, cosine, prep_df)
    recommendation_tab.markdown('**<h3 style="font-size: 24px;color: #FF1493;">Seçilen İsme Göre Önerilerimiz</h3>**', unsafe_allow_html=True)

    if isinstance(recommendations, str):
        recommendation_tab.video("https://www.youtube.com/watch?v=cJM56tlv2Q4", autoplay=True)
        # print('otur evinde be ya !!!')
    elif recommendations is None:
        recommendation_tab.write(f"Bütçeniz yetersiz !!{warning_icon}")
        recommendation_tab.image(
            "https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExbzJtb3YyMGowbWU5d3gwbDYzNnYwcTE3eXV2bXgwZGJ5Mm4weWp5dCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/PyoyQRPyZXYq7mfxxs/giphy.gif")
        # recommendation_tab.write("Bütçeniz yetersiz !!")
    else:
        # app_r.display_cards(recommendations)
        num_cols = 3
        with recommendation_tab.container():
            for i in range(0, len(recommendations), num_cols):
                cols = st.columns(num_cols)
                for j in range(num_cols):
                    index = i + j
                    if index < len(recommendations):
                        row = recommendations.iloc[index]
                        with cols[j]:
                            st.markdown(f"""
                            <div style="border: 1px solid #dee2e6; border-radius: 0.25rem; margin-bottom: 1rem;">
                                <img src="{row['picture_url']}" class="card-img-top" alt="{row['name']}" style="height: 350px; width: 100%; object-fit: cover;">
                                <div style="padding: 1rem;">                            
                                    <h5 class="card-header" style="font-weight: bold; color: #C71585;">{row['name']}</h5>
                                    <p>
                                        <b>Evin URL Bilgisi:</b><a href="{row['listing_url']}" target="_blank">{row['listing_url']}</a><br>                                
                                        <span class="card-text" style="font-weight: bold; color: #C71585;"><b>Fiyat: </b>{round(float(row['price']), 2)}</span><br/>                                                                               
                                    </p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

#######################################################################################################################
# best comment tab  ###################################################################################################
home_tab.markdown('**<h3 style="font-size: 24px;margin-top:30px;color: #FF1493;">En çok Beğenilen Evlerimiz</h3>**', unsafe_allow_html=True)

num_cols = 3
with home_tab.container():
    for i in range(0, len(comment), num_cols):
        cols = st.columns(num_cols)
        for j in range(num_cols):
            index = i + j
            if index < len(comment):
                row = comment.iloc[index]
                with cols[j]:
                    st.markdown(f"""
                    <div style="border: 1px solid #dee2e6; border-radius: 0.25rem; margin-bottom: 1rem;">
                        <img src="{row['picture_url']}" class="card-img-top" alt="{row['name']}" style="height: 350px; width: 100%; object-fit: cover;">
                        <div style="padding: 1rem;">                            
                            <h5 class="card-header" style="font-weight: bold; color: #C71585;">{row['name']}</h5>
                            <p>
                                <b>Evin URL Bilgisi:</b> <a href="{row['listing_url']}" target="_blank">{row['listing_url']}</a><br>
                                <b>Yorum Sayısı:</b> {row['number_of_reviews']}<br>
                                <b>Ortalama Değerlendirme Puanı:</b> {round(float(row['weighted_score']), 2)}
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

#######################################################################################################################
# random tab ##########################################################################################################
random_tab.markdown("**<h1 style='font-size: 24px;color: #FF1493;'>Nerede kalmak istediğiniz konusunda kararsız mısınız? Size en iyi öneride bulunmak için buradayız!</h1>**",
                    unsafe_allow_html=True)

# Sayfa düzeni için kolonları oluştur
col1, col2, col3, col4, col5 = random_tab.columns(5, gap="small")
columns = [col1, col2, col3, col4, col5]
empty_col1, empty_col2 = random_tab.columns([1,1])

# Sayfa düzeni için kolonları oluştur
col1, col2, col3, col4, col5 = random_tab.columns(5, gap="small")
columns = [col1, col2, col3, col4, col5]

# Butonun tam genişlikte ve ekranın en üst kısmında olmasını sağlamak için

if empty_col1.button("Rastgele Öner"):
    empty_col1.markdown('**<h3 style="font-size: 24px;margin-top:30px;color: #008B8B;">Sizler için Derlediklerimiz</h3>**', unsafe_allow_html=True)

    random_home = df[~df["price"].isna()].sample(5)
    for i, col in enumerate(columns):
        with col:
            col.markdown(f"""
                <div style="border: 1px solid #dee2e6; border-radius: 0.25rem; margin-bottom: 1rem; padding: 10px;">
                    <img src="{random_home.iloc[i]['picture_url']}" alt="{random_home.iloc[i]['name']}" style="height: 150px; width: 100%; object-fit: cover;">
                    <div style="padding: 10px;">
                        <h5 style="color: #008B8B;">{random_home.iloc[i]['name']}</h5>
                        <p>
                            <b>Ev Sahibi URL Bilgisi:</b> <a href="{random_home.iloc[i]['listing_url']}" target="_blank">{random_home.iloc[i]['listing_url']}</a><br>
                            <b>Açıklama:</b> {random_home.iloc[i]['description'] if pd.notna(random_home.iloc[i]['description']) else 'N/A'}<br>
                            <b style="color: #DC143C;font-size:17px;">Fiyat:</b> <span style="color: #DC143C;font-size:17px;">{round(float(random_home.iloc[i]['price']), 2)} $</span>
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

random_tab.markdown('</div>', unsafe_allow_html=True)
#######################################################################################################################
# prediction tab ######################################################################################################
predict_tab.markdown("**<h1 style='font-size: 24px;color: #FF1493;'>Amsterdam'da evimi kaça kiraya verebilirim?</h1>**",
                    unsafe_allow_html=True)


predict_tab.markdown(
    "Amsterdam'da bulunan evinizi Miuulbng güvencesiyle kiraya vermek istiyorsanız hemen evinizin değerini öğrenin!!")
predict_tab.image("https://wise.com/imaginary-v2/8fab8a52eaaa8b543e70cafe5cf716d8.jpg?width=1200")

col6, col7 = predict_tab.columns([1, 1])

neighbourhood = col7.selectbox(" Mahalle Adı", ['Centrum-Oost', 'Westerpark', 'Centrum-West', 'Oud-Oost',
       'Oostelijk Havengebied - Indische Buurt', 'Buitenveldert - Zuidas',
       'Bos en Lommer', 'IJburg - Zeeburgereiland', 'Zuid',
       'De Pijp - Rivierenbuurt', 'Slotervaart', 'Noord-Oost',
       'De Baarsjes - Oud-West', 'Watergraafsmeer', 'Oud-Noord',
       'Noord-West', 'Geuzenveld - Slotermeer', 'De Aker - Nieuw Sloten',
       'Osdorp', 'Bijlmer-Centrum', 'Gaasperdam - Driemond',
       'Bijlmer-Oost'])

accommodates = col6.slider('Kaç kişi kalabilir?',min_value=0.0, max_value=16.0, step=1.0)

room_type = col7.selectbox("Evinizin türü", ["Entire home/apt", "Private room", "Hotel room","Shared room"])
#room_type = col7.radio("Odanızın Tipi", ['room_type_Hotel room', 'room_type_Private room', 'room_type_Shared room'])

bathrooms = col6.slider('Evinizde kaç banyo var?',min_value=0.0, max_value=16.0, step=1.0)

bedrooms = col6.slider('Evinizde kaç oda var?',min_value=0.0, max_value=17.0, step=1.0)

beds = col6.slider('Evinizde kaç yatak var?',min_value=0.0, max_value=17.0, step=1.0)

minimum_nights = col7.number_input('Evinizde minimum kaç gece kalınabilir?',min_value=1.0, max_value=17.0, step=1.0)

maximum_nights = col7.number_input('Evinizde maximum kaç gece kalınabilir?',min_value=1.0, max_value=17.0, step=1.0)

col7.write("Evinizde olan özellikler")

has_private_entrance = col7.checkbox("Özel Giriş")
has_self_checkin = col7.checkbox("Checkin")
has_kitchen = col7.checkbox("Mutfak")
has_bathtub = col7.checkbox("Küvet")
has_host_greeting = col7.checkbox("Karşılama")
has_dishwasher = col7.checkbox("Bulaşık Makinesi")
has_longterm = col7.checkbox("Uzun süreli konaklama")
has_fireplace = col7.checkbox("Şömine")
has_parking = col7.checkbox("Otopark")

col6.write("Superhost musunuz?")
host_is_superhost = col6.checkbox("Evet Superhostum")

col6.write("Ev şu an kiralanabilir mi")
instant_bookable = col6.checkbox("Evet")


if col7.button('Öneri'):
    col7.write(f'Ortalama olarak kiranız şu şekilde olmalıdır:')

#######################################################################################################################
# superhost tab #######################################################################################################

super_host.markdown('**<h1 style="font-size: 24px;color: #FF1493;">En İyi Ev Sahiplerimiz</h1>**',
                    unsafe_allow_html=True)

col1, col2= super_host.columns([1, 1])
col1.image(
    "https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExNDd4OHJ0Zm5ubW9kcGVqODNzYjZoaG1zMHIyMGp1ZXV6emJoN2p0NiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/WoFPRKUTbXMClPCSjl/giphy.gif")
col2.subheader("Amsterdam'ın En İyi Ev Sahiplerine Sahibiz")
col2.markdown(
    "Amsterdam, sadece tarihi ve kültürel zenginlikleriyle değil, aynı zamanda mükemmel konaklama deneyimleriyle de tanınır. Bu sunumda, şehrin en iyi ev sahiplerini sizlere tanıtmaktan büyük mutluluk duyuyoruz. "
    "Her biri, misafirlerini özel hissettirmek için olağanüstü bir misafirperverlik sergileyen, profesyonel ve deneyimli ev sahipleridir. Bu kişiler, konaklamanın ötesinde, misafirlerine unutulmaz bir deneyim sunmak için her detayı titizlikle planlar. Sunduğu kişiselleştirilmiş hizmetler, derin yerel bilgisi ve konforlu yaşam alanları ile Amsterdam'da konaklamayı bir sanat haline getiriyorlar. Burada, size şehrin en misafirperver, profesyonel ve dikkatli ev sahiplerini tanıtacağız.")


# Başlık yazısını HTML ile stilize etme
col1.markdown(f"""
    <h2 style="color: #ff6347; font-size: 28px; font-weight: bold; text-align: left;margin-top:30px;">
        En İyi Ev Sahiplerimizi Sizler için Listeledik
    </h2>
""", unsafe_allow_html=True)


# Kartların yan yana düzenlenmesi için grid düzeni
cols = super_host.columns(2)  # 2 sütunlu bir düzen

for i, row in superhost.iterrows():
    col = cols[i % 2]  # Kolonları döndürmek için mod kullanın
    with col:
        col.markdown(f"""
        <div class="card" style="border: 1px solid #dee2e6; border-radius: 0.25rem; margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; padding: 10px;">
                <img src="{row['host_picture_url']}" class="card-img-top" alt="{row['host_name']}" style="height: 100px; width: 100px; object-fit: contain; border-radius: 50%; margin-right: 10px;">
                <h5 class="card-title" style="margin-left: 10px;">{row['host_name']}</h5>
            </div>
            <div class="card-body">
                <p class="card-text" style='margin-left: 1rem;'>
                    <b>Ev Sahibi URL Bilgisi:</b> <a href="{row['host_url']}" target="_blank">{row['host_url']}</a><br>
                    <b>Kaç Yıldır Ev Sahibi:</b> {row['years_as_host']}<br>
                    <b>Yorum Sayısı:</b> {row['number_of_reviews']}<br>
                    <b>Değerlendirme Puanı:</b> {row['review_scores_rating']}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

#######################################################################################################################
# comment tab #########################################################################################################
comment_tab.markdown('**<h1 style="font-size: 30px;color: #FF1493;">En Yüksek Skora Sahip ve En Güzel Yorumları Alan Evlerimiz</h1>**',
                    unsafe_allow_html=True)

comment_tab.markdown('**<h1 style="font-size: 20px;color: #008B8B;">Kullanıcılarımızın en çok sevdiği ve gitmekten aşırı derecede zevk aldığı evleri sizler için listeledik. Aşağıdaki evlerimiz en yüksek puanları almış olup en güzel yorumlara sahip olan evlerimizdir.</h1>**',
                    unsafe_allow_html=True)

# Kartları 3 sütunlu düzen içinde göster
num_cols = 3
with comment_tab.container():
    for i in range(0, len(comment), num_cols):
        cols = st.columns(num_cols)
        for j in range(num_cols):
            index = i + j
            if index < len(comment):
                row = comment.iloc[index]
                with cols[j]:
                    st.markdown(f"""
                    <div style="border: 1px solid #dee2e6; border-radius: 0.25rem; margin-bottom: 1rem;">
                        <img src="{row['picture_url']}" class="card-img-top" alt="{row['name']}" style="height: 350px; width: 100%; object-fit: cover;">
                        <div style="padding: 1rem;">
                            <h5 class="card-header" style="font-weight: bold; color: #C71585;">{row['name']}</h5>
                            <p>
                                <b>Ev Sahibi URL Bilgisi:</b> <a href="{row['listing_url']}" target="_blank">{row['listing_url']}</a><br>
                                <b>Yorum Sayısı:</b> {row['number_of_reviews']}<br>
                                <b>Ortalama Değerlendirme Puanı:</b> {round(float(row['weighted_score']), 2)}
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


