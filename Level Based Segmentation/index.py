import pandas as pd
import streamlit as st

st.set_page_config(layout="centered", page_title='Level Based Segmentation')


@st.cache
def read_data():
    df_ = pd.read_csv('customers_level_based.csv')
    return df_

st.image("segmentation.gif")

df = read_data()

age_cat = {"0_18": [i for i in range(19)],
           "19_23": [i for i in range(19, 24)],
           "24_30": [i for i in range(24, 31)],
           "31_40": [i for i in range(31, 41)],
           "41_66": [i for i in range(41, 67)],
           "67_100": [i for i in range(67, 101)]}

st.markdown("<h2 style='text-align: center; color: white;'>Level Based Segmentation</h2>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

cnt_int = col1.selectbox('Country', ['TUR', 'CAN', 'BRA', 'USA', 'GER', 'FRA'])
src_int = col2.selectbox('Operating System', ['IOS', 'ANDROID'])
sx_int = col3.selectbox('Sex', ['MALE', 'FEMALE'])
age_int = col4.number_input('Age', 1, 100)

for k, v in age_cat.items():
    if age_int in v:
        ag_cat = k

new_user = "_".join([cnt_int, src_int, sx_int, ag_cat])
user_df = df[df.customers_level_based == new_user][
    ['PRICE', 'SEGMENT']]

# Kullanıcı {segment} segmentidir ve {price} getirmesi beklenir.
try:
    if user_df.SEGMENT.values[0] == 'A':
        original_title = '<p style="text-align: center; color:#2AFF00; font-size: 22px;">' \
                         f'User belongs to {user_df.SEGMENT.values[0]} segment and expected to spend {round(user_df.PRICE.values[0], 2)} $.</p>'
        st.markdown(original_title, unsafe_allow_html=True)
    if user_df.SEGMENT.values[0] == 'B':
        original_title = '<p style="text-align: center; color:Yellow; font-size: 22px;">' \
                         f'User belongs to {user_df.SEGMENT.values[0]} segment and expected to spend {round(user_df.PRICE.values[0], 2)} $.</p>'
        st.markdown(original_title, unsafe_allow_html=True)
    if user_df.SEGMENT.values[0] == 'C':
        original_title = '<p style="text-align: center; color:#FF924F; font-size: 22px;">' \
                         f'User belongs to {user_df.SEGMENT.values[0]} segment and expected to spend {round(user_df.PRICE.values[0], 2)} $.</p>'
        st.markdown(original_title, unsafe_allow_html=True)
    if user_df.SEGMENT.values[0] == 'D':
        original_title = '<p style="text-align: center; color:#FF4949; font-size: 22px;">' \
                         f'User belongs to {user_df.SEGMENT.values[0]} segment and expected to spend {round(user_df.PRICE.values[0], 2)} $.</p>'
        st.markdown(original_title, unsafe_allow_html=True)

except IndexError:
    st.write('No data available.')
