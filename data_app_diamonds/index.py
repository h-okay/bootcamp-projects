import joblib
import numpy as np
import pandas as pd
import streamlit as st


@st.cache
def get_data():
    path = r"diamonds.csv"
    return pd.read_csv(path)


@st.cache
def convert_df(df_):
    return df_.to_csv().encode('utf-8')


# Model import
model = joblib.load("data_preserved_model.pkl")

# Sample data
df = get_data()
df = df.dropna()

st.header('Diamonds 💎')
st.write("""Use filter below to filter the data. You can use side menu to get 
an estimated value.""")

# Multiselect filtering
cut_options = df["cut"].unique()
color_options = df["color"].unique()
clarity_options = df["clarity"].unique()

cut_select = st.multiselect("Cut Quality", cut_options, default=cut_options)
color_select = st.multiselect("Color", color_options, default=color_options)
clarity_select = st.multiselect(
    "Clarity", clarity_options, default=clarity_options)

after_filter = df[(df["cut"].isin(cut_select)) & (
    df["color"].isin(color_select)) & (df["clarity"].isin(clarity_select))]

# Show dataframe
st.dataframe(after_filter)
st.write(str(len(after_filter)) + " rows, " +
         str(len(after_filter.columns)) + " columns")

# Download filtered data as csv
csv = convert_df(after_filter)
st.download_button(
    label="Download Filtered Data (CSV)",
    data=csv,
    file_name='diamonds_filtered.csv',
    mime='text/csv')

# Placeholder for model answer
placeholder = st.sidebar.empty()
# ML model feature sliders
carat_opt = st.sidebar.select_slider(
    "Carat", options=np.round(np.linspace(0, 5, 100), 1), value=2.5)
cut_opt = st.sidebar.select_slider(
    "Cut", options=["Fair", "Good", "Very Good", "Premium", "Ideal"],
    value="Very Good")
color_opt = st.sidebar.select_slider(
    "Color", options=["J", "I", "H", "G", "F", "E", "D"], value="G")
clarity_opt = st.sidebar.select_slider("Clarity", options=[
    "I1", "SI2", "VS2", "VS1", "VVS2", "VVS1", "IF"], value="VS1")
depth_opt = st.sidebar.select_slider(
    "Depth", np.round(np.linspace(40, 80, 101), 1), value=60)
table_opt = st.sidebar.select_slider(
    "Table", options=np.round(np.linspace(40, 100, 101), 1), value=70.0)
x_opt = st.sidebar.select_slider(
    "x", options=np.round(np.linspace(3, 10, 100), 1), value=6.5)
y_opt = st.sidebar.select_slider(
    "y", options=np.round(np.linspace(3, 60, 101), 1), value=31.5)
z_opt = st.sidebar.select_slider(
    "z", options=np.round(np.linspace(1, 35, 101), 1), value=18)

u_dict = {"carat": carat_opt, "cut": cut_opt, "color": color_opt,
          "clarity": clarity_opt,
          "depth": depth_opt, "table": table_opt, "x": x_opt, "y": y_opt,
          "z": z_opt}

user_data = pd.DataFrame(index=np.arange(1), data=u_dict)

# Encode
user_data["cut"] = user_data["cut"].map(
    {"Fair": 0, "Good": 1, "Ideal": 2, "Premium": 3, "Very Good": 4})
user_data["color"] = user_data["color"].map(
    {"D": 0, "E": 1, "F": 2, "G": 3, "H": 4, "I": 5, "J": 6})
user_data["clarity"] = user_data["clarity"].map(
    {"I1": 0, "IF": 1, "SI1": 2, "SI2": 3, "VS1": 4, "VS2": 5, "VVS1": 6,
     "VVS2": 7})

# Display model.predict
placeholder.header(f"Estimated value:  ${int(model.predict(user_data)[0])}")
