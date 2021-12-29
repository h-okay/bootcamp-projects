# Customer Segmentation using BGNBD & GG CLTV


import datetime as dt

import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def create_rfm(dataframe):
    # VERIYI HAZIRLAMA
    dataframe.dropna(inplace=True)
    returns = dataframe[
        dataframe.Invoice.astype(str).str.contains('C', na=False)].index
    dataframe.drop(returns, axis=0, inplace=True)
    dataframe = dataframe[(dataframe['Quantity'] > 0)]
    dataframe = dataframe[(dataframe['Price'] > 0)]
    replace_with_thresholds(dataframe, 'Price')
    replace_with_thresholds(dataframe, 'Quantity')
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]

    # RFM METRIKLERININ HESAPLANMASI
    today_date = dt.datetime(2011, 12, 11)
    rfm = dataframe.groupby('Customer ID').agg(
        {'InvoiceDate': lambda date: (today_date - date.max()).days,
         'Invoice': lambda num: num.nunique(),
         "TotalPrice": lambda price: price.sum()})
    rfm.columns = ['recency', 'frequency', "monetary"]
    rfm = rfm[(rfm['monetary'] > 0)]

    # RFM SKORLARININ HESAPLANMASI
    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5,
                                     labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

    rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                        rfm['frequency_score'].astype(str))

    # SEGMENTLERIN ISIMLENDIRILMESI
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
    rfm = rfm[["recency", "frequency", "monetary", "segment"]]
    return rfm


def add_clv(final_df, df_, month: int):
    cltv = ggf.customer_lifetime_value(bgf,
                                       df_['frequency'],
                                       df_['recency'],
                                       df_['T'],
                                       df_['monetary'],
                                       time=month,
                                       freq="W",
                                       discount_rate=0.01)

    cltv = cltv.reset_index()
    cltv.rename(columns={'clv': f'{month}m_clv'}, inplace=True)
    final = final_df.merge(cltv, on='Customer ID', how='left')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(final[[f"{month}m_clv"]])
    final[f"{month}m_scaled_clv"] = scaler.transform(final[[f"{month}m_clv"]])
    return final


########################### DATA IMPORT AND PREP ##############################

df_ = pd.read_excel('online_retail_II.xlsx', sheet_name='Year 2010-2011')
df = df_.copy()

df.dropna(inplace=True)

returns = df[df.Invoice.str.contains('C', na=False)].index
df.drop(returns, axis=0, inplace=True)

df = df[df['Country'] == 'United Kingdom']
df = df[(df['Quantity'] > 0)]
df = df[(df['Price'] > 0)]

replace_with_thresholds(df, 'Price')
replace_with_thresholds(df, 'Quantity')

df["TotalPrice"] = df["Quantity"] * df["Price"]

############################## CREATING FEATURES ##############################

today_date = dt.datetime(2011, 12, 11)

cltv_df = df.groupby('Customer ID').agg(
    {'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                     lambda date: (today_date - date.min()).days],
     'Invoice': lambda num: num.nunique(),
     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7  # weekly

################################## MODELLING ##################################

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # MONTHS
                                   freq="W",
                                   discount_rate=0.01)

cltv = cltv.reset_index()
cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(10)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_final[["clv"]])
cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])

################################ DISCUSSION ###################################

rfm = create_rfm(df)  # RFM segments for cross validation
rfm.rename(columns={'segment': 'rfm_segment'}, inplace=True)
final = rfm[['rfm_segment']].merge(cltv_final, on='Customer ID')
final.rename(columns={'scaled_clv': '6m_scaled_clv', 'clv': '6m_clv'},
             inplace=True)

top20customers = int(len(final) * 0.2)
final.sort_values('6m_scaled_clv', ascending=False).head(top20customers)[
    ['Customer ID', '6m_scaled_clv', 'rfm_segment']]
final.sort_values('6m_scaled_clv', ascending=False).head(top20customers)[
    'rfm_segment'].value_counts()
# champions              256
# loyal_customers        154
# potential_loyalists     62
# need_attention          21
# about_to_sleep           9
# at_risk                  8
# cant_loose               4
# Customer with most CLTV mostly belongs to champion, loyal customers and
# potential loyalists RFM segments.

# CHAMPIONS
final[final.rfm_segment == 'champions']['6m_clv'].describe().round(2)
# Descriptive CLV
# count      577.00
# mean      2980.52
# std       5723.98
# min        109.61
# max      85648.50

top_20perc1 = round(577 * 0.2)
top_champ = final[final.rfm_segment == 'champions'].sort_values(
    ['6m_scaled_clv', '6m_clv'], ascending=False).iloc[:top_20perc1][
    'Customer ID']
list(top_champ)  # Top %20 of Champions RFM Segment with highest CLTV

# -----------------------------------------------------------------------------

# LOYALISTS
final[final.rfm_segment == 'loyal_customers']['6m_clv'].describe().round(2)
# Descriptive CLV
# count      730.00
# mean      1495.96
# std       1449.31
# min         24.29
# max      17816.21

top_20perc2 = round(730 * 0.2)
top_loyal = final[final.rfm_segment == 'loyal_customers'].sort_values(
    ['6m_scaled_clv', '6m_clv'], ascending=False).iloc[:top_20perc2][
    'Customer ID']
list(top_loyal)  # Top %20 of Loyal Customers RFM Segment with highest CLTV

# -----------------------------------------------------------------------------

# POTENTIAL LOYALISTS
final[final.rfm_segment == 'potential_loyalists']['6m_clv'].describe().round(2)
# Descriptive CLV
# count      352.00
# mean      1301.44
# std       1522.02
# min         47.12
# max      21107.94

top_20perc3 = round(352 * 0.2)
top_poten = final[final.rfm_segment == 'potential_loyalists'].sort_values(
    ['6m_scaled_clv', '6m_clv'], ascending=False).iloc[:top_20perc3][
    'Customer ID']
list(top_poten)  # Top %20 of Potential Loyalists RFM Segment with highest CLTV

################## CLTV FOR DIFFERENT TIME PERIODS(1,12 Months) ###############

final = add_clv(final, cltv_df, 1)
final = add_clv(final, cltv_df, 12)

best_1m = final.sort_values('1m_scaled_clv', ascending=False)[
    ['Customer ID', '1m_clv', '1m_scaled_clv']].head(10)
best_12m = final.sort_values('12m_scaled_clv', ascending=False)[
    ['Customer ID', '12m_clv', '12m_scaled_clv']].head(10)

list(best_1m) == list(best_12m)  # False

# Customer ID         1m_clv    1m_scaled_clv        #  Customer ID        12m_clv  12m_scaled_clv
# 18102         14884.500498         1.000000        #  18102        163586.717976        1.000000
# 14096          9855.142320         0.662108        #  14096        104893.741912        0.641212
# 17450          8434.507606         0.566664        #  17450         92691.902872        0.566622
# 17511          6394.139434         0.429584        #  17511         70283.954918        0.429643
# 16684          4360.925582         0.292984        #  16684         47889.189122        0.292745
# 14088          4355.369061         0.292611        #  13694         47870.662086        0.292632
# 13694          4354.334840         0.292542        #  14088         47687.833143        0.291514
# 15311          4098.736670         0.275369        #  15311         45066.570490        0.275490
# 13089          3983.921766         0.267656        #  13089         43794.263478        0.267713
# 16000          3843.408761         0.258216        #  15061         40347.775629        0.246645

final[final['Customer ID'] == 15601].unstack().unstack()
final[final['Customer ID'] == 16000].unstack().unstack()

# Customer ID            15601           # Customer ID                   16000
# rfm_segment       champions            # rfm_segment     potential_loyalists
# recency           51.571429            # recency                         0.0
# T                 53.285714            # T                          0.428571
# frequency                14            # frequency                         3
# monetary             484.04            # monetary                2055.786667
# 1m_clv           518.424707            # 1m_clv                  3843.408761
# 1m_scaled_clv       0.03483            # 1m_scaled_clv              0.258216
# 12m_clv         5695.407853            # 12m_clv                39233.195047
# 12m_scaled_clv     0.034816            # 12m_scaled_clv             0.239831

# ▪ CLTV favors high monetary over high frequency which results higher score in
# short terms for customer with high monetary.
# ▪ In the long run monetary's dominance over frequency gets lower.
# This can be seen on users 16000 and 15601.

############################### CLTV SEGMENTATION #############################

final["cltv_segment"] = pd.qcut(final["6m_scaled_clv"], 4,
                                labels=["D", "C", "B", "A"])

final[['Customer ID', '6m_clv', '6m_scaled_clv', 'cltv_segment',
       'rfm_segment']].sort_values('6m_scaled_clv',
                                   ascending=False).value_counts(
    ['cltv_segment', 'rfm_segment'])

# top20customers --> 514
# rfm_segment
# champions              295
# loyal_customers        202
# potential_loyalists     80
# need_attention          28
# at_risk                 16
# about_to_sleep          13
# cant_loose               9

# A SEGMENT
final[final.cltv_segment == 'A'][
    ['Customer ID', '6m_clv', '6m_scaled_clv', 'cltv_segment',
     'rfm_segment']].sort_values('6m_scaled_clv',
                                 ascending=False).value_counts(
    ['cltv_segment', 'rfm_segment'])

# cltv_segment  rfm_segment
# A             champions              295
#               loyal_customers        202
#               potential_loyalists     80
#               need_attention          28
#               at_risk                 16
#               about_to_sleep          13
#               cant_loose               9

# Segment A customers also mostly in Champions and Loyal Customers in
# RFM Segmentation.

# ▪ Organize loyalty programs.
# ▪ Advertise Limited Edition products.
# ▪ Provide special discounts.
# ▪ Tune recommendation systems.
# ▪ Pay attetion feedbacks of this group
# ▪ Provide special discounts and free shipments etc.

# B SEGMENT
final[final.cltv_segment == 'B'][
    ['Customer ID', '6m_clv', '6m_scaled_clv', 'cltv_segment',
     'rfm_segment']].sort_values('6m_scaled_clv',
                                 ascending=False).value_counts(
    ['cltv_segment', 'rfm_segment'])

# cltv_segment  rfm_segment
# B             loyal_customers        245
#               champions              161
#               potential_loyalists     90
#               at_risk                 74
#               need_attention          37
#               about_to_sleep          14
#               cant_loose              13
#               hibernating              8

# Segment B customers also mostly in Loyal Customers and Champions in
# RFM Segmentation.

# ▪ Organize loyalty programs.
# ▪ Advertise Limited Edition products.
# ▪ Provide special discounts.
# ▪ Tune recommendation systems.
# ▪ Pay attetion feedbacks of this group
# ▪ Provide special discounts and free shipments etc.
# ▪ Turn loyalist customers into "brand advocates". Because these users are
# already extremely loyal to the brand and products, they are open to attracting
# new users with Word-of-Mouth and defending the products more than the company.

########################## EXPECTED PURCHASE AND SALES ########################

to_db = final[
    ['Customer ID', 'recency', 'T', 'frequency', 'monetary', '6m_clv',
     '6m_scaled_clv', 'cltv_segment']]

to_db["expected_purc_1_week"] = bgf.predict(1,
                                            to_db['frequency'],
                                            to_db['recency'],
                                            to_db['T'])

to_db["expected_purc_1_month"] = bgf.predict(4,
                                             to_db['frequency'],
                                             to_db['recency'],
                                             to_db['T'])

to_db["expected_average_profit"] = ggf.conditional_expected_average_profit(
    to_db['frequency'],
    to_db['monetary'])

to_db = to_db.reindex(
    columns=['Customer ID', 'recency', 'frequency', 'monetary',
             'expected_purc_1_week',
             'expected_purc_1_month', 'expected_average_profit', '6m_clv',
             '6m_scaled_clv', 'cltv_segment'])

to_db.rename(columns={'6m_clv': 'clv', 'cltv_segment': 'segment',
                      '6m_scaled_clv': 'scaled_clv'}, inplace=True)

# Customer ID   recency     frequency   monetary    expected_purc_1_week
# 12747.0       52.285714   11          381.455455  0.202475
# 12748.0       53.142857   209         154.564163  3.237513
# 12749.0       29.857143   5           814.488000  0.167144
# 12820.0       46.142857   4           235.585000  0.103970
# 12822.0        2.285714   2           474.440000  0.129143

# expected_purc_1_month  expected_average_profit   clv              scaled_clv
# 0.807661               387.822977                1937.009093      0.022616
# 12.915924              154.708635                12366.071920     0.144382
# 0.665709               844.095338                3445.922970      0.040233
# 0.414561               247.081182                631.934113       0.007378
# 0.512661               520.829235                1612.132952      0.018823

# segment
# A
# A
# A
# C
# B
