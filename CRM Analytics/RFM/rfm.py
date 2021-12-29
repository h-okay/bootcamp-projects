# Customer Segmentation using RFM

import datetime as dt

import matplotlib.colors
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import squarify


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


pd.set_option('display.max_columns', None)

###############################  EDA AND PREP  ################################

df_ = pd.read_excel('online_retail_II.xlsx', sheet_name='Year 2010-2011')
df = df_.copy()

df.shape
df.head()
df.info()
df.describe().T

df.isnull().sum()

df.dropna(inplace=True)

df.Description.nunique()
df.Description.unique()

df.Description.value_counts().head()

df.groupby('Description').Quantity.sum().sort_values(ascending=False).head()

returns = df[df.Invoice.str.contains('C')].index
df.drop(returns, axis=0, inplace=True)
df.shape

df = df[(df['Quantity'] > 0)]
df = df[(df['Price'] > 0)]
replace_with_thresholds(df, 'Price')
replace_with_thresholds(df, 'Quantity')
df.shape

df['TotalPrice'] = df.Quantity * df.Price
df['Customer ID'] = df['Customer ID'].astype('int64')
df.head()

################################  RFM METRICS  ################################

today = dt.datetime(2011, 12, 11)
rfm = df.groupby('Customer ID').agg(
    {'InvoiceDate': lambda InvoiceDate: (today - InvoiceDate.max()).days,
     'Invoice': lambda Invoice: Invoice.nunique(),
     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

rfm.columns = ['recency', 'frequency', 'monetary']
rfm = rfm[rfm.monetary > 0]
rfm.head()

#################################  RFM SCORE  #################################

rfm['recency_score'] = pd.qcut(rfm.recency, 5, labels=[5, 4, 3, 2, 1])
rfm['frequency_score'] = pd.qcut(rfm.frequency.rank(method='first'), 5,
                                 labels=[1, 2, 3, 4, 5])
rfm['monetary_score'] = pd.qcut(rfm.monetary, 5, labels=[1, 2, 3, 4, 5])

rfm['RFM_SCORE'] = rfm.recency_score.astype(str) + rfm.frequency_score.astype(
    str)
rfm.head()

################################# SEGMENTATION ################################

seg_map = {r'[1-2][1-2]': 'hibernating',
           r'[1-2][3-4]': 'at_Risk',
           r'[1-2]5': 'cant_loose',
           r'3[1-2]': 'about_to_sleep',
           r'33': 'need_attention',
           r'[3-4][4-5]': 'loyal_customers',
           r'41': 'promising',
           r'51': 'new',
           r'[4-5][2-3]': 'potential_loyalists',
           r'5[4-5]': 'champions'}

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
rfm.reset_index(inplace=True)
rfm.head()

################################## DISCUSSION #################################

get_segment = lambda segment_: rfm[rfm.segment == segment_]

champ = get_segment('champions')
loyal = get_segment('loyal_customers')
potential = get_segment('potential_loyalists')

champ[['recency', 'frequency', 'monetary']].describe().T
# r = 6 , f = 12, m = 6498
# Customers with high frequency and high recency.

# 1. Champions (55, 54)
# ---------------------
# ▪ Organize loyalty programs.
# ▪ Advertise Limited Edition products.
# ▪ Provide special discounts.
# ▪ Tune recommendation systems.
# ▪ Pay attetion feedbacks of this group

##############################################################################

loyal[['recency', 'frequency', 'monetary']].describe().T
# r = 33 , f = 6, m = 2752
# Customers with high frequency and decent recency.

# 2. Loyals (34, 35, 44, 45)
# --------------------------
# ▪ Organize loyalty programs.
# ▪ Provide special discounts and free shipments etc.
# ▪ Tune recommendation systems.

##############################################################################

potential[['recency', 'frequency', 'monetary']].describe().T
# r = 17 , f = 2, m = 674
# Customers with high recency and have potential to shop frequently.

# 3. Potential Loyals (42, 43, 52, 53)
# -------------------
# ▪ Cross-selling ya da up-selling.
# Try to invole them into Membership or Loyalty programs.

############################ EXPORT LOYAL CUSTOMERS ###########################

loyals = rfm[rfm.segment == 'loyal_customers']
loyals.to_excel('loyals.xlsx')

############################ 2D VISUALIZATION #################################

sq1 = rfm.groupby('segment')['Customer ID'].nunique().sort_values(
    ascending=False).reset_index()
cmap = plt.cm.coolwarm
mini = min(sq1['Customer ID'])
maxi = max(sq1['Customer ID'])
norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
colors = [cmap(norm(value)) for value in sq1['Customer ID']]
fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(14, 10)
squarify.plot(sizes=sq1['Customer ID'],
              label=sq1.segment,
              alpha=1,
              color=colors)
plt.axis('off')
plt.show()

################################ 3D VISUALIZATION #############################


data = rfm[['Customer ID', 'recency', 'frequency', 'monetary', 'segment']]
fig = px.scatter_3d(data, x='recency', y='frequency', z='monetary',
                    hover_data=['Customer ID'], color='segment', opacity=0.5)
fig.update_layout(scene_zaxis_type="log")
fig.show()
