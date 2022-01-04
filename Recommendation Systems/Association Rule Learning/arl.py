# IMPORTS
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from helpers.helpers import replace_with_thresholds


# FUNCTIONS
def check_id(dataframe, stock_code):
    product_name = \
        dataframe[dataframe["StockCode"] == stock_code][["Description"]]. \
            values[0].tolist()
    return product_name


def recommend(item_id):
    return [list(sorted_rules.iloc[i]["consequents"])[0]
            for i, product in enumerate(sorted_rules.antecedents)
            for j in product if j == item_id]


def all_recos(id_: int):
    recommendations = set(recommend(id_))
    return [check_id(df_ger, id)[0] for id in recommendations]


# SETTING UP PANDAS
pd.set_option('display.max_columns', None)

# DATA IMPORT AND PREPROCESSING
# -----------------------------------------------------------------------------

df_ = pd.read_excel('online_retail_II.xlsx', sheet_name='Year 2010-2011')
df = df_.copy()

df.dropna(inplace=True)
returns = df[df.Invoice.str.contains('C', na=False)].index
df.drop(returns, axis=0, inplace=True)
df = df[(df['Quantity'] > 0)]
df = df[(df['Price'] > 0)]
replace_with_thresholds(df, 'Price')
replace_with_thresholds(df, 'Quantity')

df.head()
df.describe().T

###############################################################################

df_ger = df[df.Country == 'Germany']

df_ger = df_ger[df_ger.StockCode != 'POST']
# NOT: POST stock kodu postalanmış ürünleri temsil ediyor.
# (Tüm siparişlerin %81'i --> ONLINE RETAIL olduğu için gayet mantıklı)
# Yani faturada POST ifadesi var ise bu faturadaki ürünler posta yoluyla
# gönderilmiş. POST'lar ürün temsil etmiyor. Sadece StockCode üzerinden birliktelik
# kuralı oluşturulduğunda bakılan iki ürünün support değeri çok düşük iken POST
# bu ürünü içerdiğinde yüksek çıkmasına sebep olabilir. Bu sebeple birliktelik
# kuralı oluştururken çıkarılması, ürünler arasında kuralların daha doğru
# oluşturulmasını ve bu sayede daha iyi önerilerde bulunulmasını sağlayabilir.

inv_pro_df = df_ger.groupby(['Invoice', 'StockCode']).Quantity.sum(). \
    unstack(). \
    fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0)

inv_pro_df.shape  # 449 Fatura, 1663 Ürün
# (POST çıkarılmadığında 457 Fatura, 1664 Ürün)

freq_sets = apriori(inv_pro_df, min_support=0.01, use_colnames=True)
rules = association_rules(freq_sets, metric="support", min_threshold=0.01)
sorted_rules = rules.sort_values("support", ascending=False)
sorted_rules.head()

###############################################################################

check_id(df_ger, 21987)[0]  # PACK OF 6 SKULL PAPER CUPS
check_id(df_ger, 23235)[0]  # STORAGE TIN VINTAGE LEAF
check_id(df_ger, 22747)[0]  # POPPY'S PLAYHOUSE BATHROOM

###############################################################################

for id in [21987, 23235, 22747]:
    print(f"Recommendations for product ID: {str(id)}",
          "\n".join(all_recos(id)) + '\n' + (37 * '-'), sep='\n')

################################## MANTIKLI MI ################################
# 1. Kullanıcı sepetine PACK OF 6 SKULL PAPER CUPS eklemişti yani
# KURUKAFA DESENLİ KAĞIT BARDAKLAR

# PACK OF 20 SKULL PAPER NAPKINS - KURUKAFA DESENLİ PEÇETELER ✓
# SET/6 RED SPOTTY PAPER CUPS - NOKTA DESENLİ KAĞIT BARDAKLAR ✓
# SET/6 RED SPOTTY PAPER PLATES - NOKTA DESENLİ KAĞIT TABAKLAR ✓
# PACK OF 6 SKULL PAPER PLATES - KURUKAFA DESENLİ KAĞIT TABAKLAR ✓

# -----------------------------------------------------------------------------
# 2. Kullanıcı sepetine STORAGE TIN VINTAGE LEAF eklemişti yani
# YAPRAK DESENLİ TENEKE KUTU

# SET OF 4 KNICK KNACK TINS DOILEY - DANTEL DESENLİ TENEKE KUTU SETİ ✓
# DOILEY STORAGE TIN - DÜZ TENEKE KUTU ✓
# SET OF 4 KNICK KNACK TINS LEAVES - YAPRAK DESENLİ TENEKE KUTU SETİ ✓
# SET OF 3 REGENCY CAKE TINS - TENEKE KEK PİŞİRME KALIBI ~
# SET OF TEA COFFEE SUGAR TINS PANTRY - KAHVELİK/ŞEKERLİK TENEKE KUTU SETİ ✓
# ROUND STORAGE TIN VINTAGE LEAF - YUVARLAK YAPRAK DESENLİ TENEKE KUTU ✓

# -----------------------------------------------------------------------------
# 3. Kullanıcı sepetine POPPY'S PLAYHOUSE BATHROOM eklemişti yani
# OYUNCAK BEBEK EVİ / BANYO
#
# POPPY'S PLAYHOUSE KITCHEN - OYUNCAK BEBEK EVİ / MUTFAK ✓
# POPPY'S PLAYHOUSE LIVINGROOM - OYUNCAK BEBEK EVİ / OTURMA ODASI ✓
# POPPY'S PLAYHOUSE BEDROOM - OYUNCAK BEBEK EVİ / YATAK ODASI ✓
