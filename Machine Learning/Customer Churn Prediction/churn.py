import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd
import seaborn as sns
from lazypredict.Supervised import LazyClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    plot_roc_curve,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import (
    RobustScaler,
    MinMaxScaler,
)
import itertools
from helpers import *
from scikitplot.estimators import plot_learning_curve
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from kmodes.kmodes import KModes

#################################### PREP #####################################

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

df_ = pd.read_csv("Telco-Customer-Churn.csv")
df = df_.copy()

################################## GÖREV 1 ####################################
# Adım 1 / Genel resmi inceleyiniz

df.shape
df.info()
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.drop(df[df.TotalCharges.isnull()].index, axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)
check_df(df, 5, 5)

# -----------------------------------------------------------------------------
# Adım 2 / Numerik ve kategorik değişkenleri yakalayınız.

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

# -----------------------------------------------------------------------------
# Adım 3 /  Numerik ve kategorik değişkenlerin analizini yapınız.

for col in cat_cols:
    cat_summary(df, col)

num_analysis = pd.DataFrame()
for col in num_cols:
    num_analysis = pd.concat([num_analysis, num_summary(df, col)], axis=1)
num_analysis

# -----------------------------------------------------------------------------
# Adım 4 /  Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef
# değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)

df['Churn'] = np.where(df['Churn']=='Yes',1,0)
rare_analyser(df, 'Churn', cat_cols) # Rare kategori yok.

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

summary = pd.DataFrame()
for col in num_cols:
    summary = pd.concat([summary, target_summary_with_num(df, "Churn", col)],
                        axis=1)
    summary = summary.loc[:, ~summary.columns.duplicated()]
summary

# -----------------------------------------------------------------------------
# Adım 5 / Aykırı gözlem analizi yapınız.

for col in num_cols:
    print(col + " - " + str(check_outlier(df, col)))  # False / False / False

for col in num_cols:
    sns.boxplot(y=df[col])
    plt.show()  # Boxplotlarda outlier görünmüyor.

clf = LocalOutlierFactor()
clf.fit(df[num_cols])
df_scores = clf.negative_outlier_factor_

scores = pd.DataFrame(np.sort(df_scores))
for i in [15, 20, 30, 50]:
    scores.plot(
        stacked=True,
        xlim=[0, i],
        style=".-",
        title="Elbow metoduyla LOF eşik değeri belirleme",
        xlabel="Index",
        ylabel="Eşik Değerleri",
    )
    plt.show()

threshold = np.sort(df_scores)[14]  # 14. index iyi bir eşik değeri gibi duruyor.

df[df_scores < threshold].shape  # 14 aykırı gözlem
idx = df[df_scores < threshold].index
df.drop(idx, axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)
df.shape
df.head()

need_to_check = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                 'TechSupport', 'StreamingTV', 'StreamingMovies']

def check_errors(dataframe, target, col_name):
    temp = dataframe.loc[dataframe[target] == 'No', col_name].value_counts()
    print(col_name, "\n", temp.index.values, temp.values[0],
          end="\n"+int(len(str(temp))/2)*"#"+"\n")

for col in need_to_check:
    check_errors(df, 'InternetService', col)

# Internet servisi almadığı halde, internet servisi gerektiren hizmetlerden
# faydalanan kullanıcı girdisi yok.

# -----------------------------------------------------------------------------
# Adım 6 / Eksik gözlem analizi yapınız.

df.head()
df.isnull().sum()  # 0

msno.bar(df)  # 0
plt.show()


# -----------------------------------------------------------------------------
# Adım 7 / Korelasyon analizi yapınız.

corr_matrix = df[num_cols].corr().abs()
upper_tri = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
sns.heatmap(upper_tri, annot=True, square=True)
plt.show()

sns.scatterplot(data=df, x='tenure', y='TotalCharges')
plt.show()

# Total Charges ve Tenure'un yüksek korelasyon göstermesi beklenen bir durum.
# Müşteri daha uzun süre bizimle kaldığı sürece daha çok harcama yapacaktır.

################################## GÖREV 2 ####################################
# Adım 1 / Eksik ve aykırı değerler için gerekli işlemleri yapınız.

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

df[num_cols].describe()

sns.pairplot(df[num_cols])
plt.show()

# Feature Engineering öncesi model performansı --------------------------------

df_no_fe = df.copy()

X = df_no_fe.drop("Churn", axis=1)
y = df_no_fe["Churn"]

X.drop('customerID', axis=1, inplace=True)
X = pd.get_dummies(X, drop_first=True)

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(X)

rs = RobustScaler()
X[num_cols] = rs.fit_transform(X[num_cols])
X.head()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=123)

clf = LazyClassifier(
    verbose=1, ignore_warnings=True, custom_metric=None, random_state=42)

models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models.sort_values(['F1 Score','Accuracy'], ascending=False).iloc[:5])

# Model                       Accuracy  Balanced Accuracy  ROC AUC  F1 Score
# AdaBoostClassifier           0.81197            0.72965  0.72965   0.80535
# RidgeClassifier              0.80840            0.71956  0.71956   0.80016
# RidgeClassifierCV            0.80840            0.71956  0.71956   0.80016
# LinearSVC                    0.80484            0.71969  0.71969   0.79775
# LinearDiscriminantAnalysis   0.80199            0.72626  0.72626   0.79746

from sklearn.ensemble import AdaBoostClassifier

params = {'n_estimators':[25,50,75,100,200],
          'learning_rate':[0.1,0.5,1.0,1.2,1.5],
          'algorithm':['SAMME','SAMME.R'],
          'random_state':[42]}

adaboost = AdaBoostClassifier()
cv = GridSearchCV(adaboost, params, cv=5, verbose=1, n_jobs=-1, scoring='f1').\
    fit(X_train, y_train)

final_wfe = AdaBoostClassifier(**cv.best_params_).fit(X_train, y_train)
y_pred = final_wfe.predict(X_test)
accuracy_score(y_test, y_pred) # 0.8119
confusion_matrix(y_test, y_pred) # [933,  97]
                                 # [167, 207]
print(classification_report(y_test, y_pred))
plot_roc_curve(final_wfe, X_test, y_test)
plt.show()

plot_learning_curve(final_wfe, X_test, y_test)
plt.show()

# -----------------------------------------------------------------------------
# Adım 2 / Yeni değişkenler oluşturunuz.

df.head()
df.drop('customerID', axis=1, inplace=True)
df.reset_index(drop=True, inplace=True)

df.tenure.nunique()
df.tenure.max() # 72
df.tenure.min() # 1
sns.histplot(df.tenure, kde=True)
plt.show()

df.MonthlyCharges.nunique()
df.MonthlyCharges.max() # 118.75
df.MonthlyCharges.min() # 18.25
sns.histplot(df.MonthlyCharges, kde=True)
plt.show()

df.TotalCharges.nunique()
df.TotalCharges.max() # 8684.8
df.TotalCharges.min() # 18.8
sns.histplot(df.TotalCharges, kde=True)
plt.show()

# Benzer aylık harcama, toplam harcama ve tenure gösteren müşterileri gruplandıralım.
km = df[['tenure','MonthlyCharges','TotalCharges']]

# Tenure
kmeans = KMeans()
visu = KElbowVisualizer(kmeans, k=(1,20))
visu.fit(km['tenure'].to_frame())
visu.show() # k=3

km_final1 = KMeans(n_clusters=3)
km_final1.fit(km)
km['tenure_labels'] = km_final1.labels_
km.tenure_labels.value_counts()
# 0    4146
# 2    1613
# 1    1259
df['tenure_labels'] = km['tenure_labels']
df.drop('tenure', axis=1, inplace=True)
df.head()

# Monthly Charges
kmeans = KMeans()
visu = KElbowVisualizer(kmeans, k=(1,20))
visu.fit(km['MonthlyCharges'].to_frame())
visu.show() # k=3

km_final2 = KMeans(n_clusters=3)
km_final2.fit(km)
km['monthly_labels'] = km_final2.labels_
km.monthly_labels.value_counts()
# 2    4146
# 0    1613
# 1    1259
df['monthly_labels'] = km['monthly_labels']
df.drop('MonthlyCharges', axis=1, inplace=True)
df.head()

# Total Charges
kmeans = KMeans()
visu = KElbowVisualizer(kmeans, k=(1,20))
visu.fit(km['TotalCharges'].to_frame())
visu.show() # k=3

km_final3 = KMeans(n_clusters=3)
km_final3.fit(km)
km['total_labels'] = km_final3.labels_
km.total_labels.value_counts()
# 1    4146
# 2    1613
# 0    1259
df['total_labels'] = km['total_labels']
df.drop('TotalCharges', axis=1, inplace=True)
df.head()

sns.heatmap(df.corr(), annot=True)
plt.show()

# Birbirleriyle örüntüsü yüksek olan servisleri belirlemeye çalışalım.
temp_df = df[['StreamingTV','StreamingMovies']]

def check_service_relation(dataframe, col1, col2):
    temp_df  = dataframe[[col1,col2]]
    x = len(temp_df)
    y = len(temp_df[(temp_df[col1] == 'Yes') & (temp_df[col2] == 'Yes')])
    z = len(temp_df[(temp_df[col1] == 'No') & (temp_df[col2] == 'No')])
    t = len(temp_df[temp_df[col1]!=temp_df[col2]]) # 1552
    both_yes = y / x * 100
    both_no = z / x * 100
    different = t / x * 100
    cols = f"{col1} vs. {col2}"
    print(pd.DataFrame({'same':both_yes+both_no,
                         'different':different}, index=[cols]).T)
    print((len(cols) * '-'))

combinations = list(itertools.combinations(service_cols, 2))

for duo in combinations:
    check_service_relation(df, duo[0],duo[1]) # Belirgib bir örüntü gözlemlenemedi.


# Servis değişkenleri clustering için KModes

kmod_df = df[['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup','DeviceProtection',
 'TechSupport','StreamingTV','StreamingMovies']]

cost = []
K = range(1, 10)
for num_clusters in list(K):
    kmode = KModes(n_clusters=num_clusters, init="random", n_init=5, verbose=1)
    kmode.fit_predict(kmod_df)
    cost.append(kmode.cost_)

plt.plot(K, cost, 'bx-')
plt.xlabel('No. of clusters')
plt.ylabel('Cost')
plt.title('Elbow Method For Optimal k')
plt.show() # k = 3

kmode = KModes(n_clusters=3, init = "random", n_init = 5, verbose=1)
clusters = kmode.fit_predict(kmod_df)
clusters

kmod_df.insert(0, "Cluster", clusters, True)
df['Cat_Cluster'] = kmod_df['Cluster']
df.drop(['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup','DeviceProtection',
 'TechSupport','StreamingTV','StreamingMovies'], axis=1, inplace=True)
df.head()

# Ödeme değişkenleri clustering için KModes
odeme_df = df[['Contract','PaperlessBilling','PaymentMethod']]

cost = []
K = range(1, 10)
for num_clusters in list(K):
    kmode = KModes(n_clusters=num_clusters, init="random", n_init=5, verbose=1)
    kmode.fit_predict(odeme_df)
    cost.append(kmode.cost_)

plt.plot(K, cost, 'bx-')
plt.xlabel('No. of clusters')
plt.ylabel('Cost')
plt.title('Elbow Method For Optimal k')
plt.show() # k = 3

kmode = KModes(n_clusters=2, init = "random", n_init = 5, verbose=1)
clusters = kmode.fit_predict(odeme_df)
clusters

odeme_df.insert(0, "Cluster", clusters, True)
df['Pay_Cluster'] = odeme_df['Cluster']
df.drop(['Contract','PaperlessBilling','PaymentMethod'], axis=1, inplace=True)
df.head()

# -----------------------------------------------------------------------------
# Adım 3 / Encoding işlemlerini gerçekleştiriniz.

df = pd.get_dummies(df, drop_first=True)
df.head()
df.info()

# -----------------------------------------------------------------------------
# Adım 4 / Numerik değişkenler için standartlaştırma yapınız.

pass

# -----------------------------------------------------------------------------
# Adım 5 / Model oluşturunuz.

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

clf = LazyClassifier(
    verbose=0, ignore_warnings=True, custom_metric=None, random_state=42
)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

top_models = models.sort_values(["F1 Score", "Accuracy"], ascending=False)[:5].index

print(top_models)

############################ HYPERPARAMETER TUNING ############################
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

# BernoulliNB -----------------------------------------------------------------

params = {'alpha':[0.1,0.5,1.0,1.5,2]}
berno = BernoulliNB()
cv =  GridSearchCV(berno, params, scoring='f1', cv=5, verbose=2, n_jobs=-1).\
    fit(X_train, y_train)

berno_final = BernoulliNB(**cv.best_params_).fit(X_train,y_train)
y_pred = berno_final.predict(X_test)

accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

plot_roc_curve(berno_final, X_test, y_test)
plt.show()

plot_learning_curve(berno_final, X_test, y_test)
plt.show() # ***********HOCAM BU NASIL YORUMLANABİLİR?*******************

# AdaBoostClassifier ----------------------------------------------------------

params = {'n_estimators':[25,50,75,100,200],
          'learning_rate':[0.1,0.5,1.0,1.2,1.5],
          'algorithm':['SAMME','SAMME.R'],
          'random_state':[42]}

ada = AdaBoostClassifier()
cv =  GridSearchCV(ada, params, scoring='f1', cv=5, verbose=2, n_jobs=-1).\
    fit(X_train, y_train)

ada_final = AdaBoostClassifier(**cv.best_params_).fit(X_train,y_train)
y_pred = ada_final.predict(X_test)

accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

plot_roc_curve(ada_final, X_test, y_test)
plt.show()

plot_learning_curve(ada_final, X_test, y_test)
plt.show() # ***********HOCAM BU NASIL YORUMLANABİLİR?*******************

# XGBClassifier ---------------------------------------------------------------

params = {'n_estimators':[25,50,75,100,200],
          'learning_rate':[0.1,0.5,1.0,1.2,1.5],
          'max_depth ':[1,2,3,4,5],
          'random_state':[42]}

xgb = XGBClassifier()
cv =  GridSearchCV(xgb, params, scoring='f1', cv=5, verbose=2, n_jobs=-1).\
    fit(X_train, y_train)

xgb_final = XGBClassifier(**cv.best_params_).fit(X_train,y_train)
y_pred = xgb_final.predict(X_test)

accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

plot_roc_curve(xgb_final, X_test, y_test)
plt.show()

plot_learning_curve(xgb_final, X_test, y_test)
plt.show()

################################### FINAL #####################################

from sklearn.ensemble import VotingClassifier

final_model = VotingClassifier(
    estimators=[("berno", berno_final), ("ada", ada_final), ("xgb",xgb_final)],
    voting="soft").fit(X_train, y_train)

y_pred = final_model.predict(X_test)

accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

plot_roc_curve(final_model, X_test, y_test)
plt.show()

plot_learning_curve(final_model, X_test, y_test)
plt.show()