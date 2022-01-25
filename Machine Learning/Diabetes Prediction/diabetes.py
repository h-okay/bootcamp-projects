import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import missingno as msno
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

from helpers import *
from scikitplot.estimators import plot_learning_curve
#################################### PREP #####################################

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
df_ = pd.read_csv("diabetes.csv")
df = df_.copy()

################################## GÖREV 1 ####################################
# Adım 1 / Genel resmi inceleyiniz

df.shape
df.info()

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
num_analysis.T

# -----------------------------------------------------------------------------
# Adım 4 /  Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef
# değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)

summary = pd.DataFrame()
for col in num_cols:
    summary = pd.concat([summary, target_summary_with_num(df, "Outcome", col)],
                        axis=1)
    summary = summary.loc[:, ~summary.columns.duplicated()]
summary.T

# -----------------------------------------------------------------------------
# Adım 5 / Aykırı gözlem analizi yapınız.

for col in df.columns:
    print(col + " - " + str(check_outlier(df, col)))  # Insulinde outlier
    # gözlemleniyor. Değişkenlerin birbirleriyle olan bağlarından kaynaklı
    # durumları kapsamıyor.

replace_with_thresholds(df, 'Insulin')

for col in num_cols:
    sns.boxplot(y=df[col])
    plt.show()  # Boxplotlarda outlier görünüyor.

# sns.pairplot(data=df, hue="Outcome", palette="viridis")
# plt.show()  # Burada da bazı outlierlar gözlemleniyor.

# Tek başına outlier olarak tanımlanmadığı halde birbirleriyle ilişkileri
# sonucunda outlier özelliği taşıyan gözlem birimlerini çıkarmak için LOF
# kullanalım.
clf = LocalOutlierFactor()
clf.fit(df)
df_scores = clf.negative_outlier_factor_

# Eşik değeri belirleme
scores = pd.DataFrame(np.sort(df_scores))
for i in [5, 10, 15, 20, 30, 50]:
    scores.plot(
        stacked=True,
        xlim=[0, i],
        style=".-",
        title="Elbow metoduyla LOF eşik değeri belirleme",
        xlabel="Index",
        ylabel="Eşik Değerleri",
    )
    plt.show()

threshold = np.sort(df_scores)[10]  # 10.index iyi bir eşik değeri gibi duruyor.

df[df_scores < threshold].shape  # 10 aykırı gözlem
idx = df[df_scores < threshold].index
df.drop(idx, axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)
df.shape

# -----------------------------------------------------------------------------
# Adım 6 / Eksik gözlem analizi yapınız.

df.isnull().sum()  # 0

msno.bar(df)  # 0
plt.show()

# -----------------------------------------------------------------------------
# Adım 7 / Korelasyon analizi yapınız.

corr_matrix = df.corr().abs()
upper_tri = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
sns.heatmap(upper_tri, annot=True, square=True)
plt.show()

# 90 üstü korelasyon olanları çıkaralım. Aynı bilgiyi taşıyan 2 sütuna gerek yok.
drop_list = [col for col in upper_tri.columns if any(upper_tri[col] > 0.90)]
len(drop_list)  # 0

################################## GÖREV 2 ####################################
# Adım 1 / Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde
# eksik gözlem bulunmamakta ama Glikoz, Insulin vb. değişkenlerde 0 değeri içeren
# gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz
# veya insulin değeri 0 olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini
# ilgili değerlerde NaN olarak atama yapıp sonrasında eksik değerlere işlemleri
# uygulayabilirsiniz.

have_zeros = [print(col) for col in num_cols if 0 in df[col].values]


# Pregnancies -- 0 olabilir.
# BloodPressure -- 0 olamaz.
# SkinThickness -- 0 olamaz.
# Insulin -- 0 olamaz.
# BMI -- 0 olamaz.


def add_nans(dataframe, col_name):
    dataframe[col_name] = dataframe[col_name].apply(
        lambda x: np.nan if x == 0 else x)
    count = dataframe[col_name].apply(
        lambda x: np.nan if x == 0 else x).isnull().sum()
    return f"{count} errors converted to NaN."


add_nans(df, "BloodPressure")  # '35 errors converted to NaN.'
add_nans(df, "SkinThickness")  # '226 errors converted to NaN.'
add_nans(df, "Insulin")  # '368 errors converted to NaN.'
add_nans(df, "BMI")  # '11 errors converted to NaN.'

missing_values_table(df)

msno.bar(df)
plt.show()

# Insulin seviyesinin şeker hastalığı üzerindeki etkisi düşünüldüğünde neredeyse
# değerlerin yarısının boş olması sorun teşkil ediyor. Bu değerleri olduğu gibi
# düşüremeyiz.

scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
dff.head()

imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)

df["Impute_Age"] = dff["Age"]
df["Impute_Insulin"] = dff["Insulin"]
df["Impute_SkinThickness"] = dff["SkinThickness"]
df["Impute_BMI"] = dff["BMI"]


df.drop(["Age", "Insulin", "SkinThickness", "BMI"], axis=1, inplace=True)
df.reset_index(drop=True, inplace=True)
df = df.rename(
    columns={
        "Impute_Age": "Age",
        "Impute_Insulin": "Insulin",
        "Impute_SkinThickness": "SkinThickness",
        "Impute_BMI": "BMI",
    }
)

missing_values_table(df)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
df.info()

df.BloodPressure = df.BloodPressure.astype("int64")
df.SkinThickness = df.SkinThickness.astype("int64")
df.Age = df.Age.astype("int64")

# Feature Engineering öncesi model performansı --------------------------------

df_no_fe = df.copy()
cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df_no_fe)

rs = RobustScaler()
df_no_fe[num_cols] = rs.fit_transform(df_no_fe[num_cols])

X = df_no_fe.drop("Outcome", axis=1)
y = df_no_fe["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=123)

clf = LazyClassifier(
    verbose=0, ignore_warnings=True, custom_metric=None, random_state=42)

models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models.sort_values(["F1 Score", "Accuracy"], ascending=False).iloc[:5])

# Model                   Accuracy  Balanced Accuracy  ROC AUC  F1 Score
# KNeighborsClassifier       0.759              0.715    0.715     0.752
# BaggingClassifier          0.759              0.711    0.711     0.751
# ExtraTreesClassifier       0.752              0.714    0.714     0.748
# SVC                        0.752              0.687    0.687     0.736
# RandomForestClassifier     0.738              0.704    0.704     0.735

# -----------------------------------------------------------------------------
# Adım 2 / Yeni değişkenler oluşturunuz.

# Yapılan araştırmalarda deri kalınlığının insulin seviyesiyle bağlantılı olduğu
# belirlenmiş. Özellikle kadın hastalarda şeker hastalığının yeni tahminleyicisi
# olarak bahsediliyor. İkisininde bilgisini taşıyan bir değişken oluşturalım.

df[["Insulin", "SkinThickness","Outcome"]].corr()
100 -( 0.260 / 0.408  * 100) # Insulin ile Outcome ilişkisi SkinThickness'a göre
# yuvarlak olarak %40 daha yüksek.
scaler = MinMaxScaler()
temp_df = pd.DataFrame(
    scaler.fit_transform(df[["Insulin", "SkinThickness"]]),
    columns=["Insulin", "SkinThickness"])
temp_df["Combined"] = (temp_df["Insulin"] * 60/100) + (temp_df["SkinThickness"] * 40/100)
df = pd.concat([df, temp_df["Combined"]], axis=1)
df.head()

# Age Kategorileri ------------------------------------------------------------

print(f"Max Age: {df.Age.min()}\nMin Age: {df.Age.max()}")

sns.histplot(df["Age"])
plt.show()

df["AgeGroup"] = pd.cut(
    df["Age"], bins=[20, 35, 50, 65, 100],
    labels=["Young", "Early", "Late", "Old"]
)
df.AgeGroup.value_counts()
df.head()

# Pregnancy Binary ------------------------------------------------------------

print(
    f"Max Pregnancy: {df.Pregnancies.min()}\nMin Pregnancy: {df.Pregnancies.max()}")

sns.histplot(df["Pregnancies"])
plt.show()

df["Pregnancy"] = np.where(df.Pregnancies > 0, 1, 0)
df.Pregnancy.value_counts()
df.head()

# BMI Groups ------------------------------------------------------------------

# BMI Categories:
# Underweight = <18.5
# Normal weight = 18.5–24.9
# Overweight = 25–29.9
# Obesity = BMI of 30 or greater

df["BMI_Cat"] = pd.cut(
    df["BMI"],
    bins=[0, 18, 25, 30, 100],
    labels=["Underweight", "Normal", "Overweight", "Obesity"])
df.BMI_Cat.value_counts()
df.head()

# -----------------------------------------------------------------------------
# Adım 3 / Encoding işlemlerini gerçekleştiriniz.

df.drop(["Pregnancies", "Age", "Insulin", "SkinThickness", "BMI"], axis=1,
        inplace=True)

df = pd.get_dummies(df, drop_first=True)
df.head()
df.info()

# -----------------------------------------------------------------------------
# Adım 4 / Numerik değişkenler için standartlaştırma yapınız.

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])
df.head()

# -----------------------------------------------------------------------------
# Adım 5 / Model oluşturunuz.

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=123
)

# Base Models -----------------------------------------------------------------

clf = LazyClassifier(
    verbose=0, ignore_warnings=True, custom_metric=None, random_state=42
)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

top_models = models.sort_values(["F1 Score", "Accuracy"], ascending=False)[:3].index

print(top_models)
#
# Model               Accuracy  Balanced Accuracy  ROC AUC  F1 Score
# AdaBoostClassifier     0.752              0.701    0.701     0.743
# BernoulliNB            0.724              0.707    0.707     0.726
# NuSVC                  0.738              0.677    0.677     0.724

############################ HYPERPARAMETER TUNING ############################

from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import NuSVC

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

accuracy_score(y_test, y_pred) # 0.758
confusion_matrix(y_test, y_pred)#[86,  8]
                                #[27, 24]
print(classification_report(y_test, y_pred))
#               precision    recall  f1-score   support
#            0       0.76      0.91      0.83        94
#            1       0.75      0.47      0.58        51
#     accuracy                           0.76       145
#    macro avg       0.76      0.69      0.70       145
# weighted avg       0.76      0.76      0.74       145

plot_roc_curve(ada_final, X_test, y_test)
plt.show()

plot_learning_curve(ada_final, X_test, y_test)
plt.show()

# BernoulliNB -----------------------------------------------------------------

params = {'alpha':[0.1,0.5,1.0,1.5,2]}
berno = BernoulliNB()
cv =  GridSearchCV(berno, params, scoring='f1', cv=5, verbose=2, n_jobs=-1).\
    fit(X_train, y_train)

berno_final = BernoulliNB(**cv.best_params_).fit(X_train,y_train)
y_pred = berno_final.predict(X_test)

accuracy_score(y_test, y_pred) # 0.717
confusion_matrix(y_test, y_pred) #[69, 25]
                                 #[16, 35]
print(classification_report(y_test, y_pred))
#               precision    recall  f1-score   support
#            0       0.81      0.73      0.77        94
#            1       0.58      0.69      0.63        51
#     accuracy                           0.72       145
#    macro avg       0.70      0.71      0.70       145
# weighted avg       0.73      0.72      0.72       145
plot_roc_curve(berno_final, X_test, y_test)
plt.show()

plot_learning_curve(berno_final, X_test, y_test)
plt.show()

# NuSVC ----------------------------------------------------------------------

param_grid = {"nu": [0.1,0.3,0.5,0.7,1],
              "kernel":['linear','poly','rbf','sigmoid'],
              "degree":[1,2,3,4,5]}

nu_svc = NuSVC()
cv = GridSearchCV(nu_svc, param_grid, cv=5, scoring="f1", verbose=2, n_jobs=-1).\
    fit(X_train,y_train)

final_nu = NuSVC(**cv.best_params_, probability=True).fit(X_train, y_train)
y_pred = final_nu.predict(X_test)

accuracy_score(y_test, y_pred) # 0.724
confusion_matrix(y_test, y_pred)#[82, 12]
                                #[28, 23]
print(classification_report(y_test, y_pred))
#               precision    recall  f1-score   support
#            0       0.75      0.87      0.80        94
#            1       0.66      0.45      0.53        51
#     accuracy                           0.72       145
#    macro avg       0.70      0.66      0.67       145
# weighted avg       0.71      0.72      0.71       145
plot_roc_curve(final_nu, X_test, y_test)
plt.show()

plot_learning_curve(final_nu, X_test, y_test)
plt.show() # overfitting?

################################### FINAL #####################################

from sklearn.ensemble import VotingClassifier

final_model = VotingClassifier(
    estimators=[("ada", ada_final), ("berno", berno_final), ("nu", final_nu)], voting="soft"
).fit(X_train, y_train)

y_pred = final_model.predict(X_test)

accuracy_score(y_test, y_pred)  # 0.744
confusion_matrix(y_test, y_pred)#[77, 17]
                                #[20, 31]
print(classification_report(y_test, y_pred))
#               precision    recall  f1-score   support
#            0       0.79      0.82      0.81        94
#            1       0.65      0.61      0.63        51
#     accuracy                           0.74       145
#    macro avg       0.72      0.71      0.72       145
# weighted avg       0.74      0.74      0.74       145

plot_roc_curve(final_model, X_test, y_test)
plt.show()

plot_learning_curve(final_model, X_test, y_test)
plt.show() # overfitting?

