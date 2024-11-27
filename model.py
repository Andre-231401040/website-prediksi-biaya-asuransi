import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("medical_insurance.csv")

# menghapus outliers berdasarkan kolom charges
df = df[np.abs(stats.zscore(df["charges"])) <= 3]

# membuat objek LabelEncoder
le = LabelEncoder() 
# membuat representasi angka dari setiap nilai kolom sex
label_sex = le.fit_transform(df["sex"])
# membuat representasi angka dari setiap nilai kolom region
label_region = le.fit_transform(df["region"])
# membuat representasi angka dari setiap nilai kolom smoker
label_smoker = le.fit_transform(df["smoker"])

# menyimpan semua kolom dari df ke data_model kecuali kolom sex, region, dan smoker
data_model = df.drop(columns=["sex", "region", "smoker"])
# membuat kolom sex pada data_model dengan nilai sama dengan label_sex
data_model["sex"] = label_sex
# membuat kolom region pada data_model dengan nilai sama dengan label_region
data_model["region"] = label_region
#membuat kolom smoker pada data_model dengan nilai sama dengan label_smoker
data_model["smoker"] = label_smoker

# variabel X menyimpan semua kolom dari data_model kecuali kolom charges
X = data_model.drop(columns=["charges"])
# variabel y hanya berisi kolom charges dari data_model
y = data_model["charges"]

# membagi data menjadi training set dan testing set dengan 20% dari data akan digunakan sebagai testing set dan 80% dari data akan digunakan sebagai training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7) # random_state=7 berfungsi untuk memastikan bahwa pembagian data akan konsisten setiap kali kode dijalankan

# membuat model Random Forest Regression
model = RFR()
# melatih model yang telah dibuat menggunakan training set yang disimpan pada variabel X_train dan y_train
model = model.fit(X_train, y_train)

# menyimpan model yang telah dilatih ke dalam file model.pkl dengan format yang dapat diserialisasi
pickle.dump(model, open("model.pkl", "wb"))

