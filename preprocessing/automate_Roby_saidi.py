import pandas as pd
import numpy as np
import re
import os
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

# --- 1. Load Data ---
RAW_PATH = "../namadataset_raw/jobs_dataset.csv"
df = pd.read_csv(RAW_PATH)

# --- 2. Preprocessing Salary ---
def extract_salary(s):
    try:
        numbers = re.findall(r'[\d,]+', str(s))
        if len(numbers) >= 1:
            return int(numbers[0].replace(',', ''))
    except:
        return None
    return None

df['salary_num'] = df['salary'].apply(extract_salary)
df['desc_len'] = df['description'].astype(str).apply(len)
df['posname_len'] = df['positionName'].astype(str).apply(len)

# --- 3. Missing Values ---
df = df.dropna(subset=['salary'])
job_cols = ['jobType/0', 'jobType/1', 'jobType/2', 'jobType/3']
df[job_cols] = df[job_cols].fillna(False)
df['externalApplyLink'] = df['externalApplyLink'].fillna('Unavailable')

# --- 4. Drop Duplicates ---
df = df.drop_duplicates(subset=['company', 'positionName', 'location', 'salary', 'description'])

# --- 5. Binning Salary dan Rating ---
bins_salary = [0, 70000, 120000, 200000, df['salary_num'].max()]
labels_salary = ['Low', 'Medium', 'High', 'Very High']
df['salary_bin'] = pd.cut(df['salary_num'], bins=bins_salary, labels=labels_salary)

bins_rating = [0, 2, 3.5, 4.5, 5]
labels_rating = ['Poor', 'Average', 'Good', 'Excellent']
df['rating_bin'] = pd.cut(df['rating'], bins=bins_rating, labels=labels_rating)

df = df[df['rating_bin'].isin(['Good', 'Excellent'])].copy()
df = df.dropna(subset=['salary_bin'])

# --- 6. Split ---
y = df['salary_bin']
X = df.drop(columns=['salary_bin'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- 7. Scaling ---
cols_to_scale = ['rating', 'salary_num', 'desc_len', 'posname_len']
scaler = StandardScaler()
X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

# --- 8. Outlier Handling (IQR clipping) ---
for col in cols_to_scale:
    Q1 = X_train[col].quantile(0.25)
    Q3 = X_train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    X_train[col] = X_train[col].clip(lower, upper)

# --- 9. Encoding ---
cols_to_encode = [
    'company', 'location', 'positionName',
    'jobType/0', 'jobType/1', 'jobType/2', 'jobType/3',
    'searchInput/country', 'searchInput/position'
]
X_train = pd.get_dummies(X_train, columns=cols_to_encode, drop_first=True)
X_test = pd.get_dummies(X_test, columns=cols_to_encode, drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# --- 10. Simpan Output ---
Xy_train = X_train.copy()
Xy_train['salary_bin'] = y_train

os.makedirs("namadataset_preprocessing", exist_ok=True)
output_path = "namadataset_preprocessing/data_preprocessed.csv"
Xy_train.to_csv(output_path, index=False)

print("✅ File disimpan di:", output_path)
print("📊 Jumlah baris:", Xy_train.shape[0])
print("📊 Jumlah kolom:", Xy_train.shape[1])
