import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import zscore
import os

# --- 1. Load Data ---
RAW_PATH = "namadataset_raw/jobs_dataset.csv"
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

# --- 3. Feature Engineering ---
df['desc_len'] = df['description'].astype(str).apply(len)
df['posname_len'] = df['positionName'].astype(str).apply(len)

# --- 4. Handle Duplicates ---
df = df.drop_duplicates()

# --- 5. Winsorizing (IQR 1.0) ---
for col in ['salary_num', 'desc_len', 'posname_len']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.0 * IQR
    upper = Q3 + 1.0 * IQR
    df[col] = df[col].clip(lower, upper)

# --- 6. Encoding ---
categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()

# Fill missing values in categorical columns before encoding
df[categorical_cols] = df[categorical_cols].fillna("Missing")

# One-hot encode
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = ohe.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(categorical_cols), index=df.index)

# Combine encoded and numeric data
df_encoded = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

# --- 7. Binning Rating ---
if 'rating' in df_encoded.columns:
    df_encoded['rating_bin'] = pd.cut(df_encoded['rating'], bins=3, labels=["Low", "Medium", "High"])

# --- 8. Binning Salary (wajib untuk target klasifikasi) ---
bins_salary = [0, 70000, 120000, 200000, df['salary_num'].max()]
labels_salary = ['Low', 'Medium', 'High', 'Very High']
df_encoded['salary_bin'] = pd.cut(df['salary_num'], bins=bins_salary, labels=labels_salary)

# --- 9. Save Output ---
OUTPUT_DIR = "namadataset_preprocessing"
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = f"{OUTPUT_DIR}/data_preprocessed.csv"
df_encoded.to_csv(output_path, index=False)

# --- Done ---
print("✅ Preprocessing selesai. File disimpan di:", output_path)
print("📊 Jumlah baris:", df_encoded.shape[0])
print("📊 Jumlah kolom:", df_encoded.shape[1])
print(df_encoded.head())
