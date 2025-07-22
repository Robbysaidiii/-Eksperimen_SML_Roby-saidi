import pandas as pd
import numpy as np
import re
import os
from sklearn.preprocessing import OneHotEncoder

# --- 1. Load Data ---
RAW_PATH = "namadataset_raw/jobs_dataset.csv"
df = pd.read_csv(RAW_PATH)

# --- 2. Preprocessing Salary ---
def extract_salary(s):
    """Ambil angka pertama dari kolom salary"""
    try:
        numbers = re.findall(r'[\d,]+', str(s))
        if numbers:
            return int(numbers[0].replace(',', ''))
    except Exception:
        return None
    return None

df['salary_num'] = df['salary'].apply(extract_salary)

# --- 3. Feature Engineering ---
df['desc_len'] = df['description'].astype(str).apply(len)
df['posname_len'] = df['positionName'].astype(str).apply(len)

# --- 4. Drop Duplicates ---
df = df.drop_duplicates()

# --- 5. Winsorizing (IQR = 1.0) ---
num_cols = ['salary_num', 'desc_len', 'posname_len']
for col in num_cols:
    # Isi NaN dengan median sebelum winsorizing
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.0 * IQR
    upper = Q3 + 1.0 * IQR
    df[col] = df[col].clip(lower, upper)

# --- 6. One-Hot Encoding ---
categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
df[categorical_cols] = df[categorical_cols].fillna("Missing")

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_array = ohe.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_array, columns=ohe.get_feature_names_out(categorical_cols), index=df.index)

# Gabungkan kembali numerik + encoded kategorik
df_encoded = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

# --- 7. Binning Rating ---
if 'rating' in df_encoded.columns and df_encoded['rating'].notna().all():
    try:
        df_encoded['rating_bin'] = pd.cut(df_encoded['rating'], bins=3, labels=["Low", "Medium", "High"])
    except Exception as e:
        print(f"⚠️ Gagal binning rating: {e}")

# --- 8. Binning Salary (target klasifikasi) ---
if df['salary_num'].notna().any():
    try:
        max_salary = df['salary_num'].max()
        if not np.isnan(max_salary) and max_salary > 200000:
            bins_salary = [0, 70000, 120000, 200000, max_salary + 1]
            labels_salary = ['Low', 'Medium', 'High', 'Very High']
            df_encoded['salary_bin'] = pd.cut(df['salary_num'], bins=bins_salary, labels=labels_salary)
        elif df['salary_num'].nunique() >= 4:
            # fallback ke qcut kalau max_salary terlalu kecil
            df_encoded['salary_bin'] = pd.qcut(df['salary_num'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
            print("⚠️ Menggunakan qcut karena salary_num max terlalu kecil untuk bin tetap.")
        else:
            print("⚠️ Tidak cukup variasi pada salary_num untuk melakukan binning.")
    except Exception as e:
        print(f"❌ Gagal binning salary: {e}")
else:
    print("❌ Tidak ada nilai salary_num yang valid. Binning salary dibatalkan.")

# --- 9. Simpan Output ---
OUTPUT_DIR = "namadataset_preprocessing"
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, "data_preprocessed.csv")
df_encoded.to_csv(output_path, index=False)

# --- Done ---
print("✅ Preprocessing selesai. File disimpan di:", output_path)
print("📊 Jumlah baris:", df_encoded.shape[0])
print("📊 Jumlah kolom:", df_encoded.shape[1])
print(df_encoded.head())
