import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def run_preprocessing(raw_data_path, output_folder):
    print("Mulai proses preprocessing otomatis...")
    
    # Memuat Dataset
    print(f"Membaca data dari: {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    
    # Menangani Missing Values
    df_clean = df.dropna().copy()
    print(f"Sisa baris setelah menghapus data kosong: {len(df_clean)}")
    
    # Encoding Data Kategorikal
    le = LabelEncoder()
    df_clean['type'] = le.fit_transform(df_clean['type'])
    
    # Memisahkan Fitur (X) dan Target (y)
    X = df_clean.drop('quality', axis=1)
    y = df_clean['quality']
    
    # Membagi data menjadi Train & Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standarisasi Fitur
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Konversi Kembali Ke DataFrame
    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # Simpan Hasil Preprocessing Ke Folder
    os.makedirs(output_folder, exist_ok=True)
    
    X_train_df.to_csv(os.path.join(output_folder, 'X_train.csv'), index=False)
    X_test_df.to_csv(os.path.join(output_folder, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_folder, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_folder, 'y_test.csv'), index=False)
    
    print(f"Preprocessing selesai!: {output_folder}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    RAW_DATA_PATH = os.path.join(current_dir, '..', 'wine_quality_raw', 'wine-quality.csv')
    OUTPUT_FOLDER = os.path.join(current_dir, 'wine_quality_preprocessing')
    
    run_preprocessing(RAW_DATA_PATH, OUTPUT_FOLDER)
