import os
import shutil
import boto3
from botocore.exceptions import ClientError

# Yerel test modu açık mı kontrol et
USE_LOCAL_STORAGE = os.getenv("USE_LOCAL_STORAGE", "false").lower() == "true"
LOCAL_INPUT_DIR = "/app/local_inputs"
LOCAL_OUTPUT_DIR = "/app/local_outputs"

def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=os.getenv('R2_ENDPOINT_URL'),
        aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY')
    )

def download_input_from_r2(object_name, file_name):
    if USE_LOCAL_STORAGE:
        # R2 yerine yerel klasörden (local_inputs) kopyala
        local_path = os.path.join(LOCAL_INPUT_DIR, object_name)
        print(f"[YEREL MOD] Dosya alınıyor: {local_path} -> {file_name}")
        if os.path.exists(local_path):
            shutil.copy(local_path, file_name)
            return True
        print(f"Hata: Yerel dosya bulunamadı: {local_path}")
        return False

    # Production (R2) Modu
    s3_client = get_s3_client()
    bucket_name = os.getenv('R2_INPUT_BUCKET_NAME')
    try:
        print(f"R2'den indiriliyor: {object_name}...")
        s3_client.download_file(bucket_name, object_name, file_name)
        return True
    except ClientError as e:
        print(f"İndirme Hatası: {e}")
        return False

def upload_output_to_r2(file_name, object_name):
    if USE_LOCAL_STORAGE:
        # R2 yerine yerel klasöre (local_outputs) kopyala
        local_path = os.path.join(LOCAL_OUTPUT_DIR, object_name)
        
        # Klasör yapısı yoksa oluştur (Örn: outputs/user_1/resim.png)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        print(f"[YEREL MOD] Dosya kaydediliyor: {file_name} -> {local_path}")
        shutil.copy(file_name, local_path)
        return True

    # Production (R2) Modu
    s3_client = get_s3_client()
    bucket_name = os.getenv('R2_OUTPUT_BUCKET_NAME')
    try:
        print(f"R2'ye yükleniyor: {object_name}...")
        s3_client.upload_file(file_name, bucket_name, object_name)
        return True
    except ClientError as e:
        print(f"Yükleme Hatası: {e}")
        return False