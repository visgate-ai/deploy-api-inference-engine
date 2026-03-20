import os
import shutil
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
load_dotenv()

def get_r2_client():
    """R2 S3 Client oluştur"""
    return boto3.client(
        's3',
        endpoint_url=os.getenv('R2_ENDPOINT_URL'),
        aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY'),
        config=Config(signature_version='s3v4')
    )

def download_input_from_r2(object_name: str, file_name: str) -> bool:
    """
    R2 input bucket'tan dosya indirir
    
    Args:
        object_name: R2'deki dosya yolu (örn: 'inference/inputs/job_xxx/input.png')
        file_name: Yerel kayıt yolu
    
    Returns:
        bool: Başarılı ise True
    """
    s3_client = get_r2_client()
    input_bucket = os.getenv('R2_INPUT_BUCKET')
    
    try:
        print(f"📥 R2 Input'dan indiriliyor: {input_bucket}/{object_name} -> {file_name}")
        s3_client.download_file(input_bucket, object_name, file_name)
        return True
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == '404':
            print(f"⚠️ R2'de dosya bulunamadı: {object_name}")
        else:
            print(f"❌ R2 İndirme Hatası: {e}")
        return False

def upload_output_to_r2(file_name: str, object_name: str) -> bool:
    """
    R2 output bucket'a dosya yükler
    
    Args:
        file_name: Yerel dosya yolu
        object_name: R2'deki hedef yol (örn: 'inference/outputs/xxx/result.png')
    
    Returns:
        bool: Başarılı ise True
    """
    s3_client = get_r2_client()
    output_bucket = os.getenv('R2_OUTPUT_BUCKET')
    
    try:
        print(f"📤 R2 Output'a yükleniyor: {file_name} -> {output_bucket}/{object_name}")
        s3_client.upload_file(file_name, output_bucket, object_name)
        return True
    except ClientError as e:
        print(f"❌ R2 Yükleme Hatası: {e}")
        return False
