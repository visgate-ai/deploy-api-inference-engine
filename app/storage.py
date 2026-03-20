import os
import shutil
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
load_dotenv()

print("📦 Storage module loaded", flush=True)

def get_r2_client():
    """R2 S3 Client oluştur"""
    print("🔧 Creating R2 client...", flush=True)
    try:
        client = boto3.client(
            's3',
            endpoint_url=os.getenv('R2_ENDPOINT_URL'),
            aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY'),
            config=Config(signature_version='s3v4')
        )
        print("✅ R2 client created", flush=True)
        return client
    except Exception as e:
        print(f"❌ R2 client creation failed: {e}", flush=True)
        raise

def download_input_from_r2(object_name: str, file_name: str) -> bool:
    """
    R2 input bucket'tan dosya indirir
    """
    print(f"\n📥 DOWNLOAD INPUT: {object_name} -> {file_name}", flush=True)
    
    try:
        s3_client = get_r2_client()
        input_bucket = os.getenv('R2_INPUT_BUCKET')
        
        print(f"   Bucket: {input_bucket}", flush=True)
        print(f"   Object: {object_name}", flush=True)
        print(f"   Local file: {file_name}", flush=True)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        
        print(f"   Downloading...", flush=True)
        s3_client.download_file(input_bucket, object_name, file_name)
        
        file_size = os.path.getsize(file_name) if os.path.exists(file_name) else 0
        print(f"   ✅ Downloaded successfully! Size: {file_size} bytes", flush=True)
        return True
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        print(f"   ❌ Download failed - Error: {error_code}", flush=True)
        print(f"   Message: {e}", flush=True)
        if error_code == '404':
            print(f"   ⚠️ File not found in R2: {object_name}", flush=True)
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False

def upload_output_to_r2(file_name: str, object_name: str) -> bool:
    """
    R2 output bucket'a dosya yükler
    """
    print(f"\n📤 UPLOAD OUTPUT: {file_name} -> {object_name}", flush=True)
    
    try:
        s3_client = get_r2_client()
        output_bucket = os.getenv('R2_OUTPUT_BUCKET')
        
        print(f"   Bucket: {output_bucket}", flush=True)
        print(f"   Local file: {file_name}", flush=True)
        print(f"   Object: {object_name}", flush=True)
        
        # Check file exists
        if not os.path.exists(file_name):
            print(f"   ❌ File not found: {file_name}", flush=True)
            return False
        
        file_size = os.path.getsize(file_name)
        print(f"   File size: {file_size} bytes", flush=True)
        print(f"   Uploading...", flush=True)
        
        s3_client.upload_file(file_name, output_bucket, object_name)
        
        print(f"   ✅ Uploaded successfully!", flush=True)
        return True
        
    except ClientError as e:
        print(f"   ❌ Upload failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False

print("✅ Storage module initialized", flush=True)
