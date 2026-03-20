import os
import sys
import torch
import gc
from diffusers import StableDiffusionPipeline, AudioLDMPipeline
from transformers import pipeline, AutoProcessor, AutoModelForCausalLM
from huggingface_hub import login
import boto3
from botocore.config import Config
from dotenv import load_dotenv

print("="*60, flush=True)
print("📦 MODEL MANAGER MODULE LOADING", flush=True)
print("="*60, flush=True)

# .env dosyasını yükle
env_path = os.path.join(os.path.dirname(__file__), '.env')
print(f"📂 Loading .env from: {env_path}", flush=True)
load_dotenv(env_path)
print("✅ .env loaded", flush=True)

class ModelManager:
    def __init__(self):
        print("\n🔧 ModelManager.__init__() starting...", flush=True)
        self.active_model = None
        self.active_model_id = None
        self.r2_client = None
        self.active_model_type = None
        
        # Hugging Face Hub'a giriş yap
        hf_token = os.getenv("HF_HUB_TOKEN")
        print(f"   HF_HUB_TOKEN: {'SET' if hf_token else 'NOT SET'}", flush=True)
        if hf_token:
            print("🔐 Attempting HuggingFace login...", flush=True)
            try:
                login(token=hf_token)
                print("✅ HuggingFace login successful", flush=True)
            except Exception as e:
                print(f"⚠️ HuggingFace login failed (continuing without login): {e}", flush=True)
                import traceback
                traceback.print_exc()
        
        # R2 Client oluştur
        print("🔧 Initializing R2 client...", flush=True)
        self._init_r2_client()
        print("✅ ModelManager initialized!", flush=True)
    
    def _init_r2_client(self):
        """R2 S3 client'ı oluştur"""
        print("\n📡 _init_r2_client() called", flush=True)
        try:
            r2_endpoint = os.getenv("R2_ENDPOINT_URL")
            r2_access_key = os.getenv("R2_ACCESS_KEY_ID")
            r2_secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
            
            print(f"   R2_ENDPOINT_URL: {'SET' if r2_endpoint else 'NOT SET'}", flush=True)
            print(f"   R2_ACCESS_KEY_ID: {'SET' if r2_access_key else 'NOT SET'}", flush=True)
            print(f"   R2_SECRET_ACCESS_KEY: {'SET' if r2_secret_key else 'NOT SET'}", flush=True)
            
            if r2_endpoint and r2_access_key and r2_secret_key:
                print("🔄 Creating boto3 S3 client...", flush=True)
                self.r2_client = boto3.client(
                    's3',
                    endpoint_url=r2_endpoint,
                    aws_access_key_id=r2_access_key,
                    aws_secret_access_key=r2_secret_key,
                    config=Config(signature_version='s3v4')
                )
                print("✅ R2 Client başlatıldı", flush=True)
            else:
                print("⚠️ R2 credentials not set, R2 disabled", flush=True)
        except Exception as e:
            print(f"❌ R2 Client başlatılamadı: {e}", flush=True)
            import traceback
            traceback.print_exc()
    
    def check_model_in_r2(self, model_id: str) -> bool:
        """Modelin R2 bucket'ta olup olmadığını kontrol et"""
        print(f"\n🔍 check_model_in_r2({model_id})", flush=True)
        
        if not self.r2_client:
            print("   R2 client not initialized, returning False", flush=True)
            return False
        
        models_bucket = os.getenv("R2_MODELS_BUCKET")
        model_path = f"models/{model_id}"
        
        print(f"   Bucket: {models_bucket}", flush=True)
        print(f"   Path: {model_path}", flush=True)
        
        try:
            print("   Querying R2...", flush=True)
            response = self.r2_client.list_objects_v2(
                Bucket=models_bucket,
                Prefix=model_path,
                MaxKeys=1
            )
            
            has_model = 'Contents' in response and len(response['Contents']) > 0
            print(f"   Model in R2: {has_model}", flush=True)
            return has_model
        except Exception as e:
            print(f"❌ R2 model kontrol hatası: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return False
    
    def download_model_from_r2(self, model_id: str, local_path: str) -> bool:
        """Modeli R2 bucket'tan indir"""
        print(f"\n📥 download_model_from_r2({model_id}, {local_path})", flush=True)
        
        if not self.r2_client:
            print("   R2 client not initialized, returning False", flush=True)
            return False
        
        models_bucket = os.getenv("R2_MODELS_BUCKET")
        model_path = f"models/{model_id}"
        
        print(f"   Bucket: {models_bucket}", flush=True)
        print(f"   Path: {model_path}", flush=True)
        
        try:
            print("   Listing objects in R2...", flush=True)
            response = self.r2_client.list_objects_v2(
                Bucket=models_bucket,
                Prefix=model_path
            )
            
            if 'Contents' not in response:
                print("   No objects found in R2", flush=True)
                return False
            
            os.makedirs(local_path, exist_ok=True)
            file_count = 0
            total_size = 0
            
            for obj in response['Contents']:
                key = obj['Key']
                relative_path = key[len(model_path):].lstrip('/')
                if relative_path:
                    local_file = os.path.join(local_path, relative_path)
                    os.makedirs(os.path.dirname(local_file), exist_ok=True)
                    
                    print(f"   Downloading: {key}", flush=True)
                    self.r2_client.download_file(models_bucket, key, local_file)
                    file_size = os.path.getsize(local_file)
                    total_size += file_size
                    file_count += 1
            
            print(f"✅ R2'den {file_count} dosya indirildi (Total: {total_size} bytes)", flush=True)
            return True
            
        except Exception as e:
            print(f"❌ R2 model indirme hatası: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return False
    
    def upload_model_to_r2(self, model_id: str, local_path: str) -> bool:
        """İndirilen modeli R2 bucket'a yedekle"""
        print(f"\n📤 upload_model_to_r2({model_id}, {local_path})", flush=True)
        
        if not self.r2_client:
            print("   R2 client not initialized, skipping upload", flush=True)
            return False
        
        models_bucket = os.getenv("R2_MODELS_BUCKET")
        model_path = f"models/{model_id}"
        
        print(f"   Bucket: {models_bucket}", flush=True)
        print(f"   Path: {model_path}", flush=True)
        
        try:
            print(f"   Walking directory: {local_path}", flush=True)
            file_count = 0
            total_size = 0
            
            for root, dirs, files in os.walk(local_path):
                for file in files:
                    local_file = os.path.join(root, file)
                    relative_path = os.path.relpath(local_file, local_path)
                    r2_key = f"{model_path}/{relative_path}"
                    
                    file_size = os.path.getsize(local_file)
                    print(f"   Uploading: {r2_key} ({file_size} bytes)", flush=True)
                    self.r2_client.upload_file(local_file, models_bucket, r2_key)
                    total_size += file_size
                    file_count += 1
            
            print(f"✅ R2'ye {file_count} dosya yüklendi (Total: {total_size} bytes)", flush=True)
            return True
            
        except Exception as e:
            print(f"❌ R2 model yükleme hatası: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return False
    
    def clear_vram(self):
        """GPU belleğini temizler"""
        print("\n🧹 clear_vram() called", flush=True)
        if self.active_model is not None:
            print(f"   Deleting model: {self.active_model_id}", flush=True)
            del self.active_model
            self.active_model = None
            self.active_model_id = None
            print("   Model deleted", flush=True)
        
        print("   Running garbage collection...", flush=True)
        gc.collect()
        
        if torch.cuda.is_available():
            print("   Clearing CUDA cache...", flush=True)
            torch.cuda.empty_cache()
        
        print("✅ VRAM temizlendi", flush=True)

    def load_model(self, model_id: str):
        """
        Modeli yükle - Inference Engine Akışı:
        1. R2 bucket'ta kontrol et
        2. Yoksa HuggingFace'ten indir
        3. R2 bucket'a yedekle
        4. Modeli GPU'ya yükle
        """
        print(f"\n{'='*60}", flush=True)
        print(f"📦 LOAD MODEL: {model_id}", flush=True)
        print(f"{'='*60}", flush=True)
        
        # Aynı model zaten yüklü mü kontrol et
        if self.active_model_id == model_id:
            print(f"ℹ️ Model zaten GPU'da: {model_id}", flush=True)
            return self.active_model
        
        # Önceki modeli temizle
        if self.active_model is not None:
            print("   Previous model found, clearing...", flush=True)
            self.clear_vram()
        
        print(f"\n📦 Model yükleniyor: {model_id}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        print(f"   Device: {device}", flush=True)
        print(f"   Dtype: {dtype}", flush=True)
        
        # Model tipini belirle
        model_type = self._get_model_type(model_id)
        print(f"🧠 Model tipi: {model_type}", flush=True)
        
        # Lokal cache yolu
        cache_dir = f"/root/.cache/huggingface/hub/{model_id.replace('/', '_')}"
        print(f"   Cache dir: {cache_dir}", flush=True)
        
        # 1. R2 bucket'ta kontrol et
        from_r2 = False
        if self.check_model_in_r2(model_id):
            print(f"📦 Model R2 bucket'ta bulundu, indiriliyor...", flush=True)
            if not os.path.exists(cache_dir) or not os.listdir(cache_dir):
                print("   Cache empty, downloading from R2...", flush=True)
                self.download_model_from_r2(model_id, cache_dir)
            else:
                print("   Cache exists, skipping download", flush=True)
            from_r2 = True
        else:
            print(f"🌐 Model R2'de yok, HuggingFace'ten indiriliyor...", flush=True)
            from_r2 = False
        
        # 2. HuggingFace'ten indir (veya local'den yükle)
        print(f"\n📥 Model yükleniyor from {'R2 cache' if from_r2 else 'HuggingFace'}...", flush=True)
        print(f"   Source: {model_id if not from_r2 else cache_dir}", flush=True)
        
        try:
            if model_type == "image-generation":
                print("   Loading StableDiffusionPipeline...", flush=True)
                ml_pipeline = StableDiffusionPipeline.from_pretrained(
                    model_id if not from_r2 else cache_dir, 
                    torch_dtype=dtype
                )
            elif model_type == "audio-generation":
                print("   Loading AudioLDMPipeline...", flush=True)
                ml_pipeline = AudioLDMPipeline.from_pretrained(
                    model_id if not from_r2 else cache_dir, 
                    torch_dtype=dtype
                )
            elif model_type == "music-generation":
                print("   Loading MusicGen...", flush=True)
                from transformers import MusicgenForConditionalGeneration
                processor = AutoProcessor.from_pretrained(
                    model_id if not from_r2 else cache_dir
                )
                model = MusicgenForConditionalGeneration.from_pretrained(
                    model_id if not from_r2 else cache_dir, 
                    torch_dtype=dtype
                )
                ml_pipeline = {"processor": processor, "model": model}
            elif model_type == "text-generation":
                print("   Loading text-generation pipeline...", flush=True)
                ml_pipeline = pipeline(
                    "text-generation", 
                    model=model_id if not from_r2 else cache_dir, 
                    torch_dtype=dtype, 
                    device=device
                )
            elif model_type == "speech-recognition":
                print("   Loading speech-recognition pipeline...", flush=True)
                ml_pipeline = pipeline(
                    "automatic-speech-recognition", 
                    model=model_id if not from_r2 else cache_dir, 
                    torch_dtype=dtype, 
                    device=device
                )
            elif model_type == "text-to-speech":
                print("   Loading text-to-speech pipeline...", flush=True)
                ml_pipeline = pipeline(
                    "text-to-speech", 
                    model=model_id if not from_r2 else cache_dir, 
                    torch_dtype=dtype, 
                    device=device
                )
            else:
                raise ValueError(f"Desteklenmeyen model tipi: {model_type}")
        except Exception as e:
            print(f"❌ Model loading failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            
            # HF'den indirirken hata olursa R2'den tekrar dene
            if not from_r2:
                print("   Retrying from R2...", flush=True)
                self.download_model_from_r2(model_id, cache_dir)
                return self.load_model(model_id)
            raise
        
        print("✅ Model downloaded successfully", flush=True)
        
        # 3. Modeli R2'ye yedekle (eğer HF'den indirildiyse)
        if not from_r2 and self.r2_client:
            print("💾 Backing up model to R2...", flush=True)
            self.upload_model_to_r2(model_id, cache_dir)
        
        # 4. GPU optimizasyonları
        print("⚡ Applying GPU optimizations...", flush=True)
        if device == "cuda" and model_type in ["image-generation", "audio-generation"]:
            try:
                print("   Enabling model CPU offload...", flush=True)
                ml_pipeline.enable_model_cpu_offload()
            except Exception as e:
                print(f"   CPU offload failed, using to(): {e}", flush=True)
                ml_pipeline.to(device)
            try:
                if hasattr(ml_pipeline, "enable_attention_slicing"):
                    print("   Enabling attention slicing...", flush=True)
                    ml_pipeline.enable_attention_slicing()
                if hasattr(ml_pipeline, "enable_vae_slicing"):
                    print("   Enabling VAE slicing...", flush=True)
                    ml_pipeline.enable_vae_slicing()
                    ml_pipeline.enable_vae_tiling()
            except Exception as e:
                print(f"   VAE optimization warning: {e}", flush=True)
        elif device == "cuda" and model_type == "music-generation":
            print("   Moving MusicGen model to GPU...", flush=True)
            ml_pipeline["model"].to(device)
        
        self.active_model = ml_pipeline
        self.active_model_type = model_type
        self.active_model_id = model_id
        
        # Report VRAM usage
        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / (1024 ** 2)
            print(f"   GPU VRAM used: {vram:.2f} MB", flush=True)
        
        print(f"\n{'='*60}", flush=True)
        print(f"✅ Model GPU'ya yüklendi: {model_id}", flush=True)
        print(f"{'='*60}", flush=True)
        return self.active_model
    
    def _get_model_type(self, model_id: str) -> str:
        """Model tipini HuggingFace model ID'den belirle"""
        print(f"\n🔎 _get_model_type({model_id})", flush=True)
        model_id_lower = model_id.lower()
        
        if any(x in model_id_lower for x in ["stable-diffusion", "sd-turbo", "wan", "flux", "sdxl"]):
            print(f"   Detected: image-generation", flush=True)
            return "image-generation"
        elif "audioldm" in model_id_lower:
            print(f"   Detected: audio-generation", flush=True)
            return "audio-generation"
        elif "musicgen" in model_id_lower:
            print(f"   Detected: music-generation", flush=True)
            return "music-generation"
        elif any(x in model_id_lower for x in ["gpt2", "llama", "mistral", "qwen", "gemma"]):
            print(f"   Detected: text-generation", flush=True)
            return "text-generation"
        elif "whisper" in model_id_lower:
            print(f"   Detected: speech-recognition", flush=True)
            return "speech-recognition"
        elif any(x in model_id_lower for x in ["tts", "bark", "speecht5", "mms-tts"]):
            print(f"   Detected: text-to-speech", flush=True)
            return "text-to-speech"
        else:
            print(f"   Detected: text-generation (default)", flush=True)
            return "text-generation"

    def get_status(self):
        """Sistemin durumunu döndürür"""
        print("\n📊 get_status() called", flush=True)
        vram_used = 0
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / (1024 ** 2) 
        
        status = {
            "active_model": self.active_model_id,
            "active_model_type": self.active_model_type,
            "vram_used_mb": round(vram_used, 2),
            "gpu_available": torch.cuda.is_available(),
            "r2_connected": self.r2_client is not None
        }
        print(f"   Status: {status}", flush=True)
        return status

print("\n🔄 Creating singleton manager instance...", flush=True)
# Singleton instance
manager = ModelManager()
print("✅ Singleton manager created!", flush=True)
print("="*60, flush=True)
