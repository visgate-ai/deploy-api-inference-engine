import os
import torch
import gc
from diffusers import StableDiffusionPipeline, AudioLDMPipeline
from transformers import pipeline, AutoProcessor, AutoModelForCausalLM
from huggingface_hub import login
import boto3
from botocore.config import Config
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

class ModelManager:
    def __init__(self):
        self.active_model = None
        self.active_model_id = None
        self.r2_client = None
        self.active_model_type = None
        
        # Hugging Face Hub'a giriş yap
        hf_token = os.getenv("HF_HUB_TOKEN")
        if hf_token:
            print("HuggingFace Hub'a giriş yapılıyor...")
            try:
                login(token=hf_token)
            except Exception as e:
                print(f"⚠️ HuggingFace giriş hatası (devam ediliyor): {e}")
        
        # R2 Client oluştur
        self._init_r2_client()
    
    def _init_r2_client(self):
        """R2 S3 client'ı oluştur"""
        try:
            r2_endpoint = os.getenv("R2_ENDPOINT_URL")
            r2_access_key = os.getenv("R2_ACCESS_KEY_ID")
            r2_secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
            
            if r2_endpoint and r2_access_key and r2_secret_key:
                self.r2_client = boto3.client(
                    's3',
                    endpoint_url=r2_endpoint,
                    aws_access_key_id=r2_access_key,
                    aws_secret_access_key=r2_secret_key,
                    config=Config(signature_version='s3v4')
                )
                print("✅ R2 Client başlatıldı")
        except Exception as e:
            print(f"⚠️ R2 Client başlatılamadı: {e}")
    
    def check_model_in_r2(self, model_id: str) -> bool:
        """Modelin R2 bucket'ta olup olmadığını kontrol et"""
        if not self.r2_client:
            return False
        
        models_bucket = os.getenv("R2_MODELS_BUCKET")
        model_path = f"models/{model_id}"
        
        try:
            response = self.r2_client.list_objects_v2(
                Bucket=models_bucket,
                Prefix=model_path,
                MaxKeys=1
            )
            return 'Contents' in response and len(response['Contents']) > 0
        except Exception as e:
            print(f"R2 model kontrol hatası: {e}")
            return False
    
    def download_model_from_r2(self, model_id: str, local_path: str) -> bool:
        """Modeli R2 bucket'tan indir"""
        if not self.r2_client:
            return False
        
        models_bucket = os.getenv("R2_MODELS_BUCKET")
        model_path = f"models/{model_id}"
        
        try:
            response = self.r2_client.list_objects_v2(
                Bucket=models_bucket,
                Prefix=model_path
            )
            
            if 'Contents' not in response:
                return False
            
            os.makedirs(local_path, exist_ok=True)
            file_count = 0
            
            for obj in response['Contents']:
                key = obj['Key']
                relative_path = key[len(model_path):].lstrip('/')
                if relative_path:
                    local_file = os.path.join(local_path, relative_path)
                    os.makedirs(os.path.dirname(local_file), exist_ok=True)
                    
                    print(f"📥 R2'den indiriliyor: {key}")
                    self.r2_client.download_file(models_bucket, key, local_file)
                    file_count += 1
            
            print(f"✅ R2'den {file_count} dosya indirildi")
            return True
            
        except Exception as e:
            print(f"R2 model indirme hatası: {e}")
            return False
    
    def upload_model_to_r2(self, model_id: str, local_path: str) -> bool:
        """İndirilen modeli R2 bucket'a yedekle"""
        if not self.r2_client:
            return False
        
        models_bucket = os.getenv("R2_MODELS_BUCKET")
        model_path = f"models/{model_id}"
        
        try:
            print(f"📤 Model R2'ye yedekleniyor: {local_path} -> {models_bucket}/{model_path}")
            file_count = 0
            
            for root, dirs, files in os.walk(local_path):
                for file in files:
                    local_file = os.path.join(root, file)
                    relative_path = os.path.relpath(local_file, local_path)
                    r2_key = f"{model_path}/{relative_path}"
                    
                    self.r2_client.upload_file(local_file, models_bucket, r2_key)
                    file_count += 1
            
            print(f"✅ R2'ye {file_count} dosya yüklendi")
            return True
            
        except Exception as e:
            print(f"R2 model yükleme hatası: {e}")
            return False
    
    def clear_vram(self):
        """GPU belleğini temizler"""
        if self.active_model is not None:
            print(f"VRAM temizleniyor: {self.active_model_id}")
            del self.active_model
            self.active_model = None
            self.active_model_id = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✅ VRAM temizlendi")

    def load_model(self, model_id: str):
        """
        Modeli yükle - Inference Engine Akışı:
        1. R2 bucket'ta kontrol et
        2. Yoksa HuggingFace'ten indir
        3. R2 bucket'a yedekle
        4. Modeli GPU'ya yükle
        """
        # Aynı model zaten yüklü mü kontrol et
        if self.active_model_id == model_id:
            print(f"ℹ️ Model zaten GPU'da: {model_id}")
            return self.active_model
        
        # Önceki modeli temizle
        if self.active_model is not None:
            self.clear_vram()
        
        print(f"\n📦 Model yükleniyor: {model_id}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        # Model tipini belirle
        model_type = self._get_model_type(model_id)
        print(f"🧠 Model tipi: {model_type}")
        
        # Lokal cache yolu
        cache_dir = f"/root/.cache/huggingface/hub/{model_id.replace('/', '_')}"
        
        # 1. R2 bucket'ta kontrol et
        if self.check_model_in_r2(model_id):
            print(f"📦 Model R2 bucket'ta bulundu, indiriliyor...")
            if not os.path.exists(cache_dir) or not os.listdir(cache_dir):
                self.download_model_from_r2(model_id, cache_dir)
            from__r2 = True
        else:
            print(f"🌐 Model R2'de yok, HuggingFace'ten indiriliyor...")
            from_2 = False
        
        # 2. HuggingFace'ten indir (veya local'den yükle)
        print(f"📥 Model yükleniyor...")
        
        try:
            if model_type == "image-generation":
                ml_pipeline = StableDiffusionPipeline.from_pretrained(
                    model_id if not from_2 else cache_dir, 
                    torch_dtype=dtype
                )
            elif model_type == "audio-generation":
                ml_pipeline = AudioLDMPipeline.from_pretrained(
                    model_id if not from_2 else cache_dir, 
                    torch_dtype=dtype
                )
            elif model_type == "music-generation":
                from transformers import MusicgenForConditionalGeneration
                processor = AutoProcessor.from_pretrained(
                    model_id if not from_2 else cache_dir
                )
                model = MusicgenForConditionalGeneration.from_pretrained(
                    model_id if not from_2 else cache_dir, 
                    torch_dtype=dtype
                )
                ml_pipeline = {"processor": processor, "model": model}
            elif model_type == "text-generation":
                ml_pipeline = pipeline(
                    "text-generation", 
                    model=model_id if not from_2 else cache_dir, 
                    torch_dtype=dtype, 
                    device=device
                )
            elif model_type == "speech-recognition":
                ml_pipeline = pipeline(
                    "automatic-speech-recognition", 
                    model=model_id if not from_2 else cache_dir, 
                    torch_dtype=dtype, 
                    device=device
                )
            elif model_type == "text-to-speech":
                ml_pipeline = pipeline(
                    "text-to-speech", 
                    model=model_id if not from_2 else cache_dir, 
                    torch_dtype=dtype, 
                    device=device
                )
            else:
                raise ValueError(f"Desteklenmeyen model tipi: {model_type}")
        except Exception as e:
            # HF'den indirirken hata olursa R2'den tekrar dene
            if not from_2:
                print(f"⚠️ HF indirme hatası, R2'den deneniyor: {e}")
                self.download_model_from_r2(model_id, cache_dir)
                return self.load_model(model_id)
            raise
        
        # 3. Modeli R2'ye yedekle (eğer HF'den indirildiyse)
        if not from_2 and self.r2_client:
            print(f"💾 Model R2'ye yedekleniyor...")
            self.upload_model_to_r2(model_id, cache_dir)
        
        # 4. GPU optimizasyonları
        if device == "cuda" and model_type in ["image-generation", "audio-generation"]:
            try:
                ml_pipeline.enable_model_cpu_offload()
            except Exception:
                ml_pipeline.to(device)
            try:
                if hasattr(ml_pipeline, "enable_attention_slicing"):
                    ml_pipeline.enable_attention_slicing()
                if hasattr(ml_pipeline, "enable_vae_slicing"):
                    ml_pipeline.enable_vae_slicing()
                    ml_pipeline.enable_vae_tiling()
            except Exception:
                pass
        elif device == "cuda" and model_type == "music-generation":
            ml_pipeline["model"].to(device)
        
        self.active_model = ml_pipeline
        self.active_model_type = model_type
        self.active_model_id = model_id
        
        print(f"✅ Model GPU'ya yüklendi: {model_id}")
        return self.active_model
    
    def _get_model_type(self, model_id: str) -> str:
        """Model tipini HuggingFace model ID'den belirle"""
        model_id_lower = model_id.lower()
        
        if any(x in model_id_lower for x in ["stable-diffusion", "sd-turbo", "wan", "flux", "sdxl"]):
            return "image-generation"
        elif "audioldm" in model_id_lower:
            return "audio-generation"
        elif "musicgen" in model_id_lower:
            return "music-generation"
        elif any(x in model_id_lower for x in ["gpt2", "llama", "mistral", "qwen", "gemma"]):
            return "text-generation"
        elif "whisper" in model_id_lower:
            return "speech-recognition"
        elif any(x in model_id_lower for x in ["tts", "bark", "speecht5", "mms-tts"]):
            return "text-to-speech"
        else:
            return "text-generation"  # Default

    def get_status(self):
        """Sistemin durumunu döndürür"""
        vram_used = 0
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / (1024 ** 2) 
        
        return {
            "active_model": self.active_model_id,
            "active_model_type": self.active_model_type,
            "vram_used_mb": round(vram_used, 2),
            "gpu_available": torch.cuda.is_available(),
            "r2_connected": self.r2_client is not None
        }

# Singleton instance
manager = ModelManager()
