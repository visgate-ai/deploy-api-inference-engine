import os
import torch
import gc
from diffusers import StableDiffusionPipeline, AudioLDMPipeline
from transformers import pipeline, AutoProcessor, AutoModelForCausalLM, AutoModelForSpeechSeq2Seq, AutoModelForTextToWaveform
from huggingface_hub import login

class ModelManager:
    def __init__(self):
        self.active_model = None
        self.active_model_id = None
        
        # Hugging Face Hub'a giriş yap (Kapalı modeller için)
        hf_token = os.getenv("HF_HUB_TOKEN")
        if hf_token and hf_token != "senin_huggingface_tokenin":
            print("Hugging Face Hub'a giriş yapılıyor...")
            login(token=hf_token)
        else:
            print("Uyarı: Geçerli bir HF_HUB_TOKEN bulunamadı. Gated (kapalı) modeller indirilemeyebilir.")

    def clear_vram(self):
        """GPU belleğini acımasızca temizler"""
        if self.active_model is not None:
            print(f"Bellek boşaltılıyor: {self.active_model_id}")
            del self.active_model
            self.active_model = None
            self.active_model_id = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("VRAM Temizlendi!")

    def load_model(self, model_id):
        """Modeli yükler. Zaten yüklüyse dokunmaz, farklıysa eskisini siler."""
        if self.active_model_id == model_id:
            print(f"Model {model_id} zaten GPU'da yüklü.")
            return self.active_model

        if self.active_model is not None:
            self.clear_vram()

        print(f"Yeni model yükleniyor: {model_id} ...")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Kullanıcının talebi üzerine: GPU desteklenmiyorsa CPU'ya düşmek yerine direkt hata döndürüyoruz.
        if device == "cuda" and torch.cuda.get_device_capability(0)[0] < 7:
            raise ValueError(
                f"Sisteminizdeki GPU (Örn: MX150, Compute Capability {torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}), "
                "güncel PyTorch kütüphanesi tarafından donanımsal düzeyde desteklenmemektedir (Minimum 7.0 gerekmektedir). "
                "Cihazınızda hiçbir model (küçük modeller dahil) GPU üzerinde donanım uyuşmazlığından dolayı çalıştırılamaz! Lütfen CPU moduna (daha eski PyTorch sürümlerine vs.) dönerek sisteminizi konfigüre edin."
            )

        # Model yükleme: 2GB VRAM için optimize edildi
        # Modele göre farklı Pipeline/AutoModel kullan...
        
        model_type = "unknown"
        if "stable-diffusion" in model_id.lower() or "sd-turbo" in model_id.lower():
            model_type = "image-generation"
        elif "audioldm" in model_id.lower():
            model_type = "audio-generation"
        elif "musicgen" in model_id.lower():
            model_type = "music-generation"
        elif "gpt2" in model_id.lower():
            model_type = "text-generation"
        elif "whisper" in model_id.lower():
            model_type = "speech-recognition"

        dtype = torch.float16 if device == "cuda" else torch.float32

        if model_type == "image-generation":
            ml_pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
        elif model_type == "audio-generation":
            ml_pipeline = AudioLDMPipeline.from_pretrained(model_id, torch_dtype=dtype)
        elif model_type == "music-generation":
            from transformers import AutoProcessor, MusicgenForConditionalGeneration
            processor = AutoProcessor.from_pretrained(model_id)
            model = MusicgenForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
            ml_pipeline = {"processor": processor, "model": model, "type": "music-generation"}
        elif model_type == "text-generation":
            ml_pipeline = pipeline("text-generation", model=model_id, torch_dtype=dtype, device=device if device == "cuda" else "cpu")
        elif model_type == "speech-recognition":
            ml_pipeline = pipeline("automatic-speech-recognition", model=model_id, torch_dtype=dtype, device=device if device == "cuda" else "cpu")
        else:
            raise ValueError(f"Desteklenmeyen model tipi: {model_id}")

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
                if hasattr(ml_pipeline, "enable_xformers_memory_efficient_attention"):
                    ml_pipeline.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        elif device == "cuda" and model_type == "music-generation":
            ml_pipeline["model"].to(device)

        self.active_model = ml_pipeline
        self.active_model_type = model_type
        self.active_model_id = model_id
        return self.active_model

    def get_status(self):
        """Sistemin o anki durumunu döndürür"""
        vram_used = 0
        if torch.cuda.is_available():
            # Kullanılan VRAM'i MB cinsinden hesapla
            vram_used = torch.cuda.memory_allocated() / (1024 ** 2) 
        
        return {
            "active_model": self.active_model_id,
            "vram_used_mb": round(vram_used, 2),
            "gpu_available": torch.cuda.is_available()
        }

# Tüm uygulamada tek bir Manager örneği kullanılacak
manager = ModelManager()