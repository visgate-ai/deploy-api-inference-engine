import os
import sys
import threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

print("="*60, flush=True)
print("🚀 VISGATE INFERENCE ENGINE STARTING", flush=True)
print("="*60, flush=True)

# Print all environment variables (sanitized)
print("\n📋 Environment Variables:", flush=True)
for key in ['HF_HUB_TOKEN', 'R2_ENDPOINT_URL', 'R2_ACCESS_KEY_ID', 'R2_MODELS_BUCKET', 'R2_INPUT_BUCKET', 'R2_OUTPUT_BUCKET']:
    val = os.getenv(key, 'NOT SET')
    if val and val != 'NOT SET' and 'KEY' in key:
        val = val[:8] + "..." + val[-4:] if len(val) > 12 else "***"
    print(f"  {key}: {val}", flush=True)

# .env dosyasındaki ayarları yükle
load_dotenv()

print("\n📦 Importing modules...", flush=True)
from .storage import download_input_from_r2, upload_output_to_r2
from .model_manager import manager

print("✅ All modules imported successfully", flush=True)
print("="*60, flush=True)

app = FastAPI(title="Visgate AI Sunucusuz GPU Worker")

# --- KİLİT MEKANİZMASI (GPU KORUMASI) ---
gpu_lock = threading.Lock()

class LoadModelRequest(BaseModel):
    model_id: str

class PredictRequest(BaseModel):
    model_id: str
    prompt: str
    input_image_key: str | None = None
    output_image_key: str

# --- ENDPOINT'LER ---

@app.get("/health")
def health_check():
    print("\n🔵 /health called", flush=True)
    status = manager.get_status()
    print(f"   Status: {status}", flush=True)
    return {"status": "healthy", "gpu_available": status["gpu_available"]}

@app.get("/status")
def system_status():
    print("\n🔵 /status called", flush=True)
    status = manager.get_status()
    print(f"   Status: {status}", flush=True)
    return status

@app.post("/load-model")
def pre_warm_model(req: LoadModelRequest):
    print(f"\n🔵 /load-model called with: {req.model_id}", flush=True)
    with gpu_lock:
        try:
            print(f"   Loading model...", flush=True)
            manager.load_model(req.model_id)
            print(f"   ✅ Model loaded: {req.model_id}", flush=True)
            return {"status": "success", "message": f"{req.model_id} başarıyla yüklendi."}
        except Exception as e:
            print(f"   ❌ Error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-vram")
def clear_gpu_memory():
    print("\n🔵 /clear-vram called", flush=True)
    with gpu_lock:
        manager.clear_vram()
        return {"status": "success", "message": "VRAM tamamen temizlendi."}

@app.post("/predict")
def predict(req: PredictRequest):
    print(f"\n🔵 /predict called", flush=True)
    print(f"   Model: {req.model_id}", flush=True)
    print(f"   Prompt: {req.prompt[:50]}...", flush=True)
    print(f"   Input: {req.input_image_key}", flush=True)
    print(f"   Output: {req.output_image_key}", flush=True)
    
    with gpu_lock:
        try:
            input_path = None
            
            # 1. R2'den girdi çek
            if req.input_image_key:
                print(f"   📥 Downloading input from R2...", flush=True)
                input_path = f"/app/temp_in_{os.path.basename(req.input_image_key)}"
                success = download_input_from_r2(req.input_image_key, input_path)
                if not success:
                    raise HTTPException(status_code=400, detail="Girdi dosyası R2'den indirilemedi.")
            
            # 2. Modeli hazırla
            print(f"   📦 Loading model...", flush=True)
            pipeline = manager.load_model(req.model_id)

            # 3. Üretim işlemi
            print(f"   🎨 Running inference...", flush=True)
            output_path = f"/app/temp_out_{os.path.basename(req.output_image_key)}"
            
            model_type = manager.active_model_type
            print(f"   Model type: {model_type}", flush=True)

            if model_type == "image-generation":
                print(f"   Generating image...", flush=True)
                res = pipeline(prompt=req.prompt, num_inference_steps=20)
                res.images[0].save(output_path)
                
            elif model_type == "audio-generation":
                print(f"   Generating audio...", flush=True)
                import scipy.io.wavfile
                res = pipeline(req.prompt, num_inference_steps=10, audio_length_in_s=5.0)
                audio = res.audios[0]
                scipy.io.wavfile.write(output_path, rate=16000, data=audio)
                
            elif model_type == "music-generation":
                print(f"   Generating music...", flush=True)
                import scipy.io.wavfile
                import torch
                processor = pipeline["processor"]
                model = pipeline["model"]
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                inputs = processor(text=[req.prompt], padding=True, return_tensors="pt").to(dev)
                audio_values = model.generate(**inputs, max_new_tokens=256)
                sampling_rate = model.config.audio_encoder.sampling_rate
                scipy.io.wavfile.write(output_path, rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())
                
            elif model_type == "text-generation":
                print(f"   Generating text...", flush=True)
                res = pipeline(req.prompt, max_new_tokens=50)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(str(res[0]["generated_text"]))
                    
            elif model_type == "speech-recognition":
                print(f"   Recognizing speech...", flush=True)
                if not input_path:
                    raise Exception("Speech recognition işlemi için input_image_key (ses dosyası) sağlamalısınız!")
                res = pipeline(input_path)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(res["text"])
            else:
                raise Exception(f"Bilinmeyen model tipi: {model_type}")

            # 4. R2'ye Yükle
            print(f"   📤 Uploading output to R2...", flush=True)
            success = upload_output_to_r2(output_path, req.output_image_key)
            if not success:
                raise HTTPException(status_code=500, detail="Sonuç kaydedilemedi.")

            # 5. Temizlik
            if os.path.exists(output_path):
                os.remove(output_path)
            if input_path and os.path.exists(input_path):
                os.remove(input_path)

            print(f"   ✅ Inference complete!", flush=True)
            return {
                "status": "success",
                "message": "İşlem tamamlandı.",
                "output_key": req.output_image_key,
                "used_model": req.model_id
            }

        except Exception as e:
            print(f"   ❌ Predict error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

print("\n✅ API initialized and ready!", flush=True)
print("="*60, flush=True)
