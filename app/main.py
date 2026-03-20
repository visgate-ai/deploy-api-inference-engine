import os
import threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# .env dosyasındaki ayarları yükle
load_dotenv()

from .storage import download_input_from_r2, upload_output_to_r2
from .model_manager import manager

app = FastAPI(title="Visgate AI Sunucusuz GPU Worker")

# --- KİLİT MEKANİZMASI (GPU KORUMASI) ---
# Bu kilit sayesinde aynı anda kaç istek gelirse gelsin, GPU'ya sadece 1 tanesi girebilir.
# Diğerleri API'de bekler, işi biten çıkınca sıradaki girer.
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
    # Kilit dışında! Model çalışırken bile anında cevap verir
    return {"status": "healthy", "gpu_available": manager.get_status()["gpu_available"]}

@app.get("/status")
def system_status():
    # Kilit dışında! Anlık VRAM durumunu her zaman okuyabilirsiniz.
    return manager.get_status()

@app.post("/load-model")
def pre_warm_model(req: LoadModelRequest):
    # Model yüklemek de VRAM kullanır, bu yüzden kilide tabidir.
    with gpu_lock:
        try:
            manager.load_model(req.model_id)
            return {"status": "success", "message": f"{req.model_id} başarıyla yüklendi."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-vram")
def clear_gpu_memory():
    with gpu_lock:
        manager.clear_vram()
        return {"status": "success", "message": "VRAM tamamen temizlendi."}

@app.post("/predict")
def predict(req: PredictRequest):
    # ANA İŞLEM MERKEZİ: İçeri giren kilitlenir, dışarıdakiler bekler.
    with gpu_lock:
        try:
            input_path = None
            
            # 1. R2'den girdi çek (Opsiyonel)
            if req.input_image_key:
                input_path = f"/app/temp_in_{os.path.basename(req.input_image_key)}"
                success = download_input_from_r2(req.input_image_key, input_path)
                if not success:
                    raise HTTPException(status_code=400, detail="Girdi dosyası R2'den indirilemedi.")
            
            # 2. Modeli hazırla
            pipeline = manager.load_model(req.model_id)

            # 3. Üretim işlemi
            print(f"Üretim başlatıldı. Prompt: '{req.prompt}'")
            output_path = f"/app/temp_out_{os.path.basename(req.output_image_key)}"
            
            model_type = manager.active_model_type

            if model_type == "image-generation":
                res = pipeline(prompt=req.prompt, num_inference_steps=20)
                res.images[0].save(output_path)
                
            elif model_type == "audio-generation": # AudioLDM
                import scipy.io.wavfile
                res = pipeline(req.prompt, num_inference_steps=10, audio_length_in_s=5.0)
                audio = res.audios[0]
                scipy.io.wavfile.write(output_path, rate=16000, data=audio)
                
            elif model_type == "music-generation": # MusicGen
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
                res = pipeline(req.prompt, max_new_tokens=50)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(str(res[0]["generated_text"]))
                    
            elif model_type == "speech-recognition":
                if not input_path:
                    raise Exception("Speech recognition işlemi için input_image_key (ses dosyası) sağlamalısınız!")
                res = pipeline(input_path)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(res["text"])
            else:
                raise Exception("Bilinmeyen model tipi.")

            # 4. R2'ye Yükle
            success = upload_output_to_r2(output_path, req.output_image_key)
            if not success:
                raise HTTPException(status_code=500, detail="Sonuç kaydedilemedi.")

            # 5. Temizlik
            if os.path.exists(output_path):
                os.remove(output_path)
            if input_path and os.path.exists(input_path):
                os.remove(input_path)

            return {
                "status": "success",
                "message": "İşlem tamamlandı.",
                "output_key": req.output_image_key,
                "used_model": req.model_id
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
