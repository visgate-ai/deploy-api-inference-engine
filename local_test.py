#!/usr/bin/env python3
"""
Lokal Modda AI Model Test Scripti
Bu script, modelleri Docker olmadan doğrudan GPU'da test eder.
"""

import os
import sys
import torch
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv("/workspace/deploy-api-inference-engine/app/.env")

from model_manager import manager

def test_model(model_id, prompt, description):
    """Modeli test et ve sonucu göster"""
    print(f"\n{'='*60}")
    print(f"📦 Test: {description}")
    print(f"🧠 Model: {model_id}")
    print(f"{'='*60}")
    
    try:
        # Model durumunu göster
        status = manager.get_status()
        print(f"VRAM Kullanımı: {status['vram_used_mb']:.2f} MB")
        
        # Modeli yükle
        print(f"⏳ Model yükleniyor...")
        pipeline = manager.load_model(model_id)
        print(f"✅ Model yüklendi!")
        
        # Durumu tekrar göster
        status = manager.get_status()
        print(f"VRAM Kullanımı: {status['vram_used_mb']:.2f} MB")
        
        # İnference
        print(f"⏳ Inference yapılıyor...")
        print(f"📝 Prompt: {prompt}")
        
        # Model tipine göre inference
        if manager.active_model_type == "text-generation":
            result = pipeline(prompt, max_new_tokens=50)
            print(f"\n📤 Sonuç:\n{result[0]['generated_text']}")
        elif manager.active_model_type == "image-generation":
            result = pipeline(prompt, num_inference_steps=20)
            output_path = "/workspace/deploy-api-inference-engine/test_output.png"
            result.images[0].save(output_path)
            print(f"\n✅ Görsel kaydedildi: {output_path}")
        else:
            print(f"Bu test scripti sadece text-generation ve image-generation modellerini destekler.")
        
        print(f"✅ Test başarılı!")
        return True
        
    except Exception as e:
        print(f"❌ Hata: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # VRAM'i temizle
        print(f"🧹 VRAM temizleniyor...")
        manager.clear_vram()

def main():
    print("🚀 Lokal AI Model Test Scripti")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Bulunamadı'}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    
    # Test edilecek modeller
    models_to_test = [
        {
            "id": "gpt2",
            "prompt": "Artificial intelligence will change the world by",
            "description": "GPT-2 Text Generation (En küçük model)"
        },
        {
            "id": "stabilityai/stable-diffusion-2-1",
            "prompt": "A beautiful sunset over mountains, digital art",
            "description": "Stable Diffusion 2.1 Image Generation"
        },
        {
            "id": "openai/whisper-small",
            "prompt": "test_audio.wav",
            "description": "Whisper Small (Speech Recognition)"
        }
    ]
    
    # Kullanıcıdan model seçimi
    print("\n📋 Test edilebilecek modeller:")
    for i, model in enumerate(models_to_test, 1):
        print(f"  {i}. {model['description']}")
    print(f"  0. Tüm modelleri test et")
    
    choice = input("\n▶️ Seçiminiz (0-3): ").strip()
    
    if choice == "0":
        selected = models_to_test
    else:
        idx = int(choice) - 1
        selected = [models_to_test[idx]]
    
    # Seçilen modelleri test et
    results = []
    for model in selected:
        success = test_model(model["id"], model["prompt"], model["description"])
        results.append((model["description"], success))
        
        # Her test arası sor
        if model != selected[-1]:
            cont = input("\n▶️ Devam et? (e/h): ").strip().lower()
            if cont != 'e':
                break
    
    # Özet
    print(f"\n{'='*60}")
    print("📊 Test Özeti:")
    print(f"{'='*60}")
    for desc, success in results:
        status = "✅ BAŞARILI" if success else "❌ BAŞARISIZ"
        print(f"  {status} - {desc}")

if __name__ == "__main__":
    main()
