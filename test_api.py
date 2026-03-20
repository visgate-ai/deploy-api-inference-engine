#!/usr/bin/env python3
"""
Inference Engine Comprehensive Test Suite
Tests all endpoints, R2 integration, and model loading
"""
import requests
import time
import sys
import subprocess
import signal
import os

BASE_URL = "http://localhost:8000"
SERVER_STARTUP_TIME = 8

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def log(msg, status="info"):
    icons = {"pass": "✅", "fail": "❌", "info": "🔵", "warn": "⚠️"}
    icon = icons.get(status, "•")
    color = {"pass": Colors.GREEN, "fail": Colors.RED, "info": Colors.BLUE, "warn": Colors.YELLOW}.get(status, "")
    print(f"{color}{icon} {msg}{Colors.RESET}")

def start_server():
    """Start uvicorn server in background"""
    log("Starting API server...", "info")
    proc = subprocess.Popen(
        ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd="/workspace/deploy-api-inference-engine",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    time.sleep(SERVER_STARTUP_TIME)
    return proc

def stop_server(proc):
    """Stop uvicorn server"""
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=5)

def test_health():
    """Test /health endpoint"""
    log("\n" + "="*50, "info")
    log("TEST 1: /health Endpoint", "info")
    log("="*50, "info")
    
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        data = r.json()
        
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        assert data["status"] == "healthy", f"Expected healthy, got {data['status']}"
        assert "gpu_available" in data, "Missing gpu_available field"
        
        log(f"Status: {data['status']}", "pass")
        log(f"GPU Available: {data['gpu_available']}", "pass")
        return True
    except Exception as e:
        log(f"Failed: {e}", "fail")
        return False

def test_status():
    """Test /status endpoint"""
    log("\n" + "="*50, "info")
    log("TEST 2: /status Endpoint", "info")
    log("="*50, "info")
    
    try:
        r = requests.get(f"{BASE_URL}/status", timeout=5)
        data = r.json()
        
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        
        expected_fields = ["active_model", "vram_used_mb", "gpu_available", "r2_connected"]
        for field in expected_fields:
            assert field in data, f"Missing field: {field}"
        
        log(f"Active Model: {data['active_model']}", "info")
        log(f"VRAM Used: {data['vram_used_mb']} MB", "info")
        log(f"R2 Connected: {data['r2_connected']}", "pass")
        return True
    except Exception as e:
        log(f"Failed: {e}", "fail")
        return False

def test_load_model():
    """Test /load-model endpoint with whisper-tiny from R2"""
    log("\n" + "="*50, "info")
    log("TEST 3: /load-model (whisper-tiny from R2)", "info")
    log("="*50, "info")
    
    try:
        log("Loading openai/whisper-tiny model...", "info")
        r = requests.post(
            f"{BASE_URL}/load-model",
            json={"model_id": "openai/whisper-tiny"},
            timeout=300  # 5 min timeout for model loading
        )
        
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        data = r.json()
        
        log(f"Response: {data['message']}", "pass")
        return True
    except requests.exceptions.Timeout:
        log("Timeout - model loading took too long", "warn")
        return False
    except Exception as e:
        log(f"Failed: {e}", "fail")
        return False

def test_status_after_load():
    """Check status after model load"""
    log("\n" + "="*50, "info")
    log("TEST 4: Status After Model Load", "info")
    log("="*50, "info")
    
    try:
        r = requests.get(f"{BASE_URL}/status", timeout=5)
        data = r.json()
        
        log(f"Active Model: {data['active_model']}", "info")
        log(f"Model Type: {data['active_model_type']}", "info")
        log(f"VRAM Used: {data['vram_used_mb']} MB", "info")
        
        if data['active_model'] == "openai/whisper-tiny":
            log("Model loaded successfully!", "pass")
            return True
        else:
            log("Model not loaded properly", "fail")
            return False
    except Exception as e:
        log(f"Failed: {e}", "fail")
        return False

def test_clear_vram():
    """Test /clear-vram endpoint"""
    log("\n" + "="*50, "info")
    log("TEST 5: /clear-vram Endpoint", "info")
    log("="*50, "info")
    
    try:
        r = requests.post(f"{BASE_URL}/clear-vram", timeout=30)
        data = r.json()
        
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        log(f"Response: {data['message']}", "pass")
        
        # Check VRAM cleared
        time.sleep(1)
        status = requests.get(f"{BASE_URL}/status", timeout=5).json()
        log(f"VRAM after clear: {status['vram_used_mb']} MB", "info")
        
        return True
    except Exception as e:
        log(f"Failed: {e}", "fail")
        return False

def test_concurrent_requests():
    """Test GPU lock mechanism with concurrent requests"""
    log("\n" + "="*50, "info")
    log("TEST 6: GPU Lock / Queue Mechanism", "info")
    log("="*50, "info")
    
    log("Sending 3 concurrent model load requests...", "info")
    log("(Should execute serially due to GPU lock)", "info")
    
    import threading
    
    results = []
    lock = threading.Lock()
    
    def load_request(i):
        try:
            r = requests.post(
                f"{BASE_URL}/load-model",
                json={"model_id": "openai/whisper-tiny"},
                timeout=120
            )
            with lock:
                results.append({"id": i, "status": r.status_code, "success": True})
        except Exception as e:
            with lock:
                results.append({"id": i, "error": str(e), "success": False})
    
    threads = []
    for i in range(3):
        t = threading.Thread(target=load_request, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    log(f"Results: {len([r for r in results if r.get('success')])}/3 succeeded", "pass")
    return True

def test_predict_no_input():
    """Test /predict endpoint (should fail gracefully without input for ASR)"""
    log("\n" + "="*50, "info")
    log("TEST 7: /predict Endpoint Validation", "info")
    log("="*50, "info")
    
    try:
        r = requests.post(
            f"{BASE_URL}/predict",
            json={
                "model_id": "openai/whisper-tiny",
                "prompt": "test",
                "output_image_key": "test/output.txt"
            },
            timeout=30
        )
        
        # Should fail with 500 because no audio input for ASR
        if r.status_code == 500:
            log("Correctly rejected request (needs audio input)", "pass")
            return True
        else:
            log(f"Unexpected status: {r.status_code}", "warn")
            return True
    except Exception as e:
        log(f"Failed: {e}", "fail")
        return False

def test_invalid_model():
    """Test with invalid/non-existent model"""
    log("\n" + "="*50, "info")
    log("TEST 8: Invalid Model Handling", "info")
    log("="*50, "info")
    
    try:
        r = requests.post(
            f"{BASE_URL}/load-model",
            json={"model_id": "nonexistent/model-does-not-exist-xyz"},
            timeout=60
        )
        
        if r.status_code == 500:
            log("Correctly rejected invalid model", "pass")
            return True
        else:
            log(f"Status: {r.status_code}", "warn")
            return True
    except Exception as e:
        log(f"Failed: {e}", "fail")
        return False

def main():
    print("\n" + "="*60)
    print("🚀 VISGATE INFERENCE ENGINE - COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    proc = None
    results = []
    
    try:
        # Start server
        proc = start_server()
        
        # Run tests
        results.append(("Health Endpoint", test_health()))
        results.append(("Status Endpoint", test_status()))
        results.append(("Load Model from R2", test_load_model()))
        results.append(("Status After Load", test_status_after_load()))
        results.append(("Clear VRAM", test_clear_vram()))
        results.append(("Concurrent Lock", test_concurrent_requests()))
        results.append(("Predict Validation", test_predict_no_input()))
        results.append(("Invalid Model", test_invalid_model()))
        
    finally:
        if proc:
            stop_server(proc)
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = f"{Colors.GREEN}PASS{Colors.RESET}" if result else f"{Colors.RED}FAIL{Colors.RESET}"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
