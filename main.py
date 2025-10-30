#!/usr/bin/env python3
# =======================================================
# 🧠 Skin Cancer Classification using Roboflow Inference API
# Author: Muhammadiyev Bahrombek
# Date: 2025-10-27
# =======================================================

from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import os
import sys

# ================= CONFIGURATION =================
API_URL = "https://detect.roboflow.com"
API_KEY = "Kz1uRQNYQfiMGbhGigCh"   # <-- bu sizning haqiqiy API keyingiz
MODEL_ID = "classification-igqvf/1"          # <-- classification model
OUTPUT_DIR = "detections"
# ==================================================

# Rangli terminal chiqishlari uchun
class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    END = "\033[0m"

# Log funksiyasi
def log_info(msg): print(f"{Colors.BLUE}[INFO]{Colors.END} {msg}")
def log_success(msg): print(f"{Colors.GREEN}[SUCCESS]{Colors.END} {msg}")
def log_warn(msg): print(f"{Colors.YELLOW}[WARNING]{Colors.END} {msg}")
def log_error(msg): print(f"{Colors.RED}[ERROR]{Colors.END} {msg}")

# Roboflow client
CLIENT = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)

def detect_skin_cancer(image_path):
    if not os.path.exists(image_path):
        log_error(f"Fayl topilmadi: {image_path}")
        return

    try:
        log_info("Tasvir tahlil qilinmoqda...")

        # Model orqali natija olish
        result = CLIENT.infer(image_path, model_id=MODEL_ID)

        # Classification modelda natija shunday keladi:
        # {"predictions": {"mel": {"confidence": 0.611, "class_id": 4}, ...}}
        predictions = result.get("predictions", {})

        if not predictions:
            log_success("✅ Teri saratoni aniqlanmadi.")
            return

        # Eng yuqori ishonchlilikdagi sinfni topamiz
        label, info = max(predictions.items(), key=lambda x: x[1]["confidence"])
        confidence = info["confidence"] * 100

        # Natijani chiqarish
        log_warn(f"🧠 Aniqlangan sinf: {label} ({confidence:.1f}%)")

        if confidence < 50:
            log_success("✅ AI tahliliga ko‘ra teri saratoni aniqlanmadi (past ishonchlilik).")
        else:
            log_warn("🚨 Teri saratoni ehtimoli mavjud!")

        # Rasmga yozib qo‘yish
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        text = f"{label} ({confidence:.1f}%)"
        draw.text((20, 20), text, fill="red")

        # Natija papkasini yaratish
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"classified_{timestamp}.jpg"
        output_path = os.path.join(OUTPUT_DIR, filename)
        image.save(output_path)

        log_success(f"💾 Natija saqlandi: {output_path}")

    except Exception as e:
        log_error(f"Xatolik yuz berdi: {e}")


if __name__ == "__main__":
    print(f"{Colors.BLUE}\n=== 🧠 Teri saratonini tasniflash v1.0 ==={Colors.END}")

    if len(sys.argv) < 2:
        log_warn("Foydalanish: python skin_cancer_classifier.py <rasm_manzili>")
        sys.exit(0)

    image_path = sys.argv[1]
    detect_skin_cancer(image_path)
