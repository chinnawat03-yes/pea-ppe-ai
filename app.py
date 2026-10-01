from flask import Flask, render_template, request
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from PIL import Image, ImageDraw, ImageOps
from google.oauth2.service_account import Credentials
import gspread
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

app = Flask(__name__)

os.makedirs("static", exist_ok=True)


# =========================
# Roboflow
# =========================

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ["ROBOFLOW_API_KEY"]
).configure(
    InferenceConfiguration(
        api_key_transport="header"
    )
)


# =========================
# Google Sheets
# =========================

def get_google_sheet():

    credentials_info = json.loads(
        os.environ["GOOGLE_CREDENTIALS_JSON"]
    )

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets"
    ]

    credentials = Credentials.from_service_account_info(
        credentials_info,
        scopes=scopes
    )

    gc = gspread.authorize(credentials)

    sheet = gc.open_by_key(
        os.environ["GOOGLE_SHEET_ID"]
    ).sheet1

    return sheet


# =========================
# แปลงชื่อ Class จากโมเดล
# =========================

def normalize_class(class_name):

    class_map = {
        "safety_shoes": "shoes",
        "shoes": "shoes",
        "helmet": "helmet",
        "reflective_vest": "reflective_vest",
        "gloves": "gloves"
    }

    return class_map.get(
        class_name,
        class_name
    )


# =========================
# Main page
# =========================

@app.route("/", methods=["GET", "POST"])
def home():

    result_image = None
    detected_items = []
    status = None
    missing_text = None

    if request.method == "POST":

        file = request.files["image"]

        # =========================
        # เปิดรูป
        # =========================

        image = Image.open(file)

        # แก้ปัญหารูปจากมือถือหมุนผิดทิศ
        image = ImageOps.exif_transpose(image)

        # แปลงเป็น RGB
        image = image.convert("RGB")

        # บันทึกรูปต้นฉบับสำหรับส่ง Roboflow
        image.save("upload.jpg")

        # =========================
        # Roboflow Detection
        # =========================

        result = client.run_workflow(
            workspace_name="chinnawat-kf2et",
            workflow_id="pea_ppe_detection",
            images={
                "image": "upload.jpg"
            },
            use_cache=False
        )

        # เปิดรูปต้นฉบับเพื่อวาดกรอบ
        image = Image.open("upload.jpg").convert("RGB")

        draw = ImageDraw.Draw(image)

        predictions = result[0]["predictions"]["predictions"]

        detected_items = []

        # =========================
        # เตรียมผลตรวจ
        # =========================

        normalized_predictions = []

        for p in predictions:

            cls = normalize_class(
                p["class"]
            )

            normalized_predictions.append({
                "class": cls,
                "confidence": p["confidence"],
                "x": p["x"],
                "y": p["y"],
                "width": p["width"],
                "height": p["height"]
            })

            detected_items.append({
                "class": cls,
                "confidence": p["confidence"]
            })

        # =========================
        # สีของกรอบ
        # =========================

        colors = {
            "helmet": (255, 0, 0),
            "reflective_vest": (0, 102, 255),
            "gloves": (0, 180, 0),
            "shoes": (255, 140, 0)
        }

        # =========================
        # วาดกรอบ
        # =========================

        for p in normalized_predictions:

            x = p["x"]
            y = p["y"]
            w = p["width"]
            h = p["height"]

            left = x - w / 2
            top = y - h / 2
            right = x + w / 2
            bottom = y + h / 2

            color = colors.get(
                p["class"],
                (160, 0, 160)
            )

            # กรอบหนาและชัด
            draw.rectangle(
                [left, top, right, bottom],
                outline=color,
                width=10
            )

            # ข้อความบนกรอบ
            label = (
                f'{p["class"]} '
                f'{p["confidence"]:.0%}'
            )

            draw.text(
                (
                    left + 3,
                    max(3, top - 25)
                ),
                label,
                fill=color,
                stroke_width=2,
                stroke_fill=(255, 255, 255)
            )

        # =========================
        # สร้างชื่อไฟล์ใหม่ทุกครั้ง
        # =========================

        now = datetime.now(
            ZoneInfo("Asia/Bangkok")
        )

        result_filename = (
            f'result_'
            f'{now.strftime("%Y%m%d_%H%M%S_%f")}'
            f'.jpg'
        )

        result_path = os.path.join(
            "static",
            result_filename
        )

        # บันทึกรูปคุณภาพสูง
        image.save(
            result_path,
            quality=95
        )

        result_image = result_filename

        # =========================
        # สรุป PPE
        # =========================

        ppe = {
            "helmet": None,
            "reflective_vest": None,
            "gloves": None,
            "shoes": None
        }

        # เลือกค่าความมั่นใจสูงสุด
        for p in normalized_predictions:

            cls = p["class"]
            confidence = p["confidence"]

            if cls in ppe:

                if (
                    ppe[cls] is None
                    or confidence > ppe[cls]
                ):
                    ppe[cls] = confidence

        # =========================
        # ตรวจว่าครบหรือไม่
        # =========================

        total_found = sum(
            1
            for value in ppe.values()
            if value is not None
        )

        status = (
            "ครบ"
            if total_found == 4
            else "ไม่ครบ"
        )

        # =========================
        # ตรวจว่าขาดอะไร
        # =========================

        item_names = {
            "helmet": "Helmet",
            "reflective_vest": "Vest",
            "gloves": "Gloves",
            "shoes": "Shoes"
        }

        missing_items = []

        for item, confidence in ppe.items():

            if confidence is None:

                missing_items.append(
                    item_names[item]
                )

        if missing_items:

            missing_text = ", ".join(
                missing_items
            )

        else:

            missing_text = "ไม่มี"

        # =========================
        # บันทึก Google Sheets
        # =========================

        try:

            sheet = get_google_sheet()

            now = datetime.now(
                ZoneInfo("Asia/Bangkok")
            )

            sheet.append_row([

                now.strftime("%d/%m/%Y"),

                now.strftime("%H:%M:%S"),

                (
                    f'{ppe["helmet"]:.0%}'
                    if ppe["helmet"] is not None
                    else "-"
                ),

                (
                    f'{ppe["reflective_vest"]:.0%}'
                    if ppe["reflective_vest"] is not None
                    else "-"
                ),

                (
                    f'{ppe["gloves"]:.0%}'
                    if ppe["gloves"] is not None
                    else "-"
                ),

                (
                    f'{ppe["shoes"]:.0%}'
                    if ppe["shoes"] is not None
                    else "-"
                ),

                status
            ])

            print(
                "บันทึก Google Sheets สำเร็จ"
            )

        except Exception as e:

            print(
                "Google Sheets Error:",
                e
            )

    return render_template(
        "index.html",
        result_image=result_image,
        detected_items=detected_items,
        status=status,
        missing_text=missing_text
    )


# =========================
# History page
# =========================

@app.route("/history")
def history():

    try:

        sheet = get_google_sheet()

        records = sheet.get_all_records()

        records.reverse()

        return render_template(
            "history.html",
            records=records
        )

    except Exception as e:

        return (
            f"เกิดข้อผิดพลาดในการโหลดประวัติ: {e}"
        )


# =========================
# Run
# =========================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),
        debug=False
    )
