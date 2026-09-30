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
# Main page
# =========================

@app.route("/", methods=["GET", "POST"])
def home():

    result_image = None
    detected_items = []

    if request.method == "POST":

        file = request.files["image"]

        # แก้ปัญหารูปจากมือถือหมุนผิดทิศ
        image = Image.open(file)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
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
            use_cache=True
        )

        image = Image.open("upload.jpg")
        draw = ImageDraw.Draw(image)

        predictions = result[0]["predictions"]["predictions"]

        detected_items = []

        for p in predictions:
            detected_items.append({
                "class": p["class"],
                "confidence": p["confidence"]
            })

        # =========================
        # สีของกรอบ
        # =========================

        colors = {
            "helmet": "red",
            "reflective_vest": "blue",
            "gloves": "green",
            "shoes": "orange"
        }

        # =========================
        # วาดกรอบ
        # =========================

        for p in predictions:

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
                "purple"
            )

            draw.rectangle(
                [left, top, right, bottom],
                outline=color,
                width=5
            )

            draw.text(
                (
                    left,
                    max(0, top - 20)
                ),
                f'{p["class"]} {p["confidence"]:.0%}',
                fill=color
            )

        # =========================
        # บันทึกรูปผลลัพธ์
        # =========================

        image.save(
            "static/static_result.jpg"
        )

        result_image = "static_result.jpg"


        # =========================
        # สรุป PPE
        # =========================

        ppe = {
            "helmet": None,
            "reflective_vest": None,
            "gloves": None,
            "shoes": None
        }

        # ถ้ามีหลายกล่องของชนิดเดียวกัน
        # จะเลือกค่าความมั่นใจสูงสุด
        for p in predictions:

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
            1 for value in ppe.values()
            if value is not None
        )

        status = (
            "ครบ"
            if total_found == 4
            else "ไม่ครบ"
        )


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

            print("บันทึก Google Sheets สำเร็จ")

        except Exception as e:

            print(
                "Google Sheets Error:",
                e
            )


    return render_template(
        "index.html",
        result_image=result_image,
        detected_items=detected_items
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
