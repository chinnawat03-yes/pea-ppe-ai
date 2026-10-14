```python
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

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ["ROBOFLOW_API_KEY"]
).configure(
    InferenceConfiguration(api_key_transport="header")
)


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


def normalize_class(class_name):
    class_map = {
        "safety_shoes": "shoes",
        "shoes": "shoes",
        "helmet": "helmet",
        "reflective_vest": "reflective_vest",
        "gloves": "gloves"
    }

    return class_map.get(class_name, class_name)


@app.route("/", methods=["GET", "POST"])
def home():
    result_image = None
    detected_items = []
    status = None
    missing_text = None

    if request.method == "POST":
        employee_name = request.form.get(
            "employee_name", ""
        ).strip()

        file = request.files["image"]

        image = Image.open(file)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image.save("upload.jpg")

        result = client.run_workflow(
            workspace_name="chinnawat-kf2et",
            workflow_id="pea_ppe_detection",
            images={"image": "upload.jpg"},
            use_cache=False
        )

        image = Image.open("upload.jpg").convert("RGB")
        draw = ImageDraw.Draw(image)

        predictions = result[0]["predictions"]["predictions"]

        detected_items = []
        normalized_predictions = []

        for p in predictions:
            cls = normalize_class(p["class"])

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

        colors = {
            "helmet": (255, 0, 0),
            "reflective_vest": (0, 102, 255),
            "gloves": (0, 180, 0),
            "shoes": (255, 140, 0)
        }

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

            draw.rectangle(
                [left, top, right, bottom],
                outline=color,
                width=10
            )

            # แสดงชื่อคลาสเท่านั้น ไม่แสดงเปอร์เซ็นต์
            label = p["class"]

            draw.text(
                (left + 3, max(3, top - 25)),
                label,
                fill=color,
                stroke_width=2,
                stroke_fill=(255, 255, 255)
            )

        now = datetime.now(ZoneInfo("Asia/Bangkok"))

        result_filename = (
            f'result_{now.strftime("%Y%m%d_%H%M%S_%f")}.jpg'
        )

        result_path = os.path.join(
            "static",
            result_filename
        )

        image.save(result_path, quality=95)
        result_image = result_filename

        ppe = {
            "helmet": None,
            "reflective_vest": None,
            "gloves": None,
            "shoes": None
        }

        for p in normalized_predictions:
            cls = p["class"]
            confidence = p["confidence"]

            if cls in ppe:
                if (
                    ppe[cls] is None
                    or confidence > ppe[cls]
                ):
                    ppe[cls] = confidence

        total_found = sum(
            1 for value in ppe.values()
            if value is not None
        )

        status = "ครบ" if total_found == 4 else "ไม่ครบ"

        item_names = {
            "helmet": "Helmet",
            "reflective_vest": "Vest",
            "gloves": "Gloves",
            "shoes": "Shoes"
        }

        missing_items = []

        for item, confidence in ppe.items():
            if confidence is None:
                missing_items.append(item_names[item])

        missing_text = (
            ", ".join(missing_items)
            if missing_items
            else "ไม่มี"
        )

        try:
            sheet = get_google_sheet()
            now = datetime.now(ZoneInfo("Asia/Bangkok"))

            sheet.append_row([
                now.strftime("%d/%m/%Y"),
                now.strftime("%H:%M:%S"),
                employee_name,
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
            print("Google Sheets Error:", e)

    return render_template(
        "index.html",
        result_image=result_image,
        detected_items=detected_items,
        status=status,
        missing_text=missing_text
    )


@app.route("/history")
def history():
    try:
        sheet = get_google_sheet()
        records = sheet.get_all_records()
        records.reverse()

        total_count = len(records)

        complete_count = sum(
            1 for row in records
            if str(row.get("ผลรวม", "")).strip() == "ครบ"
        )

        incomplete_count = sum(
            1 for row in records
            if str(row.get("ผลรวม", "")).strip() == "ไม่ครบ"
        )

        complete_rate = (
            complete_count / total_count
            if total_count > 0
            else 0
        )

        daily_data = {}

        for row in records:
            date = str(row.get("วันที่", "")).strip()
            status_value = str(
                row.get("ผลรวม", "")
            ).strip()

            if not date:
                continue

            if date not in daily_data:
                daily_data[date] = {
                    "ครบ": 0,
                    "ไม่ครบ": 0
                }

            if status_value == "ครบ":
                daily_data[date]["ครบ"] += 1
            elif status_value == "ไม่ครบ":
                daily_data[date]["ไม่ครบ"] += 1

        def parse_date(value):
            try:
                return datetime.strptime(
                    value,
                    "%d/%m/%Y"
                )
            except ValueError:
                return datetime.min

        sorted_dates = sorted(
            daily_data.keys(),
            key=parse_date
        )

        chart_dates = sorted_dates[-14:]

        chart_complete = [
            daily_data[date]["ครบ"]
            for date in chart_dates
        ]

        chart_incomplete = [
            daily_data[date]["ไม่ครบ"]
            for date in chart_dates
        ]

        return render_template(
            "history.html",
            records=records,
            total_count=total_count,
            complete_count=complete_count,
            incomplete_count=incomplete_count,
            complete_rate=complete_rate,
            chart_dates=chart_dates,
            chart_complete=chart_complete,
            chart_incomplete=chart_incomplete
        )

    except Exception as e:
        print("History Error:", e)
        return f"เกิดข้อผิดพลาดในการโหลดประวัติ: {e}"


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
```
