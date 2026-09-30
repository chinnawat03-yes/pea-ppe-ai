from flask import Flask, render_template, request
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from PIL import Image, ImageDraw
import os

app = Flask(__name__)

os.makedirs("static", exist_ok=True)

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ["ROBOFLOW_API_KEY"]
).configure(
    InferenceConfiguration(
        api_key_transport="header"
    )
)
@app.route("/", methods=["GET", "POST"])
def home():
    result_image = None

    if request.method == "POST":
        file = request.files["image"]

        file.save("upload.jpg")

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

        colors = {
            "helmet": "red",
            "reflective_vest": "blue",
            "gloves": "green",
            "shoes": "orange"
        }

        for p in predictions:
            x = p["x"]
            y = p["y"]
            w = p["width"]
            h = p["height"]

            left = x - w / 2
            top = y - h / 2
            right = x + w / 2
            bottom = y + h / 2

            color = colors.get(p["class"], "purple")

            draw.rectangle(
                [left, top, right, bottom],
                outline=color,
                width=5
            )

            draw.text(
                (left, max(0, top - 20)),
                f'{p["class"]} {p["confidence"]:.0%}',
                fill=color
            )

        image.save("static/static_result.jpg")
        result_image = "static_result.jpg"

    return render_template(
        "index.html",
        result_image=result_image
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
