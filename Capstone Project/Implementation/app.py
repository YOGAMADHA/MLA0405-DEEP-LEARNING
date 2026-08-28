from flask import Flask, render_template, request, send_file
import tensorflow as tf
import numpy as np
import cv2
import os
import uuid
import json
from datetime import datetime

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4


app = Flask(__name__)

# ============================================================
# FOLDERS
# ============================================================

UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
HEATMAP_FOLDER = "static/heatmaps"
REPORT_FOLDER = "static/reports"

for folder in [
    UPLOAD_FOLDER,
    PROCESSED_FOLDER,
    HEATMAP_FOLDER,
    REPORT_FOLDER
]:
    os.makedirs(folder, exist_ok=True)


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "signature_forgery_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# HISTORY
# ============================================================

history = []


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_for_model(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    resized = cv2.resize(
        gray,
        (128, 128)
    )

    normalized = resized.astype(
        "float32"
    ) / 255.0

    input_image = np.expand_dims(
        normalized,
        axis=-1
    )

    input_image = np.expand_dims(
        input_image,
        axis=0
    )

    return gray, resized, input_image


# ============================================================
# IMAGE QUALITY
# ============================================================

def analyze_quality(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    sharpness = float(
        laplacian.var()
    )

    brightness = float(
        np.mean(gray)
    )

    height, width = gray.shape

    if sharpness < 50:
        quality = "LOW"
    elif sharpness < 150:
        quality = "MEDIUM"
    else:
        quality = "GOOD"

    return {
        "quality": quality,
        "sharpness": round(sharpness, 2),
        "brightness": round(brightness, 2),
        "image_width": width,
        "image_height": height
    }


# ============================================================
# SIGNATURE EXTRACTION
# ============================================================

def extract_signature(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    _, threshold = cv2.threshold(
        gray,
        200,
        255,
        cv2.THRESH_BINARY_INV
    )

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return image

    points = np.vstack(contours)

    x, y, w, h = cv2.boundingRect(
        points
    )

    margin = 10

    x1 = max(
        0,
        x - margin
    )

    y1 = max(
        0,
        y - margin
    )

    x2 = min(
        image.shape[1],
        x + w + margin
    )

    y2 = min(
        image.shape[0],
        y + h + margin
    )

    return image[
        y1:y2,
        x1:x2
    ]


# ============================================================
# SIGNATURE FEATURES
# ============================================================

def analyze_signature(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    _, binary = cv2.threshold(
        gray,
        200,
        255,
        cv2.THRESH_BINARY_INV
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:

        return {
            "signature_width": 0,
            "signature_height": 0,
            "aspect_ratio": 0,
            "stroke_density": 0,
            "components": 0
        }

    points = np.vstack(contours)

    x, y, w, h = cv2.boundingRect(
        points
    )

    aspect_ratio = (
        w / h
        if h != 0
        else 0
    )

    black_pixels = np.sum(
        binary > 0
    )

    total_pixels = binary.size

    density = (
        black_pixels /
        total_pixels
    ) * 100

    return {
        "signature_width": int(w),
        "signature_height": int(h),
        "aspect_ratio": round(
            float(aspect_ratio),
            2
        ),
        "stroke_density": float(
            round(
                density,
                2
            )
        ),
        "components": int(
            len(contours)
        )
    }


# ============================================================
# CNN PREDICTION
# ============================================================

def predict_signature(image):

    gray, resized, input_image = (
        preprocess_for_model(image)
    )

    prediction = model.predict(
        input_image,
        verbose=0
    )[0][0]

    prediction = float(
        prediction
    )

    genuine_probability = prediction
    forged_probability = 1.0 - prediction

    if genuine_probability >= 0.5:

        result = "GENUINE"

        confidence = (
            genuine_probability * 100
        )

    else:

        result = "FORGED"

        confidence = (
            forged_probability * 100
        )

    return {
        "result": result,
        "confidence": round(
            confidence,
            2
        ),
        "genuine_probability": round(
            genuine_probability * 100,
            2
        ),
        "forged_probability": round(
            forged_probability * 100,
            2
        ),
        "input_image": input_image,
        "resized": resized
    }


# ============================================================
# FORENSIC SCORECARD
# ============================================================

def calculate_scorecard(
    quality_data,
    signature_data,
    prediction_data
):

    sharpness = quality_data["sharpness"]

    if sharpness >= 300:
        quality_score = 95
    elif sharpness >= 150:
        quality_score = 85
    elif sharpness >= 50:
        quality_score = 70
    else:
        quality_score = 45

    density = signature_data[
        "stroke_density"
    ]

    if 3 <= density <= 35:
        stroke_score = 90
    elif 1 <= density <= 50:
        stroke_score = 75
    else:
        stroke_score = 55

    aspect = signature_data[
        "aspect_ratio"
    ]

    if 2 <= aspect <= 7:
        shape_score = 90
    elif 1 <= aspect <= 10:
        shape_score = 75
    else:
        shape_score = 55

    components = signature_data[
        "components"
    ]

    if components <= 5:
        structure_score = 90
    elif components <= 10:
        structure_score = 75
    else:
        structure_score = 55

    cnn_score = prediction_data[
        "confidence"
    ]

    overall = (
        quality_score * 0.20
        + stroke_score * 0.20
        + shape_score * 0.20
        + structure_score * 0.20
        + cnn_score * 0.20
    )

    return {
        "image_quality": int(
            quality_score
        ),
        "stroke_consistency": int(
            stroke_score
        ),
        "shape_consistency": int(
            shape_score
        ),
        "structural_stability": int(
            structure_score
        ),
        "cnn_confidence": int(
            cnn_score
        ),
        "overall_score": round(
            overall,
            1
        )
    }


# ============================================================
# AI EXPLANATION
# ============================================================

def generate_explanation(
    result,
    confidence,
    quality,
    sharpness,
    density,
    aspect_ratio,
    components
):

    explanation = []

    if confidence >= 80:

        if result == "GENUINE":

            explanation.append(
                "The CNN strongly favors the genuine signature class."
            )

        else:

            explanation.append(
                "The CNN strongly favors the forged signature class."
            )

    elif confidence < 70:

        explanation.append(
            "The prediction is close to the decision boundary, so manual verification is recommended."
        )

    else:

        explanation.append(
            "The CNN prediction has moderate confidence."
        )

    if sharpness >= 150:

        explanation.append(
            "The image has sufficient sharpness for analysis."
        )

    else:

        explanation.append(
            "Image sharpness is relatively low and may affect reliability."
        )

    if 3 <= density <= 35:

        explanation.append(
            "Stroke density is within a reasonable signature range."
        )

    else:

        explanation.append(
            "The stroke density is outside the expected range."
        )

    if 2 <= aspect_ratio <= 7:

        explanation.append(
            "The signature has a reasonable horizontal shape ratio."
        )

    else:

        explanation.append(
            "The signature proportions are unusual."
        )

    if components <= 5:

        explanation.append(
            "The signature contains a relatively stable connected structure."
        )

    else:

        explanation.append(
            "Multiple disconnected components were detected."
        )

    return explanation


# ============================================================
# RISK
# ============================================================

def calculate_risk(
    result,
    confidence
):

    if confidence < 70:
        return "UNCERTAIN"

    if result == "FORGED" and confidence >= 80:
        return "HIGH"

    if confidence >= 70:
        return "MEDIUM"

    return "LOW"


# ============================================================
# PROCESSED IMAGE
# ============================================================

def create_processed_image(
    image,
    filename
):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    _, binary = cv2.threshold(
        gray,
        200,
        255,
        cv2.THRESH_BINARY
    )

    path = os.path.join(
        PROCESSED_FOLDER,
        filename
    )

    cv2.imwrite(
        path,
        binary
    )

    return "/" + path.replace(
        "\\",
        "/"
    )


# ============================================================
# GRAD-CAM HEATMAP
# ============================================================

def generate_heatmap(
    image,
    filename
):

    try:

        # Find last convolution layer
        conv_layers = []

        for layer in model.layers:

            if len(
                layer.output.shape
            ) == 4:

                conv_layers.append(
                    layer
                )

        if not conv_layers:

            return None

        last_conv = conv_layers[-1]

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                last_conv.output,
                model.output
            ]
        )

        _, _, input_image = (
            preprocess_for_model(image)
        )

        with tf.GradientTape() as tape:

            conv_output, predictions = (
                grad_model(
                    input_image
                )
            )

            loss = predictions[:, 0]

        grads = tape.gradient(
            loss,
            conv_output
        )

        pooled_grads = tf.reduce_mean(
            grads,
            axis=(0, 1, 2)
        )

        conv_output = conv_output[0]

        heatmap = conv_output @ (
            pooled_grads[..., tf.newaxis]
        )

        heatmap = tf.squeeze(
            heatmap
        )

        heatmap = tf.maximum(
            heatmap,
            0
        )

        max_value = tf.reduce_max(
            heatmap
        )

        heatmap = heatmap / (
            max_value + 1e-8
        )

        heatmap = heatmap.numpy()

        heatmap = cv2.resize(
            heatmap,
            (
                image.shape[1],
                image.shape[0]
            )
        )

        heatmap = np.uint8(
            255 * heatmap
        )

        heatmap_color = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(
            image,
            0.55,
            heatmap_color,
            0.45,
            0
        )

        path = os.path.join(
            HEATMAP_FOLDER,
            filename
        )

        cv2.imwrite(
            path,
            overlay
        )

        return "/" + path.replace(
            "\\",
            "/"
        )

    except Exception as e:

        print(
            "Heatmap generation error:",
            e
        )

        return None


# ============================================================
# COMPARE TWO SIGNATURES
# ============================================================

def compare_signatures(
    reference,
    questioned
):

    ref_gray = cv2.cvtColor(
        reference,
        cv2.COLOR_BGR2GRAY
    )

    test_gray = cv2.cvtColor(
        questioned,
        cv2.COLOR_BGR2GRAY
    )

    ref_gray = cv2.resize(
        ref_gray,
        (256, 256)
    )

    test_gray = cv2.resize(
        test_gray,
        (256, 256)
    )

    # Histogram similarity
    hist1 = cv2.calcHist(
        [ref_gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    hist2 = cv2.calcHist(
        [test_gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    cv2.normalize(
        hist1,
        hist1
    )

    cv2.normalize(
        hist2,
        hist2
    )

    histogram_similarity = cv2.compareHist(
        hist1,
        hist2,
        cv2.HISTCMP_CORREL
    )

    histogram_similarity = max(
        0,
        min(
            1,
            histogram_similarity
        )
    )

    # Structural similarity using normalized correlation
    correlation = cv2.matchTemplate(
        test_gray,
        ref_gray,
        cv2.TM_CCOEFF_NORMED
    )[0][0]

    correlation = max(
        0,
        min(
            1,
            float(correlation)
        )
    )

    # Feature similarity
    ref_features = analyze_signature(
        reference
    )

    test_features = analyze_signature(
        questioned
    )

    def similarity(a, b):

        if max(a, b) == 0:
            return 0

        difference = abs(a - b)

        return max(
            0,
            100 - (
                difference /
                max(a, b)
                * 100
            )
        )

    shape_similarity = similarity(
        ref_features[
            "aspect_ratio"
        ],
        test_features[
            "aspect_ratio"
        ]
    )

    density_similarity = similarity(
        ref_features[
            "stroke_density"
        ],
        test_features[
            "stroke_density"
        ]
    )

    overall = (
        histogram_similarity * 30
        + correlation * 30
        + shape_similarity / 100 * 20
        + density_similarity / 100 * 20
    )

    overall = round(
        overall,
        2
    )

    if overall >= 80:

        comparison_result = (
            "HIGH SIMILARITY"
        )

    elif overall >= 60:

        comparison_result = (
            "MODERATE SIMILARITY"
        )

    else:

        comparison_result = (
            "LOW SIMILARITY"
        )

    return {
        "similarity": overall,
        "result": comparison_result,
        "shape_similarity": round(
            shape_similarity,
            2
        ),
        "stroke_similarity": round(
            density_similarity,
            2
        ),
        "structure_similarity": round(
            correlation * 100,
            2
        )
    }


# ============================================================
# PDF REPORT
# ============================================================

def generate_report(data):

    report_id = str(
        uuid.uuid4()
    )[:8]

    filename = (
        "signature_report_"
        + report_id
        + ".pdf"
    )

    filepath = os.path.join(
        REPORT_FOLDER,
        filename
    )

    pdf = canvas.Canvas(
        filepath,
        pagesize=A4
    )

    page_width, page_height = A4

    y = page_height - 55

    pdf.setFont(
        "Helvetica-Bold",
        20
    )

    pdf.drawCentredString(
        page_width / 2,
        y,
        "AI SIGNATURE FORENSIC REPORT"
    )

    y -= 45

    pdf.setFont(
        "Helvetica",
        11
    )

    lines = [

        "Date: " + data["date"],

        "Result: " + data["result"],

        "Confidence: "
        + str(data["confidence"])
        + "%",

        "Genuine Probability: "
        + str(data["genuine_probability"])
        + "%",

        "Forged Probability: "
        + str(data["forged_probability"])
        + "%",

        "Risk: " + data["risk"],

        "",

        "FORENSIC SCORECARD",

        "Overall Score: "
        + str(data["overall_score"]),

        "Image Quality: "
        + str(data["image_quality"]),

        "Stroke Consistency: "
        + str(data["stroke_consistency"]),

        "Shape Consistency: "
        + str(data["shape_consistency"]),

        "Structural Stability: "
        + str(data["structural_stability"]),

        "CNN Confidence: "
        + str(data["cnn_confidence"]),

        "",

        "IMAGE ANALYSIS",

        "Image Quality: "
        + data["quality"],

        "Sharpness: "
        + str(data["sharpness"]),

        "Brightness: "
        + str(data["brightness"]),

        "Image Size: "
        + str(data["image_width"])
        + " x "
        + str(data["image_height"]),

        "",

        "SIGNATURE FEATURES",

        "Width: "
        + str(data["signature_width"])
        + " px",

        "Height: "
        + str(data["signature_height"])
        + " px",

        "Aspect Ratio: "
        + str(data["aspect_ratio"]),

        "Stroke Density: "
        + str(data["stroke_density"])
        + "%",

        "Connected Components: "
        + str(data["components"])
    ]

    for line in lines:

        if line == "":

            y -= 15

            continue

        if (
            line in [
                "FORENSIC SCORECARD",
                "IMAGE ANALYSIS",
                "SIGNATURE FEATURES"
            ]
        ):

            pdf.setFont(
                "Helvetica-Bold",
                13
            )

        else:

            pdf.setFont(
                "Helvetica",
                10
            )

        pdf.drawString(
            60,
            y,
            line
        )

        y -= 21

        if y < 60:

            pdf.showPage()

            y = page_height - 60

    pdf.setFont(
        "Helvetica-Oblique",
        9
    )

    pdf.drawString(
        60,
        35,
        "Generated by AI Signature Forgery Detection System"
    )

    pdf.save()

    return filename


# ============================================================
# HOME
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def index():

    result = None
    error = None

    image_path = None
    processed_path = None
    heatmap_path = None

    report_file = None

    comparison = None

    reference_path = None
    questioned_path = None

    # ========================================================
    # SINGLE SIGNATURE ANALYSIS
    # ========================================================

    if request.method == "POST" and request.form.get(
        "action"
    ) == "analyze":

        if "signature" not in request.files:

            error = (
                "Please select a signature image."
            )

        else:

            file = request.files[
                "signature"
            ]

            if file.filename == "":

                error = (
                    "Please select an image."
                )

            else:

                extension = os.path.splitext(
                    file.filename
                )[1].lower()

                if extension not in [
                    ".png",
                    ".jpg",
                    ".jpeg"
                ]:

                    error = (
                        "Only PNG, JPG and JPEG "
                        "images are supported."
                    )

                else:

                    filename = (
                        str(
                            uuid.uuid4()
                        )
                        + extension
                    )

                    filepath = os.path.join(
                        UPLOAD_FOLDER,
                        filename
                    )

                    file.save(filepath)

                    image = cv2.imread(
                        filepath
                    )

                    if image is None:

                        error = (
                            "Unable to read the image."
                        )

                    else:

                        quality_data = (
                            analyze_quality(
                                image
                            )
                        )

                        signature = (
                            extract_signature(
                                image
                            )
                        )

                        signature_data = (
                            analyze_signature(
                                signature
                            )
                        )

                        prediction_data = (
                            predict_signature(
                                signature
                            )
                        )

                        risk = calculate_risk(
                            prediction_data[
                                "result"
                            ],
                            prediction_data[
                                "confidence"
                            ]
                        )

                        scorecard = (
                            calculate_scorecard(
                                quality_data,
                                signature_data,
                                prediction_data
                            )
                        )

                        explanations = (
                            generate_explanation(
                                prediction_data[
                                    "result"
                                ],
                                prediction_data[
                                    "confidence"
                                ],
                                quality_data[
                                    "quality"
                                ],
                                quality_data[
                                    "sharpness"
                                ],
                                signature_data[
                                    "stroke_density"
                                ],
                                signature_data[
                                    "aspect_ratio"
                                ],
                                signature_data[
                                    "components"
                                ]
                            )
                        )

                        result = {

                            **quality_data,

                            **signature_data,

                            **prediction_data,

                            **scorecard,

                            "risk": risk,

                            "explanation":
                                explanations,

                            "date":
                                datetime.now().strftime(
                                    "%d-%m-%Y %H:%M:%S"
                                )
                        }

                        # Remove numpy objects
                        result.pop(
                            "input_image",
                            None
                        )

                        result.pop(
                            "resized",
                            None
                        )

                        image_path = (
                            "/"
                            + filepath.replace(
                                "\\",
                                "/"
                            )
                        )

                        processed_filename = (
                            "processed_"
                            + filename
                        )

                        processed_path = (
                            create_processed_image(
                                signature,
                                processed_filename
                            )
                        )

                        heatmap_filename = (
                            "heatmap_"
                            + filename
                        )

                        heatmap_path = (
                            generate_heatmap(
                                signature,
                                heatmap_filename
                            )
                        )

                        history.insert(
                            0,
                            {
                                "date":
                                    result["date"],
                                "result":
                                    result["result"],
                                "confidence":
                                    result["confidence"],
                                "risk":
                                    result["risk"]
                            }
                        )

                        report_file = (
                            generate_report(
                                result
                            )
                        )

    # ========================================================
    # COMPARE TWO SIGNATURES
    # ========================================================

    if request.method == "POST" and request.form.get(
        "action"
    ) == "compare":

        reference = request.files.get(
            "reference"
        )

        questioned = request.files.get(
            "questioned"
        )

        if (
            reference is None
            or questioned is None
            or reference.filename == ""
            or questioned.filename == ""
        ):

            error = (
                "Please upload both reference "
                "and questioned signatures."
            )

        else:

            ref_ext = os.path.splitext(
                reference.filename
            )[1].lower()

            test_ext = os.path.splitext(
                questioned.filename
            )[1].lower()

            allowed = [
                ".png",
                ".jpg",
                ".jpeg"
            ]

            if (
                ref_ext not in allowed
                or test_ext not in allowed
            ):

                error = (
                    "Only PNG, JPG and JPEG "
                    "images are supported."
                )

            else:

                ref_filename = (
                    "reference_"
                    + str(uuid.uuid4())
                    + ref_ext
                )

                test_filename = (
                    "questioned_"
                    + str(uuid.uuid4())
                    + test_ext
                )

                ref_path = os.path.join(
                    UPLOAD_FOLDER,
                    ref_filename
                )

                test_path = os.path.join(
                    UPLOAD_FOLDER,
                    test_filename
                )

                reference.save(
                    ref_path
                )

                questioned.save(
                    test_path
                )

                reference_image = cv2.imread(
                    ref_path
                )

                questioned_image = cv2.imread(
                    test_path
                )

                if (
                    reference_image is None
                    or questioned_image is None
                ):

                    error = (
                        "Unable to read one "
                        "of the images."
                    )

                else:

                    comparison = (
                        compare_signatures(
                            reference_image,
                            questioned_image
                        )
                    )

                    reference_path = (
                        "/"
                        + ref_path.replace(
                            "\\",
                            "/"
                        )
                    )

                    questioned_path = (
                        "/"
                        + test_path.replace(
                            "\\",
                            "/"
                        )
                    )

    return render_template(
        "index.html",
        result=result,
        error=error,
        image_path=image_path,
        processed_path=processed_path,
        heatmap_path=heatmap_path,
        report_file=report_file,
        history=history,
        comparison=comparison,
        reference_path=reference_path,
        questioned_path=questioned_path
    )


# ============================================================
# PDF DOWNLOAD
# ============================================================

@app.route(
    "/download/<filename>"
)
def download_report(filename):

    filepath = os.path.join(
        REPORT_FOLDER,
        filename
    )

    if not os.path.exists(filepath):

        return "Report not found", 404

    return send_file(
        filepath,
        as_attachment=True
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )