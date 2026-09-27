import os
import io
import base64
import numpy as np

from flask import Flask, render_template, request, jsonify

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# KONFIGURASI FLASK
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "mobilenetv2_yospan_best.keras"
)

LABEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "labels.txt"
)


# ============================================================
# LOAD MODEL MOBILENETV2
# ============================================================

print("==============================================")
print("MEMUAT MODEL MOBILENETV2 YOSPAN")
print("==============================================")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model tidak ditemukan: {MODEL_PATH}"
    )

model = load_model(
    MODEL_PATH,
    compile=False
)

print("Model berhasil dimuat.")
print("Model:", MODEL_PATH)


# ============================================================
# LOAD LABEL
# ============================================================

if os.path.exists(LABEL_PATH):

    with open(LABEL_PATH, "r", encoding="utf-8") as file:
        labels = [
            line.strip()
            for line in file
            if line.strip()
        ]

else:

    labels = [
        "bukan_yospan",
        "gale_gale",
        "jef",
        "pacul_tiga",
        "pancar",
        "seka"
    ]

print()
print("Label:", labels)
print("Jumlah kelas:", len(labels))

print("==============================================")
print("MODEL SIAP DIGUNAKAN")
print("==============================================")


# ============================================================
# NAMA GERAKAN UNTUK DITAMPILKAN DI WEBSITE
# ============================================================

DISPLAY_NAMES = {
    "bukan_yospan": "Bukan Yospan",

    "gale_gale": "Gale-gale",
    "gale-gale": "Gale-gale",
    "gale gale": "Gale-gale",

    "jef": "Jef",

    "pacul_tiga": "Pacul Tiga",
    "pacul-tiga": "Pacul Tiga",
    "pacul tiga": "Pacul Tiga",

    "pancar": "Pancar",

    "seka": "Seka"
}


# ============================================================
# FUNGSI MENGUBAH LABEL MENJADI NAMA YANG RAPI
# ============================================================

def format_label(label):
    key = label.strip().lower()

    return DISPLAY_NAMES.get(
        key,
        label.replace("_", " ").title()
    )


# ============================================================
# FUNGSI PREPROCESSING GAMBAR
#
# Gambar diproses langsung dari memory.
# TIDAK disimpan ke static/uploads.
#
# Input:
#   image_bytes
#
# Output:
#   Array dengan bentuk (1, 224, 224, 3)
# ============================================================

def preprocess_image(image_bytes):

    image = load_img(
        io.BytesIO(image_bytes),
        target_size=(224, 224)
    )

    image_array = img_to_array(
        image
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    image_array = preprocess_input(
        image_array
    )

    return image_array


# ============================================================
# FUNGSI PREDIKSI
# ============================================================

def predict_image(image_bytes):

    image_array = preprocess_image(
        image_bytes
    )

    prediction = model.predict(
        image_array,
        verbose=0
    )

    # Pastikan output model berbentuk:
    # (jumlah_data, jumlah_kelas)

    if prediction.ndim != 2:
        raise ValueError(
            "Output model tidak memiliki format yang sesuai."
        )

    if prediction.shape[0] < 1:
        raise ValueError(
            "Model tidak menghasilkan hasil prediksi."
        )

    if prediction.shape[1] != len(labels):
        raise ValueError(
            f"Jumlah output model ({prediction.shape[1]}) "
            f"tidak sesuai dengan jumlah label ({len(labels)})."
        )

    predicted_index = int(
        np.argmax(prediction[0])
    )

    predicted_label = labels[
        predicted_index
    ]

    confidence = float(
        prediction[0][predicted_index]
    )

    is_yospan = (
        predicted_label.strip().lower()
        != "bukan_yospan"
    )

    return {
        "success": True,
        "status": "success",
        "label": format_label(
            predicted_label
        ),
        "confidence": confidence,
        "is_yospan": is_yospan
    }


# ============================================================
# FUNGSI MEMBUAT DATA URL GAMBAR
#
# Gambar TIDAK disimpan ke server.
# Gambar dikembalikan sebagai Data URL.
#
# Ini aman digunakan pada filesystem Vercel yang read-only.
# ============================================================

def create_image_data_url(image_bytes, mimetype):

    encoded_image = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    return f"data:{mimetype};base64,{encoded_image}"


# ============================================================
# HALAMAN BERANDA
# ============================================================

@app.route("/")
def index():
    return render_template(
        "index.html"
    )


# ============================================================
# HALAMAN TENTANG
# ============================================================

@app.route("/tentang")
def tentang():
    return render_template(
        "tentang.html"
    )


# ============================================================
# HALAMAN KLASIFIKASI
# ============================================================

@app.route("/klasifikasi")
def klasifikasi():
    return render_template(
        "klasifikasi.html"
    )


# ============================================================
# API / PROSES KLASIFIKASI
#
# PENTING:
# Tidak ada proses penyimpanan file ke filesystem.
# Gambar dibaca langsung ke memory.
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    # --------------------------------------------------------
    # Periksa apakah file dikirim
    # --------------------------------------------------------

    if "image" not in request.files:

        return jsonify({
            "success": False,
            "status": "error",
            "message": "Gambar belum dipilih."
        }), 400

    file = request.files["image"]

    # --------------------------------------------------------
    # Periksa nama file
    # --------------------------------------------------------

    if not file.filename:

        return jsonify({
            "success": False,
            "status": "error",
            "message": "Gambar belum dipilih."
        }), 400

    try:

        # ----------------------------------------------------
        # Baca gambar langsung ke memory.
        #
        # Jangan menyimpan file ke filesystem server.
        # Vercel menggunakan filesystem deployment
        # yang bersifat read-only.
        # ----------------------------------------------------

        image_bytes = file.read()

        if not image_bytes:

            return jsonify({
                "success": False,
                "status": "error",
                "message": "File gambar kosong."
            }), 400

        # ----------------------------------------------------
        # Prediksi
        # ----------------------------------------------------

        result = predict_image(
            image_bytes
        )

        # ----------------------------------------------------
        # Buat Data URL gambar untuk frontend
        # ----------------------------------------------------

        mimetype = (
            file.mimetype
            or "image/jpeg"
        )

        image_data_url = create_image_data_url(
            image_bytes,
            mimetype
        )

        # ----------------------------------------------------
        # Kirim hasil ke frontend
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "status": "success",

            "label": result["label"],

            "confidence": result["confidence"],

            "is_yospan": result["is_yospan"],

            "image": image_data_url

        }), 200

    except Exception as error:

        # ----------------------------------------------------
        # Error lengkap dapat dilihat di Vercel Runtime Logs
        # ----------------------------------------------------

        print("==============================================")
        print("ERROR PREDIKSI")
        print("==============================================")
        print("Tipe error:", type(error).__name__)
        print("Pesan:", str(error))
        print("==============================================")

        return jsonify({

            "success": False,

            "status": "error",

            "message":
                "Terjadi kesalahan saat melakukan klasifikasi."

        }), 500


# ============================================================
# MENJALANKAN SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )
