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
# Tidak disimpan ke static/uploads.
#
# Input:
#   image_bytes
#
# Proses:
#   1. Baca gambar dari memory
#   2. Resize menjadi 224 x 224
#   3. Ubah menjadi NumPy array
#   4. Tambahkan batch dimension
#   5. preprocess_input MobileNetV2
#
# Output:
#   Array dengan bentuk (1, 224, 224, 3)
# ============================================================

def preprocess_image(image_bytes):

    # --------------------------------------------------------
    # Baca gambar langsung dari memory
    # --------------------------------------------------------

    image = load_img(
        io.BytesIO(image_bytes),
        target_size=(224, 224)
    )

    # --------------------------------------------------------
    # Ubah gambar menjadi NumPy array
    # --------------------------------------------------------

    image_array = img_to_array(
        image
    )

    # --------------------------------------------------------
    # Tambahkan dimensi batch
    #
    # Dari:
    #   (224, 224, 3)
    #
    # Menjadi:
    #   (1, 224, 224, 3)
    # --------------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # --------------------------------------------------------
    # Preprocessing khusus MobileNetV2
    # --------------------------------------------------------

    image_array = preprocess_input(
        image_array
    )

    return image_array


# ============================================================
# FUNGSI PREDIKSI
# ============================================================

def predict_image(image_bytes):

    # --------------------------------------------------------
    # PREPROCESSING GAMBAR
    # --------------------------------------------------------

    image_array = preprocess_image(
        image_bytes
    )

    # --------------------------------------------------------
    # PREDIKSI MODEL
    # --------------------------------------------------------

    prediction = model.predict(
        image_array,
        verbose=0
    )

    # --------------------------------------------------------
    # Pastikan output model berbentuk benar
    # --------------------------------------------------------

    if prediction.ndim != 2:

        raise ValueError(
            "Output model tidak memiliki format yang sesuai."
        )

    # --------------------------------------------------------
    # Pastikan jumlah output model sesuai jumlah label
    # --------------------------------------------------------

    if prediction.shape[1] != len(labels):

        raise ValueError(
            f"Jumlah output model ({prediction.shape[1]}) "
            f"tidak sesuai dengan jumlah label ({len(labels)})."
        )

    # --------------------------------------------------------
    # Ambil indeks kelas dengan probabilitas tertinggi
    # --------------------------------------------------------

    predicted_index = int(
        np.argmax(prediction[0])
    )

    # --------------------------------------------------------
    # Ambil nama label
    # --------------------------------------------------------

    predicted_label = labels[
        predicted_index
    ]

    # --------------------------------------------------------
    # Ambil confidence
    # --------------------------------------------------------

    confidence = float(
        prediction[0][predicted_index]
    )

    # --------------------------------------------------------
    # Tentukan apakah hasil merupakan gerakan Yospan
    # --------------------------------------------------------

    is_yospan = (
        predicted_label.lower() != "bukan_yospan"
    )

    # --------------------------------------------------------
    # HASIL
    # --------------------------------------------------------

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
# Gambar tidak disimpan ke server.
# Gambar dikembalikan dalam bentuk data URL
# agar frontend tetap dapat menampilkan gambar.
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

    if file.filename == "":

        return jsonify({
            "success": False,
            "status": "error",
            "message": "Gambar belum dipilih."
        }), 400

    try:

        # ----------------------------------------------------
        # Baca file langsung ke memory
        #
        # TIDAK menggunakan file.save()
        # karena filesystem Vercel bersifat read-only.
        # ----------------------------------------------------

        image_bytes = file.read()

        # ----------------------------------------------------
        # Pastikan file tidak kosong
        # ----------------------------------------------------

        if not image_bytes:

            return jsonify({
                "success": False,
                "status": "error",
                "message": "File gambar kosong."
            }), 400

        # ----------------------------------------------------
        # Jalankan prediksi
        # ----------------------------------------------------

        result = predict_image(
            image_bytes
        )

        # ----------------------------------------------------
        # Buat data URL gambar
        #
        # Ini menggantikan URL:
        # /static/uploads/nama_file.png
        #
        # sehingga tidak perlu menyimpan file
        # ke filesystem server.
        # ----------------------------------------------------

        mimetype = file.mimetype or "image/jpeg"

        image_data_url = create_image_data_url(
            image_bytes,
            mimetype
        )

        # ----------------------------------------------------
        # Jika prediksi gagal
        # ----------------------------------------------------

        if not result["success"]:

            return jsonify({

                "success": False,

                "status": result.get(
                    "status",
                    "error"
                ),

                "message": result.get(
                    "message",
                    "Gagal melakukan klasifikasi."
                ),

                "image": image_data_url

            }), 500

        # ----------------------------------------------------
        # Jika prediksi berhasil
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "status": "success",

            "label": result["label"],

            "confidence": result["confidence"],

            "is_yospan": result["is_yospan"],

            "image": image_data_url

        })

    except Exception as error:

        # ----------------------------------------------------
        # Tampilkan error lengkap di Vercel Runtime Logs
        # ----------------------------------------------------

        print("==============================================")
        print("ERROR PREDIKSI")
        print("==============================================")
        print(type(error).__name__)
        print(str(error))
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
