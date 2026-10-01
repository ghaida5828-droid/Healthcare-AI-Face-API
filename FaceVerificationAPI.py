import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from run import GetImageInfo, get_similarity


# =========================
# SETTINGS
# =========================

PATIENT_IMAGE = "test/patient.jpg"
THRESHOLD = 85


# =========================
# CREATE API
# =========================

app = FastAPI()

# يسمح لواجهة React بالتواصل مع Python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8443",
        "http://127.0.0.1:8443",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# LOAD PATIENT FACE
# =========================

print("Loading patient image...")

patient_image = cv2.imread(PATIENT_IMAGE)

if patient_image is None:
    raise RuntimeError(
        f"Could not load patient image: {PATIENT_IMAGE}"
    )

(
    patient_count,
    patient_boxes,
    patient_scores,
    patient_landmarks,
    patient_alignimgs,
    patient_features,
) = GetImageInfo(
    patient_image,
    1
)

if patient_count == 0:
    raise RuntimeError(
        "No face detected in patient image."
    )

patient_feature = patient_features[0]

print("Patient face loaded successfully.")


# =========================
# TEST ENDPOINT
# =========================

@app.get("/")
def root():
    return {
        "status": "Face Verification API is running"
    }


# =========================
# FACE VERIFICATION
# =========================

@app.post("/verify-face")
async def verify_face(file: UploadFile = File(...)):

    # اقرأ الصورة القادمة من React
    image_bytes = await file.read()

    np_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    live_image = cv2.imdecode(
        np_array,
        cv2.IMREAD_COLOR
    )

    if live_image is None:
        return {
            "success": False,
            "verified": False,
            "message": "Could not read image."
        }

    # اكتشاف الوجه واستخراج الـ feature
    (
        live_count,
        live_boxes,
        live_scores,
        live_landmarks,
        live_alignimgs,
        live_features,
    ) = GetImageInfo(
        live_image,
        1
    )

    # لا يوجد وجه
    if live_count == 0:
        return {
            "success": False,
            "verified": False,
            "message": "No face detected."
        }

    live_feature = live_features[0]

    # مقارنة الوجهين
    similarity = get_similarity(
        patient_feature,
        live_feature
    )

    similarity = float(similarity)

    verified = similarity >= THRESHOLD

    print(
        f"Similarity: {similarity:.2f}% | "
        f"Verified: {verified}"
    )

    return {
        "success": True,
        "verified": verified,
        "similarity": round(similarity, 2),
        "threshold": THRESHOLD
    }