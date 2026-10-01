import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from run import GetImageInfo, get_similarity


# =========================
# SETTINGS
# =========================

THRESHOLD = 85


# =========================
# CREATE API
# =========================

app = FastAPI()


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
async def verify_face(
    live_file: UploadFile = File(...),
    reference_file: UploadFile = File(...)
):

    # =========================
    # READ REFERENCE IMAGE
    # =========================

    reference_bytes = await reference_file.read()

    reference_array = np.frombuffer(
        reference_bytes,
        np.uint8
    )

    reference_image = cv2.imdecode(
        reference_array,
        cv2.IMREAD_COLOR
    )

    if reference_image is None:
        return {
            "success": False,
            "verified": False,
            "message": "Could not read reference image."
        }


    # =========================
    # READ LIVE IMAGE
    # =========================

    live_bytes = await live_file.read()

    live_array = np.frombuffer(
        live_bytes,
        np.uint8
    )

    live_image = cv2.imdecode(
        live_array,
        cv2.IMREAD_COLOR
    )

    if live_image is None:
        return {
            "success": False,
            "verified": False,
            "message": "Could not read live image."
        }


    # =========================
    # PROCESS REFERENCE FACE
    # =========================

    (
        reference_count,
        reference_boxes,
        reference_scores,
        reference_landmarks,
        reference_alignimgs,
        reference_features,
    ) = GetImageInfo(
        reference_image,
        1
    )

    if reference_count == 0:
        return {
            "success": False,
            "verified": False,
            "message": "No face detected in reference image."
        }

    reference_feature = reference_features[0]


    # =========================
    # PROCESS LIVE FACE
    # =========================

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

    if live_count == 0:
        return {
            "success": False,
            "verified": False,
            "message": "No face detected in live image."
        }

    live_feature = live_features[0]


    # =========================
    # COMPARE FACES
    # =========================

    similarity = get_similarity(
        reference_feature,
        live_feature
    )

    similarity = float(similarity)

    verified = similarity >= THRESHOLD


    print(
        f"Similarity: {similarity:.2f}% | "
        f"Verified: {verified}"
    )


    # =========================
    # RESPONSE
    # =========================

    return {
        "success": True,
        "verified": verified,
        "similarity": round(similarity, 2),
        "threshold": THRESHOLD
    }
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001
    )