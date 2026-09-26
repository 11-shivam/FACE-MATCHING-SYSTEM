from deepface import DeepFace
import database_connection
import cv2
import json


# -------------------------------
# Generate face embedding
# -------------------------------

def generate_embedding(image_path):

    result = DeepFace.represent(
        img_path=image_path,
        model_name="ArcFace",
        detector_backend="retinaface"
    )

    return result[0]["embedding"]
