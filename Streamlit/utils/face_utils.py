import torch
import cv2
import numpy as np
from PIL import Image


def detect_face(frame, face_cascade):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(50, 50)
    )

    if len(faces) == 0:
        return None, None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Expand box
    margin = 0.25
    x1 = int(x - w * margin)
    y1 = int(y - h * margin)
    x2 = int(x + w + w * margin)
    y2 = int(y + h + h * margin)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    face_crop = frame[y1:y2, x1:x2]
    img = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    return img, (x1, y1, x2 - x1, y2 - y1)


def get_embedding(model, img, transform):

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        emb = model(img_tensor)
        emb = emb / emb.norm(p=2, dim=1, keepdim=True)

    return emb.squeeze().cpu().numpy()