import numpy as np
from numpy.linalg import norm
from datetime import datetime

from utils.database import supabase

THRESHOLD = 0.5


# -------------------------
# COSINE SIMILARITY
# -------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


# -------------------------
# ADD EMBEDDING
# -------------------------
def add_embedding(name, embedding):

    if embedding is None:
        return

    embedding = embedding / norm(embedding)

    try:
        supabase.table("staff_embeddings").insert({
            "name": name,
            "embedding": embedding.tolist()
        }).execute()

        print(f"✅ Added embedding for {name}")

    except Exception as e:
        print("❌ Insert Error:", e)


# -------------------------
# LOAD DATABASE
# -------------------------
def load_database():

    db = {}

    try:
        response = supabase.table("staff_embeddings").select("*").execute()

        for row in response.data:
            name = row["name"]
            emb = np.array(row["embedding"])

            if name not in db:
                db[name] = []

            db[name].append(emb)

    except Exception as e:
        print("❌ Load Error:", e)

    return db


# -------------------------
# RECOGNIZE
# -------------------------
def recognize(embedding, db):

    if embedding is None or len(db) == 0:
        return "Unidentified Person", 0.0

    embedding = embedding / norm(embedding)

    best_name = "Unidentified Person"
    best_score = -1

    for name, emb_list in db.items():

        scores = [cosine_similarity(embedding, e) for e in emb_list]
        max_score = max(scores)

        if max_score > best_score:
            best_score = max_score
            best_name = name

    if best_score < THRESHOLD:
        return "Unidentified Person", float(best_score)

    return best_name, float(best_score)


# -------------------------
# LOG ATTENDANCE
# -------------------------
def log_attendance(name):

    if name == "Unidentified Person":
        return

    try:
        supabase.table("attendance").insert({
            "name": name,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S")
        }).execute()

    except Exception as e:
        print("❌ Attendance Error:", e)