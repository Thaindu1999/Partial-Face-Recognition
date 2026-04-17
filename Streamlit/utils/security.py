from cryptography.fernet import Fernet
import os

KEY_PATH = "utils/secret.key"


# -------------------------
# GENERATE KEY (RUN ONCE)
# -------------------------
def generate_key():
    key = Fernet.generate_key()

    with open(KEY_PATH, "wb") as f:
        f.write(key)


# -------------------------
# LOAD KEY
# -------------------------
def load_key():
    if not os.path.exists(KEY_PATH):
        generate_key()

    with open(KEY_PATH, "rb") as f:
        return f.read()


# -------------------------
# GET CIPHER
# -------------------------
def get_cipher():
    key = load_key()
    return Fernet(key)