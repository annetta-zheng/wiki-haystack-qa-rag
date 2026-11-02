#!/usr/bin/env python3
import json
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os

def generate_key_from_password(password: str, salt: bytes = None):
    """Generate a Fernet key from a password using PBKDF2."""
    if salt is None:
        salt = os.urandom(16)
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key, salt

def encrypt_qa_set(qa_data: list, password: str) -> dict:
    """Encrypt a Q&A dataset with a password."""
    key, salt = generate_key_from_password(password)
    fernet = Fernet(key)
    
    json_data = json.dumps(qa_data, ensure_ascii=False)
    encrypted_data = fernet.encrypt(json_data.encode())
    
    return {
        'salt': base64.b64encode(salt).decode('utf-8'),
        'data': base64.b64encode(encrypted_data).decode('utf-8')
    }

def decrypt_qa_set(encrypted_data: dict, password: str) -> list:
    """Decrypt a Q&A dataset with a password."""
    salt = base64.b64decode(encrypted_data['salt'])
    data = base64.b64decode(encrypted_data['data'])
    
    key, _ = generate_key_from_password(password, salt)
    fernet = Fernet(key)
    
    try:
        decrypted_data = fernet.decrypt(data)
        return json.loads(decrypted_data.decode())
    except Exception as e:
        raise ValueError(f"Failed to decrypt: Invalid password or corrupted data - {e}")

def save_encrypted_qa(qa_data: list, password: str, output_file: str):
    """Encrypt and save Q&A data to a file."""
    encrypted = encrypt_qa_set(qa_data, password)
    with open(output_file, 'w') as f:
        json.dump(encrypted, f, indent=2)

def load_and_decrypt_qa(input_file: str, password: str) -> list:
    """Load and decrypt Q&A data from a file."""
    with open(input_file, 'r') as f:
        encrypted = json.load(f)
    return decrypt_qa_set(encrypted, password)

if __name__ == "__main__":
    test_qa = [
        {"question": "Test question?", "answer": "Test answer"},
        {"question": "Another test?", "answer": "Another answer"}
    ]
    
    password = "test_password_123"
    
    encrypted = encrypt_qa_set(test_qa, password)
    print("Encrypted successfully")
    
    decrypted = decrypt_qa_set(encrypted, password)
    print("Decrypted successfully")
    print("Match:", test_qa == decrypted)