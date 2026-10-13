"""Usage: python test_client.py path/to/image.jpg [http://localhost:8000]"""
import sys
import requests

path = sys.argv[1]
base = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
with open(path, "rb") as f:
    r = requests.post(f"{base}/predict", files={"file": f})
print(r.status_code, r.json())
