"""
Test FastAPI /ask endpoint.

Before running this test, start API in another terminal:

uvicorn api:app --reload

Then run:
python tests/test_api_ask.py
"""

import requests
import json


BASE_URL = "http://127.0.0.1:8000"


def main():
    payload = {
        "drug": "ibuprofen",
        "question": "What are the warnings for ibuprofen?",
        "domain": "fda",
        "top_k": 3,
    }

    resp = requests.post(f"{BASE_URL}/ask", json=payload, timeout=300)

    print("Status:", resp.status_code)

    try:
        data = resp.json()
    except Exception:
        print(resp.text)
        return

    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()