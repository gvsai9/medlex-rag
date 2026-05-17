"""
Test streaming FastAPI /ask/stream endpoint.

Before running:
python -m uvicorn api:app --reload

Then in another terminal:
python tests/test_api_stream.py
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

    with requests.post(
        f"{BASE_URL}/ask/stream",
        json=payload,
        stream=True,
        timeout=300,
    ) as resp:
        print("Status:", resp.status_code)
        resp.raise_for_status()

        print("\n--- STREAM START ---\n")

        full_answer = ""

        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue

            event = json.loads(line)

            if event["type"] == "meta":
                print("Citations:")
                for c in event["citations"]:
                    print(
                        f"[{c['id']}] {c['doc_title']} | page={c['page_num']} "
                        f"| drug={c['drug']} | score={c['score']}"
                    )
                print("\nAnswer:\n")

            elif event["type"] == "token":
                print(event["content"], end="", flush=True)
                full_answer += event["content"]

            elif event["type"] == "done":
                print("\n\n--- STREAM DONE ---")
                print("Model:", event["model"])
                print("Latency:", event["latency_ms"], "ms")

        print("\n\nFull answer length:", len(full_answer))


if __name__ == "__main__":
    main()