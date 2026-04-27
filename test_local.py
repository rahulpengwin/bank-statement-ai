# test_local.py
import base64
import json

# Load a real bank statement PDF
with open("test_statement.pdf", "rb") as f:
    pdf_b64 = base64.b64encode(f.read()).decode()

# Simulate a RunPod job payload
job = {
    "id": "test-local-001",
    "input": {
        "pdf_base64": pdf_b64,
        "filename": "test_statement.pdf"
    }
}

# Import and call handler directly
from handler import handler
result = handler(job)
print(json.dumps(result, indent=2))