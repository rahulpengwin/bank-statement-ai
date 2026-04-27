"""
handler.py — RunPod Serverless entrypoint for Bank Statement AI
Models are loaded ONCE at worker startup (outside handler) for speed.
Each job receives a base64-encoded PDF and returns structured JSON.
"""

import runpod
import base64
import tempfile
import os
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger("bank_ai")

# ── Pre-load models at container startup (once per warm worker) ────────────────
log.info("Cold start: loading llama-cpp model + GLM-OCR...")
from app.parser import _load_model, parse_header, parse_transactions
from app.extractor import extract_text
from app.categorizer import categorize_transactions

_load_model()
log.info("Models ready ✓")

# ── Handler ────────────────────────────────────────────────────────────────────
def handler(job):
    job_input = job.get("input", {})
    pdf_b64   = job_input.get("pdf_base64")
    filename  = job_input.get("filename", "statement.pdf")

    if not pdf_b64:
        return {"error": "Missing 'pdf_base64' in input. Encode your PDF as base64."}

    t0 = time.time()
    pdf_bytes = base64.b64decode(pdf_b64)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        # Stage 1: PDF extraction
        raw_text = extract_text(tmp_path)
        t1 = time.time()
        log.info(f"Extracted {len(raw_text):,} chars in {t1-t0:.3f}s")

        # Stage 2: LLM parsing (header + transactions)
        acc_details  = parse_header(raw_text)
        transactions = parse_transactions(raw_text)
        t2 = time.time()
        log.info(f"LLM parsed {len(transactions)} txns in {t2-t1:.3f}s")

        # Stage 3: Categorization
        transactions = categorize_transactions(transactions)

        debits  = [float(t["debit"])  for t in transactions if t.get("debit")  and str(t["debit"]).strip()]
        credits = [float(t["credit"]) for t in transactions if t.get("credit") and str(t["credit"]).strip()]

        return {
            "account_details": acc_details,
            "transactions":    transactions,
            "summary": {
                "total_transactions": len(transactions),
                "total_debit":        round(sum(debits), 2),
                "total_credit":       round(sum(credits), 2),
                "net_flow":           round(sum(credits) - sum(debits), 2),
            },
            "source_file": filename,
            "used_image_mode": False,
            "timings": {
                "pdf_extraction_s": round(t1 - t0, 3),
                "llm_inference_s":  round(t2 - t1, 3),
                "total_s":          round(time.time() - t0, 3),
            }
        }
    except Exception as e:
        log.error(f"Handler error: {e}", exc_info=True)
        return {"error": str(e)}
    finally:
        os.unlink(tmp_path)

runpod.serverless.start({"handler": handler})