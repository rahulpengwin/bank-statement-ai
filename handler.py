import runpod
import base64
import tempfile
import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger("bank_ai")

_model_loaded = False

def ensure_model():
    global _model_loaded
    if _model_loaded:
        return
    
    MODEL_PATH = os.getenv("MODEL_PATH")
    if not MODEL_PATH or not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at: {MODEL_PATH}. /runpod-volume contents: {os.listdir('/runpod-volume') if os.path.exists('/runpod-volume') else 'MISSING'}")
    
    log.info(f"Loading model from {MODEL_PATH}...")
    from app.parser import _load_model
    _load_model()
    _model_loaded = True
    log.info("Model loaded ✓")

def handler(job):
    job_input = job.get("input", {})
    pdf_b64   = job_input.get("pdf_base64")
    filename  = job_input.get("filename", "statement.pdf")

    if not pdf_b64:
        return {"error": "Missing 'pdf_base64' in input."}

    try:
        ensure_model()  # ← loads only once per warm worker
    except RuntimeError as e:
        return {"error": str(e)}

    t0 = time.time()
    pdf_bytes = base64.b64decode(pdf_b64)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        from app.parser import parse_header, parse_transactions
        from app.extractor import extract_text
        from app.categorizer import categorize_transactions

        raw_text = extract_text(tmp_path)
        t1 = time.time()

        acc_details  = parse_header(raw_text)
        transactions = parse_transactions(raw_text)
        t2 = time.time()

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
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

runpod.serverless.start({"handler": handler})