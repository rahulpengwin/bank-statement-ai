"""
handler.py — RunPod Serverless entrypoint for Bank Statement AI
Models are loaded ONCE at worker startup (outside handler) for speed.
Each job receives a base64-encoded PDF and returns structured JSON.
"""

import runpod
import base64
import os
import tempfile
import time
import logging
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("bank_ai.handler")

# ── Load models ONCE at worker startup ────────────────────────────────────────
# RunPod keeps the worker alive between jobs (within idle timeout window),
# so this only runs once per worker lifecycle — not per request.
log.info("=" * 60)
log.info("WORKER STARTUP — Loading models...")

from app.parser import load_model, parse_header, parse_transactions, _qwen, _glm_model
from app.extractor import extract_text, extract_header_text, anonymize_sensitive, restore_sensitive
from app.categorizer import categorize_transactions

load_model()
log.info("Models loaded ✓ Worker ready.")
log.info("=" * 60)


# ── Handler ────────────────────────────────────────────────────────────────────
def handler(job):
    """
    Input schema:
    {
        "input": {
            "pdf_base64": "<base64 encoded PDF bytes>",
            "filename": "statement.pdf"   (optional)
        }
    }

    Output: full ParseResponse dict (account_details, transactions, summary, timings)
    """
    job_input = job.get("input", {})

    # ── Input validation ───────────────────────────────────────────────────────
    pdf_b64 = job_input.get("pdf_base64")
    if not pdf_b64:
        return {"error": "Missing required field: 'pdf_base64'"}

    filename = job_input.get("filename", "statement.pdf")
    if not filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files are supported"}

    # ── Decode PDF ─────────────────────────────────────────────────────────────
    try:
        pdf_bytes = base64.b64decode(pdf_b64)
    except Exception as e:
        return {"error": f"Invalid base64 PDF data: {str(e)}"}

    log.info(f"Job received: {filename} ({len(pdf_bytes)/1024:.1f} KB)")

    # ── Write to temp file ─────────────────────────────────────────────────────
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(pdf_bytes)
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()

    try:
        t0 = time.time()

        # Progress update — client can poll and see this
        runpod.serverless.progress_update(job, "Step 1/4: Extracting PDF text...")

        raw_text = extract_text(tmp_path)
        header_raw_text = extract_header_text(tmp_path)
        t1 = time.time()
        log.info(f"Extracted {len(raw_text):,} chars / header {len(header_raw_text):,} chars")

        # Anonymise PAN and mobile numbers before sending to LLM
        anon_text, restore_map = anonymize_sensitive(raw_text)
        anon_header_text, restore_map2 = anonymize_sensitive(header_raw_text)
        restore_map.update(restore_map2)

        runpod.serverless.progress_update(job, "Step 2/4: Parsing account header with Qwen...")
        acc_details = parse_header(anon_header_text)
        acc_details = restore_sensitive(acc_details, restore_map)

        runpod.serverless.progress_update(job, "Step 3/4: Running GLM-OCR + Qwen on transactions...")
        transactions = parse_transactions(anon_text, pdf_path=tmp_path)
        transactions = [restore_sensitive(t, restore_map) for t in transactions]
        t2 = time.time()

        runpod.serverless.progress_update(job, "Step 4/4: Categorizing transactions...")
        transactions = categorize_transactions(transactions)
        cats = Counter(t.get("category", "Other") for t in transactions)
        t3 = time.time()

        # ── Build summary ──────────────────────────────────────────────────────
        debits, credits = [], []
        for txn in transactions:
            try:
                if txn.get("debit") and str(txn["debit"]).strip():
                    debits.append(float(txn["debit"]))
                if txn.get("credit") and str(txn["credit"]).strip():
                    credits.append(float(txn["credit"]))
            except (ValueError, TypeError):
                pass

        total_s = round(time.time() - t0, 3)
        log.info(f"COMPLETE — {len(transactions)} txns in {total_s}s")

        return {
            "account_details": acc_details,
            "transactions": transactions,
            "summary": {
                "total_transactions": len(transactions),
                "total_debit": round(sum(debits), 2),
                "total_credit": round(sum(credits), 2),
                "net_flow": round(sum(credits) - sum(debits), 2),
                "categories": dict(cats),
            },
            "source_file": filename,
            "used_image_mode": True,
            "timings": {
                "pdf_extraction_s": round(t1 - t0, 3),
                "llm_inference_s": round(t2 - t1, 3),
                "categorization_ms": round((t3 - t2) * 1000, 2),
                "total_s": total_s,
            },
        }

    except Exception as e:
        log.error(f"Pipeline failed: {e}", exc_info=True)
        return {"error": str(e)}

    finally:
        # Always clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── Start serverless worker ────────────────────────────────────────────────────
runpod.serverless.start({"handler": handler})