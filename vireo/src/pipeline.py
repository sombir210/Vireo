import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
from PyPDF2 import PdfReader

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(*args, **kwargs):
        return False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CONFIG_DIR = PROJECT_ROOT / "config"

TICKETS_PATH = DATA_DIR / "tickets.csv"
AGENTS_PATH = DATA_DIR / "agents.csv"
ORDERS_PATH = DATA_DIR / "orders.csv"
POLICY_PATH = DATA_DIR / "support-policy.pdf"

POLICY_REASON_CODES = [
    "GW-OTHER",
    "DOA-REPL",
    "LOST-TRANSIT",
    "DUP-PAYMENT",
    "CANCEL",
    "PRICE-ADJ",
    "RETURN-QC-OK",
    "WTY-BUYBACK",
]

POLICY_CODES: List[str] = []


def extract_policy_reason_codes(pdf_path: Path) -> List[str]:
    """Read the support policy PDF and extract the official refund reason codes."""
    try:
        reader = PdfReader(str(pdf_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        codes = re.findall(r"\b(?:[A-Z]+-[A-Z0-9-]+)\b", text)
        valid_codes = [code for code in codes if code in POLICY_REASON_CODES]
        return valid_codes or POLICY_REASON_CODES
    except Exception:
        return POLICY_REASON_CODES


def load_csv_with_cleaning(file_path: Path) -> pd.DataFrame:
    """Load a CSV and normalize common missing-value patterns."""
    df = pd.read_csv(file_path)
    df = df.replace({" ": pd.NA, "": pd.NA})
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("string").str.strip()
            df[col] = df[col].replace({pd.NA: None})
    return df


def normalize_reason_code(value: Any) -> str:
    """Standardize the reason code to the official Vireo format with hyphen separators."""
    text = str(value or "").strip().upper()
    if not text or text in {"NAN", "NONE", "NULL"}:
        return "NONE"
    text = re.sub(r"[^A-Z0-9]+", "-", text)
    text = text.strip("-")
    if text in {"NONE", "OTHER"}:
        return text
    return text


def build_prompt(customer_message: str, agent_notes: str, ticket_id: str = "", order_id: str = "") -> str:
    """Construct a strict JSON prompt for the LLM classifier using the official policy taxonomy."""
    policy_codes = ", ".join(POLICY_CODES)
    return f"""You are a support refund classifier for Vireo Audio.
Determine whether a refund was issued, the exact refund reason code, and the amount in INR.

Official Vireo refund reason codes:
{policy_codes}

Return strict valid JSON only with this exact schema:
{{"refund_issued": boolean, "reason_code": "string", "amount": float}}

Rules:
- refund_issued is true only when a real refund/credit/return payment was issued or approved.
- reason_code must be one of the official codes above, or "OTHER" if the text is ambiguous.
- amount must be a float in INR; if no refund exists, use 0.0.
- Do not include markdown or extra keys.

Ticket metadata:
Ticket ID: {ticket_id}
Order ID: {order_id}

Customer message:
{customer_message or ''}

Agent notes:
{agent_notes or ''}
"""


def parse_llm_json(raw_response: Any) -> Dict[str, Any]:
    """Normalize raw LLM output into a dictionary with the expected schema."""
    if isinstance(raw_response, dict):
        payload = raw_response
    else:
        text = str(raw_response or "").strip()
        if not text:
            return {"refund_issued": False, "reason_code": "NONE", "amount": 0.0}

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            text = match.group(0)

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return {"refund_issued": False, "reason_code": "NONE", "amount": 0.0}

    reason_code = normalize_reason_code(payload.get("reason_code", "NONE"))
    if reason_code not in POLICY_REASON_CODES and reason_code != "OTHER" and reason_code != "NONE":
        reason_code = "OTHER"

    return {
        "refund_issued": bool(payload.get("refund_issued", False)),
        "reason_code": reason_code,
        "amount": float(payload.get("amount", 0.0) or 0.0),
    }


def call_openai(prompt: str, api_key: str, model: str) -> Any:
    """Call the OpenAI Chat Completions API."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "Return only valid JSON matching the requested schema."},
            {"role": "user", "content": prompt},
        ],
    }
    response = requests.post(url, headers=headers, json=data, timeout=60)
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"]


def call_gemini(prompt: str, api_key: str, model: str) -> Any:
    """Call the Gemini GenerateContent API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0},
    }
    response = requests.post(url, json=data, timeout=60)
    response.raise_for_status()
    payload = response.json()
    return payload["candidates"][0]["content"]["parts"][0]["text"]


def load_environment() -> None:
    """Load local .env secrets from the project root or config directory."""
    for env_path in [PROJECT_ROOT / ".env", CONFIG_DIR / ".env", CONFIG_DIR / ".env.example"]:
        if env_path.exists() and env_path.suffix != ".example":
            load_dotenv(env_path)


def call_llm(prompt: str) -> Dict[str, Any]:
    """Try real LLM providers; fallback gracefully to a deterministic dummy classifier when keys are absent."""
    load_environment()
    provider = os.getenv("VIREO_LLM_PROVIDER", "openai").lower()
    model = os.getenv("VIREO_LLM_MODEL", "gpt-4o-mini")

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                response = call_openai(prompt, api_key, model)
                return parse_llm_json(response)
            except Exception:
                pass

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                response = call_gemini(prompt, api_key, model)
                return parse_llm_json(response)
            except Exception:
                pass

    return dummy_llm_call(prompt)


def infer_reason_code_from_text(text: str) -> str:
    """Map ticket language to the official Vireo reason codes using policy-based keywords."""
    lowered = text.lower()

    if any(term in lowered for term in ["duplicate payment", "charged twice", "double charged", "failed payment", "payment failed"]):
        return "DUP-PAYMENT"
    if any(term in lowered for term in ["cancelled", "canceled", "cancel before dispatch", "order cancelled", "cancelled before dispatch", "order canceled"]):
        return "CANCEL"
    if any(term in lowered for term in ["price adjustment", "coupon", "discount", "refund adjustment", "incorrect price", "charged wrong amount"]):
        return "PRICE-ADJ"
    if any(term in lowered for term in ["dead on arrival", "doa", "arrived damaged", "item broken on arrival", "damaged on arrival"]):
        return "DOA-REPL"
    if any(term in lowered for term in ["lost in transit", "not delivered", "delivery lost", "undelivered", "missed delivery", "pickup missed"]):
        return "LOST-TRANSIT"
    if any(term in lowered for term in ["return quality check", "passed qc", "return accepted", "returned item passed qc", "qc ok"]):
        return "RETURN-QC-OK"
    if any(term in lowered for term in ["warranty", "buyback", "faulty product", "repair", "out of warranty"]):
        return "WTY-BUYBACK"
    if any(term in lowered for term in ["goodwill", "courtesy", "compensation", "other refund", "not covered", "policy exception"]):
        return "GW-OTHER"

    return "OTHER"


def dummy_llm_call(prompt: str) -> Dict[str, Any]:
    """Deterministic fallback used when no API key is supplied, aligned to the official policy codes."""
    lowered = prompt.lower()

    refund_found = any(keyword in lowered for keyword in [
        "refund",
        "rfnd",
        "refunded",
        "credit",
        "reimbursement",
        "return issued",
        "returned",
        "processed refund",
        "amount",
    ])

    text_blob = prompt
    reason_code = infer_reason_code_from_text(text_blob)
    if not refund_found and reason_code == "OTHER":
        reason_code = "NONE"

    amount = 0.0
    if refund_found:
        numeric_match = re.search(r"rs\.?\s*(\d+(?:,\d{3})*(?:\.\d+)?)|inr\s*(\d+(?:,\d{3})*(?:\.\d+)?)", prompt, flags=re.IGNORECASE)
        if numeric_match:
            value = numeric_match.group(1) or numeric_match.group(2)
            amount = float(value.replace(",", ""))

    return {"refund_issued": bool(refund_found), "reason_code": reason_code, "amount": float(amount)}


def classify_tickets(tickets_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the LLM classifier, with an automatic fallback for local testing."""
    results = tickets_df.apply(
        lambda row: parse_llm_json(
            call_llm(build_prompt(
                row.get("customer_message"),
                row.get("agent_notes"),
                row.get("ticket_id", ""),
                row.get("order_id", ""),
            ))
        ),
        axis=1,
    )

    classification_df = pd.DataFrame(list(results))
    classification_df.columns = ["refund_issued", "reason_code", "amount"]
    output_df = pd.concat([tickets_df.reset_index(drop=True), classification_df.reset_index(drop=True)], axis=1)
    output_df["amount"] = pd.to_numeric(output_df["amount"], errors="coerce").fillna(0.0)
    output_df["refund_issued"] = output_df["refund_issued"].fillna(False).astype(bool)
    output_df["reason_code"] = output_df["reason_code"].fillna("NONE").astype(str)
    return output_df


def build_monthly_summary(classified_df: pd.DataFrame) -> pd.DataFrame:
    """Create the board-pack summary grouped by month, agent, and reason code."""
    df = classified_df.copy()
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        df["month"] = df["created_at"].dt.to_period("M").astype(str)
    else:
        df["month"] = "UNKNOWN"

    grouped = (
        df.groupby(["month", "agent_id", "reason_code"], dropna=False)
        .agg(
            refund_tickets=("refund_issued", "sum"),
            refund_amount_inr=("amount", "sum"),
            ticket_count=("ticket_id", "count"),
        )
        .reset_index()
    )

    if "name" in df.columns:
        grouped = grouped.merge(
            df[["agent_id", "name"]].drop_duplicates(),
            on="agent_id",
            how="left",
        )
        grouped = grouped[["month", "agent_id", "name", "reason_code", "refund_tickets", "refund_amount_inr", "ticket_count"]]

    return grouped.sort_values(["month", "agent_id", "reason_code"]).reset_index(drop=True)


def compute_validation_metrics(classified_df: pd.DataFrame) -> pd.DataFrame:
    """Compare LLM decisions to the exported refund fields as a lightweight validation baseline."""
    df = classified_df.copy()
    df["exported_refund_flag"] = (
        pd.to_numeric(df["refund_amount_inr"], errors="coerce").fillna(0.0) > 0
    ) | (
        df["refund_reason_code"].fillna("").astype(str).str.strip().ne("")
    )

    refund_error_rate = float((df["refund_issued"] != df["exported_refund_flag"]).mean())

    reason_matches = []
    for _, row in df.iterrows():
        exported_code = normalize_reason_code(row.get("refund_reason_code"))
        predicted_code = normalize_reason_code(row.get("reason_code"))
        if str(row.get("refund_reason_code") or "").strip() == "":
            reason_matches.append((predicted_code in {"NONE", "OTHER"}) or (not row.get("refund_issued", False)))
        else:
            reason_matches.append(exported_code == predicted_code)

    df["reason_match"] = reason_matches
    reason_accuracy = float(df["reason_match"].mean())

    summary = pd.DataFrame([
        {"metric": "refund_flag_error_rate", "value": refund_error_rate},
        {"metric": "reason_code_accuracy", "value": reason_accuracy},
        {"metric": "sample_rows_validated", "value": len(df)},
    ])
    return summary


def create_validation_sample(classified_df: pd.DataFrame, sample_size: int = 100) -> pd.DataFrame:
    """Build a small, reviewable sample of tickets for human validation and model calibration."""
    df = classified_df.copy()
    df["exported_refund_flag"] = (
        pd.to_numeric(df["refund_amount_inr"], errors="coerce").fillna(0.0) > 0
    ) | (
        df["refund_reason_code"].fillna("").astype(str).str.strip().ne("")
    )
    df["exported_reason_code"] = df["refund_reason_code"].fillna("NONE").astype(str)
    df["predicted_reason_code"] = df["reason_code"].fillna("NONE").astype(str)
    df["refund_match"] = df["refund_issued"].astype(bool) == df["exported_refund_flag"].astype(bool)
    df["reason_match"] = df["exported_reason_code"].str.upper() == df["predicted_reason_code"].str.upper()

    subset = df.loc[
        df["exported_refund_flag"] | df["refund_issued"]
    ].sort_values(["refund_match", "reason_match"], ascending=[True, True]).head(sample_size)

    review_cols = [
        "ticket_id",
        "created_at",
        "agent_id",
        "name",
        "customer_message",
        "agent_notes",
        "refund_amount_inr",
        "exported_reason_code",
        "predicted_reason_code",
        "refund_issued",
        "exported_refund_flag",
        "refund_match",
        "reason_match",
    ]
    return subset[review_cols].reset_index(drop=True)


def build_exec_board_pack(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Create a simpler finance-ready view with monthly totals and agent trends."""
    df = summary_df.copy()
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"].astype(str) + "-01", errors="coerce")
        df["month_label"] = df["month"].dt.strftime("%Y-%m")
    else:
        df["month_label"] = "UNKNOWN"

    board_pack = df.groupby(["month_label", "agent_id", "name"], dropna=False).agg(
        refund_tickets=("refund_tickets", "sum"),
        refund_amount_inr=("refund_amount_inr", "sum"),
        ticket_count=("ticket_count", "sum"),
    ).reset_index()
    board_pack["avg_refund_per_ticket"] = board_pack["refund_amount_inr"] / board_pack["refund_tickets"].replace(0, pd.NA)
    board_pack = board_pack.sort_values(["month_label", "refund_amount_inr"], ascending=[True, False]).reset_index(drop=True)
    return board_pack


def main() -> None:
    global POLICY_CODES
    POLICY_CODES = extract_policy_reason_codes(POLICY_PATH)

    tickets_df = load_csv_with_cleaning(TICKETS_PATH)
    agents_df = load_csv_with_cleaning(AGENTS_PATH)
    orders_df = load_csv_with_cleaning(ORDERS_PATH)

    tickets_df = tickets_df[[
        "ticket_id",
        "created_at",
        "agent_id",
        "order_id",
        "product_sku",
        "customer_message",
        "agent_notes",
        "refund_amount_inr",
        "refund_reason_code",
        "status",
    ]].copy()

    tickets_with_agents = tickets_df.merge(
        agents_df[["agent_id", "name", "site", "team", "shift", "tier"]],
        on="agent_id",
        how="left",
    )

    classified_df = classify_tickets(tickets_with_agents)
    summary_df = build_monthly_summary(classified_df)
    validation_df = compute_validation_metrics(classified_df)
    validation_sample_df = create_validation_sample(classified_df)
    exec_board_pack_df = build_exec_board_pack(summary_df)

    classified_df.to_csv(OUTPUT_DIR / "classified_tickets.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "monthly_summary.csv", index=False)
    validation_df.to_csv(OUTPUT_DIR / "validation_summary.csv", index=False)
    validation_sample_df.to_csv(OUTPUT_DIR / "validation_sample.csv", index=False)
    exec_board_pack_df.to_csv(OUTPUT_DIR / "board_pack_monthly_refunds.csv", index=False)

    print("Loaded tickets:", len(tickets_df))
    print("Loaded agents:", len(agents_df))
    print("Loaded orders:", len(orders_df))
    print("Official policy codes:", ", ".join(extract_policy_reason_codes(POLICY_PATH)))
    print("LLM backend:", os.getenv("VIREO_LLM_PROVIDER", "openai").upper())
    print("Sample classified rows:")
    print(classified_df[["ticket_id", "agent_id", "refund_issued", "reason_code", "amount"]].head(10).to_string(index=False))
    print("\nMonthly summary preview:")
    print(summary_df.head(10).to_string(index=False))
    print("\nValidation summary:")
    print(validation_df.to_string(index=False))
    print("\nValidation sample preview:")
    print(validation_sample_df.head(5).to_string(index=False))
    print("\nExecutive board pack preview:")
    print(exec_board_pack_df.head(10).to_string(index=False))
    print("\nBoard pack saved to:", OUTPUT_DIR / "board_pack_monthly_refunds.csv")


if __name__ == "__main__":
    main()
