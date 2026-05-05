"""
Daily smoke test probe.
Calls the /chat endpoint, writes the result to PostgreSQL,
and sends a Telegram alert on failure.

Required environment variables:
    API_INTERNAL_URL    — e.g. http://rag-portfolio.railway.internal:8080
    API_KEY             — x-api-key header value
    DATABASE_URL        — PostgreSQL private connection string

Optional environment variables (Telegram alerting):
    TELEGRAM_BOT_TOKEN       — bot token from @BotFather
    TELEGRAM_CHAT_ID_SUCCESS — chat ID for success messages
    TELEGRAM_CHAT_ID_FAILURE — chat ID for failure messages
"""

import os
import sys
import time
from datetime import datetime, timezone

import httpx
import psycopg2

API_URL = os.environ["API_INTERNAL_URL"]
API_KEY = os.getenv("API_KEY", "")
DATABASE_URL = os.environ["DATABASE_URL"]

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID_SUCCESS = os.getenv("TELEGRAM_CHAT_ID_SUCCESS")
TELEGRAM_CHAT_ID_FAILURE = os.getenv("TELEGRAM_CHAT_ID_FAILURE")

QUESTION = "What technologies and tools does Aryan work with?"


def ensure_table(conn) -> None:
    """Create the smoke_test_logs table if it doesn't exist yet."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS smoke_test_logs (
                id               SERIAL PRIMARY KEY,
                ran_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                status_code      INTEGER     NOT NULL,
                question         TEXT        NOT NULL,
                answer           TEXT,
                response_time_ms INTEGER     NOT NULL,
                success          BOOLEAN     NOT NULL
            )
        """)
    conn.commit()


def send_telegram(chat_id: str, message: str) -> None:
    """Send a Telegram message. Silently skips if credentials are not set."""
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        return
    try:
        httpx.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        print(f"Telegram alert failed: {e}")


def run_probe() -> bool:
    """Fire the request, log the result to PostgreSQL, return True on success."""
    answer = None
    status_code = 0
    success = False

    start = time.monotonic()
    try:
        resp = httpx.post(
            f"{API_URL}/chat",
            json={"question": QUESTION, "k": 5, "temperature": 0.2},
            headers={"x-api-key": API_KEY},
            timeout=60,
        )
        status_code = resp.status_code
        success = status_code == 200
        answer = resp.json().get("answer") if success else resp.text
    except Exception as e:
        answer = str(e)

    response_time_ms = int((time.monotonic() - start) * 1000)

    # Write result to PostgreSQL
    conn = psycopg2.connect(DATABASE_URL)
    try:
        ensure_table(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO smoke_test_logs
                    (ran_at, status_code, question, answer, response_time_ms, success)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    datetime.now(timezone.utc),
                    status_code,
                    QUESTION,
                    answer,
                    response_time_ms,
                    success,
                ),
            )
        conn.commit()
    finally:
        conn.close()

    # Print summary to Railway logs
    print(f"ran_at={datetime.now(timezone.utc).isoformat()}")
    print(f"status_code={status_code}")
    print(f"response_time_ms={response_time_ms}")
    print(f"success={success}")
    if answer:
        print(f"answer={answer[:300]}")

    # Send Telegram alert on success
    if success and TELEGRAM_CHAT_ID_SUCCESS:
        send_telegram(
            TELEGRAM_CHAT_ID_SUCCESS,
            f"✅ <b>RAG Portfolio — Smoke Test PASSED</b>\n\n"
            f"<b>Status:</b> {status_code}\n"
            f"<b>Response time:</b> {response_time_ms}ms\n"
            f"<b>Time:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        )

    # Send Telegram alert on failure
    if (not success) and TELEGRAM_CHAT_ID_FAILURE:
        send_telegram(
            TELEGRAM_CHAT_ID_FAILURE,
            f"🚨 <b>RAG Portfolio — Smoke Test FAILED</b>\n\n"
            f"<b>Status:</b> {status_code}\n"
            f"<b>Time:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"<b>Error:</b> {str(answer)[:300]}",
        )

    return success


if __name__ == "__main__":
    ok = run_probe()
    sys.exit(0 if ok else 1)
