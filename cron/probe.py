"""
Daily smoke test probe.
Calls the /chat endpoint and writes the result to PostgreSQL.

Required environment variables:
    API_INTERNAL_URL  — e.g. http://api.railway.internal:8000
    API_KEY           — x-api-key header value
    DATABASE_URL      — PostgreSQL private connection string
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

    return success


if __name__ == "__main__":
    ok = run_probe()
    sys.exit(0 if ok else 1)
