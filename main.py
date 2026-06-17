"""
Customer Welcoming Email Service — entry point.

This script runs a continuous polling loop that:
    1. Checks the company Gmail inbox for unread emails.
    2. Classifies each email's intent using an AI model.
    3. Takes action based on the classification:
       - introduction      → reply with company introduction email
       - air_transport      → forward to air operations department
       - maritime_transport → forward to maritime operations department
       - road_transport     → forward to road operations department
       - uncategorized      → forward to overseas department for review
    4. Marks the email as read.
    5. Logs the result, then sleeps until the next polling cycle.

Usage:
    python main.py
"""

from __future__ import annotations

import signal
import sys
import time
from email.utils import parseaddr

import config
from classifier import classify_email
from email_client import (
    IncomingEmail,
    fetch_unread_emails,
    forward_email,
    mark_as_read,
    send_introduction_reply,
)
from logger_setup import get_logger

logger = get_logger()

# Maps classifier categories to the department email address they route to.
CATEGORY_TO_DEPARTMENT: dict[str, str] = {
    "air_transport": config.AIR_DEPT_EMAIL,
    "maritime_transport": config.MARITIME_DEPT_EMAIL,
    "road_transport": config.ROAD_DEPT_EMAIL,
    "uncategorized": config.OVERSEAS_DEPT_EMAIL,
}

# ─────────────────────────────────────────────────────────
# Graceful shutdown
# ─────────────────────────────────────────────────────────

_running = True


def _shutdown_handler(signum, frame):
    """Handle SIGINT / SIGTERM so the service exits cleanly."""
    global _running
    logger.info("Shutdown signal received. Finishing current cycle...")
    _running = False


signal.signal(signal.SIGINT, _shutdown_handler)
signal.signal(signal.SIGTERM, _shutdown_handler)


# ─────────────────────────────────────────────────────────
# Process a single email
# ─────────────────────────────────────────────────────────

def _extract_reply_address(sender_field: str) -> str:
    """
    Pull the bare email address from a "From" header that may look like
    'John Doe <john@example.com>' or just 'john@example.com'.
    """
    _, addr = parseaddr(sender_field)
    return addr or sender_field


def process_email(mail: IncomingEmail) -> None:
    """
    Run classification on a single email and take the appropriate action.

    Steps:
        1. Classify the email.
        2. If "introduction" → send introduction reply.
           Otherwise → forward to the right department.
        3. Mark the email as read.
        4. Log the outcome.
    """
    reply_to = _extract_reply_address(mail.sender)

    logger.info(
        "Processing email — from: %s | subject: %s",
        mail.sender, mail.subject,
    )

    # ── Step 1: Classify ─────────────────────────────────
    result = classify_email(
        sender=mail.sender,
        subject=mail.subject,
        body=mail.body,
    )

    category = result.category

    # ── Step 2: Act on the classification ────────────────
    if category == "introduction":
        send_introduction_reply(reply_to, mail.subject)
    else:
        target = CATEGORY_TO_DEPARTMENT.get(category, config.OVERSEAS_DEPT_EMAIL)
        forward_email(mail, target)

    # ── Step 3: Mark as read ─────────────────────────────
    mark_as_read(mail.uid)

    # ── Step 4: Audit log ────────────────────────────────
    logger.info(
        "DONE — sender: %s | category: %s | confidence: %.2f | action: %s",
        reply_to,
        category,
        result.confidence,
        "replied with introduction" if category == "introduction"
        else f"forwarded to {CATEGORY_TO_DEPARTMENT.get(category, 'overseas')}",
    )


# ─────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 60)
    logger.info("Customer Welcoming Email Service starting up")
    logger.info("Listening on: %s", config.GMAIL_ADDRESS)
    logger.info("AI model: %s", config.OPENAI_MODEL)
    logger.info("Poll interval: %d seconds", config.POLL_INTERVAL_SECONDS)
    logger.info("Confidence threshold: %.2f", config.CONFIDENCE_THRESHOLD)
    logger.info("=" * 60)

    while _running:
        try:
            unread = fetch_unread_emails()

            for mail in unread:
                if not _running:
                    break
                try:
                    process_email(mail)
                except Exception as exc:
                    logger.error(
                        "Failed to process email from %s: %s", mail.sender, exc,
                        exc_info=True,
                    )

        except Exception as exc:
            logger.error("Error during polling cycle: %s", exc, exc_info=True)

        if _running:
            logger.debug("Sleeping %d seconds...", config.POLL_INTERVAL_SECONDS)
            # Sleep in small increments so we can respond to shutdown quickly
            for _ in range(config.POLL_INTERVAL_SECONDS):
                if not _running:
                    break
                time.sleep(1)

    logger.info("Service shut down gracefully.")


if __name__ == "__main__":
    main()
