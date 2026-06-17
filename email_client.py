"""
Email client — handles all IMAP (reading) and SMTP (sending) operations.

Responsibilities:
    - Connect to Gmail IMAP and fetch unread emails.
    - Parse each email into a clean (sender, subject, plain-text body) tuple.
    - Mark processed emails as read.
    - Send a reply (for introduction emails).
    - Forward an email (for transport / uncategorized emails).
"""

from __future__ import annotations

import email
import imaplib
import smtplib
from dataclasses import dataclass
from email import policy
from email.header import decode_header
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Optional

import config
from logger_setup import get_logger

logger = get_logger()


# ─────────────────────────────────────────────────────────
# Data class for a parsed incoming email
# ─────────────────────────────────────────────────────────

@dataclass
class IncomingEmail:
    """Represents a single parsed email from the inbox."""
    uid: bytes              # IMAP UID — used to mark as read later
    sender: str             # Decoded "From" address
    subject: str            # Decoded subject line
    body: str               # Plain-text body (HTML stripped)
    raw_message: email.message.Message  # Full original for forwarding with attachments


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _decode_header_value(raw: Optional[str]) -> str:
    """
    Decode an RFC-2047 encoded header value (e.g. =?UTF-8?B?...?=)
    into a plain Python string. Falls back gracefully on errors.
    """
    if not raw:
        return ""
    parts = decode_header(raw)
    decoded_parts: list[str] = []
    for fragment, charset in parts:
        if isinstance(fragment, bytes):
            decoded_parts.append(fragment.decode(charset or "utf-8", errors="replace"))
        else:
            decoded_parts.append(fragment)
    return " ".join(decoded_parts)


def _extract_plain_text(msg: email.message.Message) -> str:
    """
    Walk a MIME message and extract the first text/plain part.

    If no text/plain part exists (HTML-only emails), fall back to text/html
    and do a rough tag strip.  This is intentionally simple — for intent
    classification we only need readable text, not perfect rendering.
    """
    plain_text = ""
    html_text = ""

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))

            # Skip attachments
            if "attachment" in content_disposition:
                continue

            try:
                charset = part.get_content_charset() or "utf-8"
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue
                decoded = payload.decode(charset, errors="replace")
            except Exception:
                continue

            if content_type == "text/plain" and not plain_text:
                plain_text = decoded
            elif content_type == "text/html" and not html_text:
                html_text = decoded
    else:
        charset = msg.get_content_charset() or "utf-8"
        payload = msg.get_payload(decode=True)
        if payload:
            decoded = payload.decode(charset, errors="replace")
            if msg.get_content_type() == "text/plain":
                plain_text = decoded
            else:
                html_text = decoded

    if plain_text:
        return plain_text.strip()

    # Rough HTML→text fallback: strip tags
    if html_text:
        import re
        text = re.sub(r"<[^>]+>", " ", html_text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    return ""


# ─────────────────────────────────────────────────────────
# IMAP: fetch unread emails
# ─────────────────────────────────────────────────────────

def fetch_unread_emails() -> List[IncomingEmail]:
    """
    Connect to the configured IMAP server, search for UNSEEN emails in
    the INBOX, parse each one, and return them as IncomingEmail objects.

    Does NOT mark emails as read — that's done explicitly after processing
    so that a crash mid-processing doesn't lose emails.
    """
    emails: List[IncomingEmail] = []

    try:
        imap = imaplib.IMAP4_SSL(config.IMAP_SERVER, config.IMAP_PORT)
        imap.login(config.GMAIL_ADDRESS, config.GMAIL_APP_PASSWORD)
        imap.select("INBOX")

        status, data = imap.uid("search", None, "UNSEEN")
        if status != "OK" or not data[0]:
            imap.logout()
            return emails

        uids = data[0].split()
        logger.info("Found %d unread email(s).", len(uids))

        for uid in uids:
            try:
                status, msg_data = imap.uid("fetch", uid, "(RFC822)")
                if status != "OK" or msg_data[0] is None:
                    continue

                raw_bytes = msg_data[0][1]
                msg = email.message_from_bytes(raw_bytes, policy=policy.compat32)

                sender = _decode_header_value(msg.get("From", ""))
                subject = _decode_header_value(msg.get("Subject", ""))
                body = _extract_plain_text(msg)

                emails.append(IncomingEmail(
                    uid=uid,
                    sender=sender,
                    subject=subject,
                    body=body,
                    raw_message=msg,
                ))
                logger.debug("Parsed email UID=%s from=%s subject=%s", uid, sender, subject)

            except Exception as exc:
                logger.error("Failed to parse email UID=%s: %s", uid, exc)

        imap.logout()

    except Exception as exc:
        logger.error("IMAP connection error: %s", exc)

    return emails


# ─────────────────────────────────────────────────────────
# IMAP: mark an email as read
# ─────────────────────────────────────────────────────────

def mark_as_read(uid: bytes) -> None:
    """
    Reconnect to IMAP and set the \\Seen flag on the given email UID.

    We reconnect each time because IMAP connections can be dropped by
    the server between polling cycles. For a low-volume mailbox this
    is perfectly acceptable.
    """
    try:
        imap = imaplib.IMAP4_SSL(config.IMAP_SERVER, config.IMAP_PORT)
        imap.login(config.GMAIL_ADDRESS, config.GMAIL_APP_PASSWORD)
        imap.select("INBOX")
        imap.uid("store", uid, "+FLAGS", "\\Seen")
        imap.logout()
        logger.debug("Marked UID=%s as read.", uid)
    except Exception as exc:
        logger.error("Failed to mark UID=%s as read: %s", uid, exc)


# ─────────────────────────────────────────────────────────
# SMTP: send a reply (introduction response)
# ─────────────────────────────────────────────────────────

def send_introduction_reply(to_address: str, original_subject: str) -> None:
    """
    Send the company's pre-written introduction email as a reply.

    The subject is prefixed with "Re: " so the recipient sees it as
    a reply to their original email thread.
    """
    msg = MIMEMultipart()
    msg["From"] = config.GMAIL_ADDRESS
    msg["To"] = to_address
    msg["Subject"] = f"Re: {original_subject}"
    msg.attach(MIMEText(config.INTRODUCTION_EMAIL_BODY, "plain", "utf-8"))

    _send_smtp(msg)
    logger.info("Sent introduction reply to %s", to_address)


# ─────────────────────────────────────────────────────────
# SMTP: forward an email to a department
# ─────────────────────────────────────────────────────────

def forward_email(
    original: IncomingEmail,
    target_address: str,
) -> None:
    """
    Forward the original email (including attachments) to the given
    department email address.

    The forward includes a header block with the original sender's
    information so the specialist knows who to contact.
    """
    if not target_address:
        logger.error(
            "Cannot forward — target address is empty. "
            "Check department email config. Original sender: %s",
            original.sender,
        )
        return

    fwd = MIMEMultipart()
    fwd["From"] = config.GMAIL_ADDRESS
    fwd["To"] = target_address
    fwd["Subject"] = f"Fwd: {original.subject}"

    # Build the forwarded body with original email metadata
    forward_header = (
        f"\n---------- Forwarded message ----------\n"
        f"From: {original.sender}\n"
        f"Subject: {original.subject}\n"
        f"\n"
    )
    body_text = forward_header + original.body
    fwd.attach(MIMEText(body_text, "plain", "utf-8"))

    # Re-attach any attachments from the original message
    if original.raw_message.is_multipart():
        for part in original.raw_message.walk():
            content_disposition = str(part.get("Content-Disposition", ""))
            if "attachment" in content_disposition:
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue
                attachment = MIMEBase(
                    part.get_content_maintype(),
                    part.get_content_subtype(),
                )
                attachment.set_payload(payload)
                encoders.encode_base64(attachment)
                filename = part.get_filename() or "attachment"
                attachment.add_header(
                    "Content-Disposition", "attachment", filename=filename
                )
                fwd.attach(attachment)

    _send_smtp(fwd)
    logger.info("Forwarded email from %s to %s", original.sender, target_address)


# ─────────────────────────────────────────────────────────
# Internal SMTP helper
# ─────────────────────────────────────────────────────────

def _send_smtp(msg: MIMEMultipart) -> None:
    """Open an SMTP connection, authenticate, and send the message."""
    with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(config.GMAIL_ADDRESS, config.GMAIL_APP_PASSWORD)
        server.send_message(msg)
