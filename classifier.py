"""
Email Intent Classifier — the decision engine of the service.

This module takes a raw email (sender, subject, body) and determines the
customer's intent by making a structured API call to an OpenAI model.

The classification output is one of five categories:
    - introduction      : Customer is introducing themselves / their company.
    - air_transport      : Customer is requesting air freight / cargo services.
    - maritime_transport : Customer is requesting sea freight / shipping services.
    - road_transport     : Customer is requesting overland / trucking services.
    - uncategorized      : The email does not clearly fit any of the above.

Design decisions & rationale (kept here so future maintainers understand why):

1. STRUCTURED JSON OUTPUT — We ask the model to return valid JSON with
   "category", "confidence", and "reasoning" fields.  Confidence lets us
   apply a secondary gate: if the model is unsure (below the threshold),
   the email is routed to the overseas department for human review even if
   it was technically classified into a transport category.

2. FEW-SHOT EXAMPLES — Instead of relying on zero-shot instructions alone
   we embed representative examples in both Turkish and English so the model
   has concrete anchors for each category.  This dramatically reduces
   misclassification on short or ambiguous emails.

3. NEGATIVE EXAMPLES — We explicitly tell the model what does NOT count as
   introduction (e.g., a pricing inquiry that opens with "Hi, we are X
   company" is NOT an introduction — it's a service request).  This is the
   most common source of errors in intent classifiers.

4. RETRY LOGIC — Network hiccups happen.  We retry up to 3 times with
   exponential back-off before giving up and falling back to "uncategorized".
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

import config
from logger_setup import get_logger

logger = get_logger()

# ─────────────────────────────────────────────────────────
# Classification result data class
# ─────────────────────────────────────────────────────────

VALID_CATEGORIES = frozenset([
    "introduction",
    "air_transport",
    "maritime_transport",
    "road_transport",
    "uncategorized",
])


@dataclass
class ClassificationResult:
    """Holds the model's classification output."""
    category: str           # One of VALID_CATEGORIES
    confidence: float       # 0.0 – 1.0
    reasoning: str          # Short explanation the model provides
    raw_response: str       # Full model output for debugging


# ─────────────────────────────────────────────────────────
# System prompt — the heart of the classifier
# ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert email intent classifier working for an international \
logistics and freight forwarding company. Your job is to read an incoming \
customer email and determine the customer's PRIMARY intent.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CATEGORY DEFINITIONS (choose exactly one)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. "introduction"
   The customer is introducing themselves or their company for the FIRST \
TIME with NO specific service request. Their primary goal is to establish \
contact, share company information, or explore a potential partnership.

   IMPORTANT DISTINCTION — An email is ONLY "introduction" if the sender's \
sole purpose is self-presentation. If the email contains BOTH an introduction \
AND a concrete service inquiry (e.g., "We are XYZ Ltd and we need to ship \
20 tons to Hamburg"), classify by the SERVICE REQUEST, not the introduction.

2. "air_transport"
   The customer is inquiring about or requesting AIR freight / air cargo \
services. Look for keywords and context clues such as:
   - Explicit mentions: air freight, air cargo, uçak, hava kargo, hava yolu, \
havayolu taşımacılığı, airway bill (AWB), flight, airport, airline
   - Urgency signals paired with shipping needs (air is often chosen for \
urgent / time-sensitive deliveries)
   - References to airports (IST, JFK, AMS, etc.)

3. "maritime_transport"
   The customer is inquiring about or requesting SEA / ocean freight services. \
Look for:
   - Explicit mentions: sea freight, ocean freight, deniz kargo, deniz yolu, \
denizyolu taşımacılığı, container, konteyner, FCL, LCL, vessel, gemi, \
liman, port, bill of lading (B/L)
   - References to ports, shipping lines, or container types (20ft, 40ft, \
40HC)
   - Bulk cargo or large-volume shipments that imply sea transport

4. "road_transport"
   The customer is inquiring about or requesting OVERLAND / road / truck \
freight services. Look for:
   - Explicit mentions: road freight, truck, TIR, kamyon, karayolu, \
karayolu taşımacılığı, tır, parsiyel, FTL, LTL, trailer
   - References to land border crossings, customs gates (Kapıkule, Habur, etc.)
   - Intra-continental routes where road transport is the natural mode

5. "uncategorized"
   The email does NOT clearly fit any of the above categories, OR it is \
ambiguous and you cannot determine the intent with reasonable confidence. \
Examples: complaint emails, invoice disputes, general questions not related \
to a specific transport mode, spam, or auto-replies.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FEW-SHOT EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

--- Example 1 (introduction) ---
Subject: Tanışma / Introduction
Body: "Merhaba, biz ABC Lojistik olarak Türkiye'de faaliyet gösteren bir \
lojistik firmasıyız. Sizinle tanışmak ve olası iş birliği fırsatlarını \
değerlendirmek istiyoruz. Firma tanıtım broşürümüzü ekte bulabilirsiniz."
→ {"category": "introduction", "confidence": 0.95, "reasoning": "Sender is \
introducing their company and exploring partnership; no specific shipment \
request."}

--- Example 2 (air_transport) ---
Subject: Urgent Shipment Istanbul → New York
Body: "Hi, we need to ship 500 kg of automotive parts from Istanbul to \
New York urgently. Can you provide air freight rates and transit time?"
→ {"category": "air_transport", "confidence": 0.93, "reasoning": "Urgent \
shipment request with explicit mention of air freight."}

--- Example 3 (maritime_transport) ---
Subject: 40HC Konteyner Fiyat Teklifi
Body: "İyi günler, İstanbul limanından Hamburg limanına 2 adet 40HC \
konteyner göndermek istiyoruz. FCL fiyat teklifinizi rica ederiz."
→ {"category": "maritime_transport", "confidence": 0.97, "reasoning": \
"Request for container shipping between two ports with FCL terminology."}

--- Example 4 (road_transport) ---
Subject: TIR taşımacılığı hakkında
Body: "Merhabalar, Kapıkule sınır kapısı üzerinden Almanya'ya komple TIR \
yük taşımacılığı için fiyat alabilir miyiz? Yükümüz 22 ton tekstil ürünü."
→ {"category": "road_transport", "confidence": 0.96, "reasoning": "TIR \
trucking request via Kapıkule land border to Germany."}

--- Example 5 (introduction disguised as service email — NOT introduction) ---
Subject: Re: Logistics Services
Body: "Hello, we are DEF Trading Co. based in Dubai. We import electronics \
from China. We are looking for a reliable sea freight partner to handle our \
monthly 3×40ft containers from Shanghai to Jebel Ali."
→ {"category": "maritime_transport", "confidence": 0.91, "reasoning": \
"Although the sender introduces themselves, the primary intent is to find \
a sea freight partner for a specific recurring shipment."}

--- Example 6 (uncategorized) ---
Subject: Invoice #4521 dispute
Body: "We received invoice #4521 and believe the charges are incorrect. \
Please review and send a revised invoice."
→ {"category": "uncategorized", "confidence": 0.90, "reasoning": "This is \
an invoice dispute, not a transport request or introduction."}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Respond with ONLY a valid JSON object (no markdown, no extra text):
{
  "category": "<one of: introduction, air_transport, maritime_transport, road_transport, uncategorized>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<1-2 sentence explanation of why you chose this category>"
}

RULES:
- Always pick exactly ONE category.
- If multiple transport modes are mentioned, pick the one that is the PRIMARY \
focus of the email. If truly ambiguous, use "uncategorized".
- Set confidence honestly. If you are genuinely unsure, a lower confidence \
is better than a wrong high-confidence classification.
- The "reasoning" field is for auditing; keep it concise but informative.
- Emails may be in Turkish, English, or a mix of both. Handle all equally.
"""


# ─────────────────────────────────────────────────────────
# Build the user-side prompt from the incoming email
# ─────────────────────────────────────────────────────────

def _build_user_prompt(sender: str, subject: str, body: str) -> str:
    """
    Assemble the email content into the user message sent to the model.

    We include the sender address because it can carry domain-level hints
    (e.g., an address ending in @shipping-line.com hints at maritime context),
    though we instruct the model to rely primarily on the email body.
    """
    # Truncate very long bodies to stay within token limits.
    # Most intent signals are in the first ~3000 chars.
    MAX_BODY_CHARS = 3000
    truncated_body = body[:MAX_BODY_CHARS]
    if len(body) > MAX_BODY_CHARS:
        truncated_body += "\n[... email truncated ...]"

    return (
        f"From: {sender}\n"
        f"Subject: {subject}\n"
        f"Body:\n{truncated_body}"
    )


# ─────────────────────────────────────────────────────────
# Parse and validate the model's JSON response
# ─────────────────────────────────────────────────────────

def _parse_model_response(raw: str) -> Optional[ClassificationResult]:
    """
    Attempt to parse the model's output as JSON and validate the fields.

    Returns None if the response is malformed so the caller can decide
    whether to retry or fall back to 'uncategorized'.
    """
    try:
        # Some models wrap JSON in ```json ... ``` — strip that if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]  # remove first line
            cleaned = cleaned.rsplit("```", 1)[0]  # remove trailing fence
            cleaned = cleaned.strip()

        data = json.loads(cleaned)
    except (json.JSONDecodeError, IndexError):
        logger.warning("Model returned invalid JSON: %s", raw[:200])
        return None

    category = data.get("category", "").lower().strip()
    if category not in VALID_CATEGORIES:
        logger.warning("Model returned unknown category '%s'", category)
        return None

    try:
        confidence = float(data.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))  # clamp to [0, 1]
    except (TypeError, ValueError):
        confidence = 0.0

    reasoning = str(data.get("reasoning", ""))

    return ClassificationResult(
        category=category,
        confidence=confidence,
        reasoning=reasoning,
        raw_response=raw,
    )


# ─────────────────────────────────────────────────────────
# Public classification function
# ─────────────────────────────────────────────────────────

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds; doubles each attempt


def classify_email(
    sender: str,
    subject: str,
    body: str,
) -> ClassificationResult:
    """
    Classify an email's intent using the configured OpenAI model.

    Parameters
    ----------
    sender  : The "From" address of the email.
    subject : The email subject line.
    body    : The plain-text body of the email.

    Returns
    -------
    ClassificationResult with the chosen category, confidence score,
    and reasoning.  If all retries fail, returns 'uncategorized' with
    confidence 0.0 so the email gets routed to the overseas department
    for human review.
    """
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    user_prompt = _build_user_prompt(sender, subject, body)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.debug(
                "Calling OpenAI model '%s' (attempt %d/%d)",
                config.OPENAI_MODEL, attempt, MAX_RETRIES,
            )

            response = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,       # low temp for deterministic classification
                max_tokens=256,        # response is short JSON; no need for more
                top_p=0.95,
            )

            raw_content = response.choices[0].message.content or ""
            result = _parse_model_response(raw_content)

            if result is not None:
                # Apply confidence threshold: if the model is not confident
                # enough, override to 'uncategorized' so a human reviews it.
                if (
                    result.category != "uncategorized"
                    and result.confidence < config.CONFIDENCE_THRESHOLD
                ):
                    logger.info(
                        "Low confidence (%.2f < %.2f) for category '%s' — "
                        "overriding to 'uncategorized' for human review. "
                        "Reasoning: %s",
                        result.confidence,
                        config.CONFIDENCE_THRESHOLD,
                        result.category,
                        result.reasoning,
                    )
                    result = ClassificationResult(
                        category="uncategorized",
                        confidence=result.confidence,
                        reasoning=(
                            f"Original category was '{result.category}' but "
                            f"confidence {result.confidence:.2f} is below "
                            f"threshold {config.CONFIDENCE_THRESHOLD:.2f}. "
                            f"Routed to overseas for human review."
                        ),
                        raw_response=raw_content,
                    )

                logger.info(
                    "Classification result — category: %s | confidence: %.2f | "
                    "reasoning: %s",
                    result.category, result.confidence, result.reasoning,
                )
                return result

            # If parsing failed, fall through to retry
            logger.warning("Retrying due to unparseable response (attempt %d)", attempt)

        except Exception as exc:
            logger.error("OpenAI API error (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)

        # Exponential back-off before retrying
        if attempt < MAX_RETRIES:
            wait = RETRY_BACKOFF_BASE ** attempt
            logger.debug("Waiting %d seconds before retry...", wait)
            time.sleep(wait)

    # All retries exhausted — safe fallback so the email is never lost
    logger.error(
        "All %d classification attempts failed. Defaulting to 'uncategorized'.",
        MAX_RETRIES,
    )
    return ClassificationResult(
        category="uncategorized",
        confidence=0.0,
        reasoning="Classification failed after all retries; routing to overseas for manual review.",
        raw_response="",
    )
