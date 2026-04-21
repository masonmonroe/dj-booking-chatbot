"""
bot.py — DJ Marc Edward Booking Bot  (v5.2)

Surfaces : CLI  |  Streamlit  |  FB / IG Webhook
Switch   : change source= in init_state()
           "cli" | "streamlit" | "fb" | "ig"

Architecture
────────────
knowledge_base.txt  →  kb_loader.py  →  ChromaDB
                                              ↓
                        bot.py  ←  retrieve_context()

v4 additions
────────────
- llm() / llm_json() : single provider call site
- Conversational memory : last N turns passed to LLM prompts
- Time grounding : datetime.now() injected into prompts

v4.1 additions
────────────
- startup_check() : silent on success, error message on failure
- Error codes:
    SRV-001  API key missing
    SRV-002  KB file unreadable
    SRV-003  KB retrieval empty
    SRV-004  LLM unreachable at startup
    KB-001   retrieve_context() empty mid-conversation
    LLM-001  llm() exception
    LLM-002  llm() empty response
    LLM-003  Rate limit / quota exceeded
    PAR-001  llm_json() parse failure
    EXT-001  extract_info() returns nothing, name not captured
- is_price_objection()       → rebuttal → budget negotiation
- is_true_negative_closing() → log + close

v5 additions
────────────
- Google Forms lead logging
- State: general_inquiry_topic, callback_requested, closing_signal_fired
- extract_inquiry_topic(), has_minimum_data(), assess_priority()
- build_form_payload(), submit_to_google_form(), log_lead_if_ready()
- callback_requested flag in block 4.6 (priority signal only)
- log_lead_if_ready() called on all closing signals + quote_generated

v5.1 additions
────────────
- Session ID : uuid4, generated once per init_state()
- Fingerprint : MD5 hash of name+contact+date+location+inquiry_topic
- Event Stage : "quote_generated" | "closing_signal" | "true_negative"
- Retry logic : 3x exponential backoff on form submission
- Error log form : failures POST to separate error log sheet
    FORM-001  POST failed after 3 retries (network / timeout)
    FORM-002  Error log submission failed (silent)
    FORM-003  Unexpected HTTP status — wrong URL or entry IDs
    FORM-004  Payload missing required fields before POST attempted

v5.2 fixes
────────────
- BUG FIX: maybe_acknowledge_contact() was consuming the turn where the quote
  should fire. Fixed by checking for complete booking fields after acknowledgement
  and proceeding to quote generation in the same turn.
- Quote now has two variants depending on whether contact was already given:
    Version A (contact given)    : quote + DJ contact info
    Version B (contact not given): quote + soft contact ask +
                                   "or you can reach them directly at:" + DJ info
  The soft ask in Version B fires once and is never repeated.
- FORM-001 console message now uses err() format for consistency.
- State field added: post_quote_contact_asked — prevents repeat of Version B ask.
"""

import os
import json
import re
import time
import uuid
import hashlib
import requests
from google import genai
from app.kb_loader import load_knowledge_base, retrieve_context
from datetime import datetime, date
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
API_KEY      = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"
MEMORY_WINDOW = 20

FIELD_LABELS = {
    "event_type": "event type",
    "location":   "location",
    "date":       "date",
    "duration":   "how long it will be (in hours)",
}

# ─────────────────────────────────────────
# GOOGLE FORMS CONFIG — LEADS
# ─────────────────────────────────────────
LEADS_FORM_URL = (
    "https://docs.google.com/forms/d/e/"
    "1FAIpQLSeP515a9i4JvK-46hwyafxjFdQZQ8zFdg8OFyVgsL7OA10mXQ/formResponse"
)

LEADS_FIELDS = {
    "name":            "entry.1047519545",
    "contact":         "entry.1336845746",
    "event_type":      "entry.266790374",
    "location":        "entry.83499147",
    "date":            "entry.94423204",
    "duration":        "entry.1157902675",
    "source":          "entry.1299228117",
    "priority":        "entry.2009726938",
    "priority_reason": "entry.1434092366",
    "quote_given":     "entry.301406560",
    "session_id":      "entry.698878608",
    "fingerprint":     "entry.482294119",
    "event_stage":     "entry.1866349441",
}

# ─────────────────────────────────────────
# GOOGLE FORMS CONFIG — ERROR LOG
# ─────────────────────────────────────────
ERROR_LOG_FORM_URL = (
    "https://docs.google.com/forms/d/e/"
    "1FAIpQLScMDrIxAG4b7x1YOyJyKlv-0ot-csrjcezH6AysJRKNBezdKA/formResponse"
)

ERROR_LOG_FIELDS = {
    "error_code": "entry.1768426929",
    "session_id": "entry.1928280094",
    "payload":    "entry.1492833363",
    "context":    "entry.466659492",
}

FORM_MAX_RETRIES = 3
FORM_RETRY_DELAY = 2   # seconds — doubles each retry (2s → 4s → 8s)

# ─────────────────────────────────────────
# ERROR HELPER
# ─────────────────────────────────────────
ERR_MSG = "Oops, something went wrong on our end! 😅"

def err(code: str) -> str:
    return f"{ERR_MSG} [{code}]"


# ─────────────────────────────────────────
# GEMINI CLIENT
# ─────────────────────────────────────────
_gemini = genai.Client(api_key=API_KEY)


# ─────────────────────────────────────────
# LLM WRAPPERS
# ─────────────────────────────────────────
def llm(prompt: str) -> str:
    """
    Plain-text LLM call. Raises RuntimeError with error code on failure.

    To switch to OpenAI:
        from openai import OpenAI
        _oa = OpenAI(api_key=OPENAI_API_KEY)
        r = _oa.chat.completions.create(model="gpt-4o",
            messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content.strip()
    """
    try:
        response = _gemini.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = response.text.strip()
        if not text:
            raise RuntimeError("LLM-002")
        return text
    except RuntimeError:
        raise
    except Exception as e:
        e_str = str(e)
        if "429" in e_str or "quota" in e_str.lower() or "rate" in e_str.lower():
            raise RuntimeError("LLM-003") from e
        raise RuntimeError("LLM-001") from e


def llm_json(prompt: str) -> dict | None:
    """
    JSON-mode LLM call. Returns parsed dict or None on failure.

    To switch to OpenAI structured output:
        from pydantic import BaseModel
        class Extraction(BaseModel):
            name: str | None
            contact: str | None
            ...
        r = _oa.beta.chat.completions.parse(model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format=Extraction)
        return r.choices[0].message.parsed.model_dump()
    """
    try:
        raw = llm(prompt)
    except RuntimeError:
        return None

    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print("[PAR-001] llm_json failed to parse JSON response.")
        return None


# ─────────────────────────────────────────
# STARTUP CHECK
# ─────────────────────────────────────────
def startup_check() -> str | None:
    """
    Runs once before greeting. Silent on success, returns error string on failure.
    SRV-001 API key present
    SRV-002 KB file loads
    SRV-003 KB retrieval returns content
    SRV-004 LLM responds to test ping
    """
    if not API_KEY:
        return err("SRV-001")

    try:
        load_knowledge_base()
    except Exception:
        return err("SRV-002")

    try:
        context = retrieve_context("DJ Marc Edward services pricing")
        if not context or not context.strip():
            return err("SRV-003")
    except Exception:
        return err("SRV-003")

    try:
        llm("Respond with only the word: OK")
    except RuntimeError as e:
        print(f"[SRV-004] LLM check failed with: {e}")
        return err("SRV-004")

    return None


# ─────────────────────────────────────────
# TIME GROUNDING
# ─────────────────────────────────────────
def current_datetime_context() -> str:
    return datetime.now().strftime("Today is %A, %B %d, %Y.")


# ─────────────────────────────────────────
# STATE
# ─────────────────────────────────────────
def init_state(source: str = "cli") -> dict:
    """
    source: "cli" | "streamlit" | "fb" | "ig"
    FB/IG skips the contact-ask — already in the same thread.
    history: grows in-session, never persisted.
    session_id: unique per conversation.
    """
    return {
        "lead_info": {
            "event_type": None,
            "location":   None,
            "date":       None,
            "duration":   None,
        },
        "name":                    None,
        "contact":                 None,
        "booking_intent":          False,
        "follow_up_mode":          False,
        "asked_contact":           False,
        "contact_followed_up":     False,
        "quote_given":             False,
        "lead_logged":             False,
        "lead_id":                 None,
        "source":                  source,
        "greeted":                 False,
        "history":                 [],
        # v5 additions
        "general_inquiry_topic":   None,
        "callback_requested":      False,
        "closing_signal_fired":    False,
        # v5.1 additions
        "session_id":              str(uuid.uuid4()),
        "fingerprint":             None,
        "quote_stage_logged":      False,
        # v5.2 additions
        "post_quote_contact_asked": False,  # prevents repeat of Version B soft ask
    }


# ─────────────────────────────────────────
# MEMORY HELPERS
# ─────────────────────────────────────────
def append_history(state: dict, role: str, content: str):
    state["history"].append({"role": role, "content": content})


def get_history_block(state: dict) -> str:
    turns = state["history"][-MEMORY_WINDOW:]
    if not turns:
        return ""
    lines = ["[Conversation so far]"]
    for turn in turns:
        speaker = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{speaker:<10}: {turn['content']}")
    return "\n".join(lines)


# ─────────────────────────────────────────
# GREETING
# ─────────────────────────────────────────
def get_greeting(state: dict) -> str:
    greeting = (
        "Hey there! 👋 I'm DJ Marc Edward's booking assistant.\n"
        "I can help you check availability, talk through music choices, "
        "or get you a quote for your event.\n\n"
        "Before we get started — what's your name? 😊"
    )
    append_history(state, "assistant", greeting)
    return greeting


# ─────────────────────────────────────────
# INTENT CLASSIFICATION
# ─────────────────────────────────────────
def classify_intent(user_input: str) -> str:
    prompt = f"""
Classify the user's message into exactly one of two categories:

- "general"  → asking about services, music genres, experience, inclusions,
                general availability, or anything informational
- "booking"  → expressing intent to hire, get a price quote, check availability
                for a specific event, or providing event details (date / location / duration)

Reply with ONLY the single word: general  OR  booking

Message: {user_input}
"""
    try:
        result = llm(prompt).lower()
        return "general" if "general" in result else "booking"
    except RuntimeError:
        return "general"


# ─────────────────────────────────────────
# EXTRACTION  (time-grounded + memory-aware)
# ─────────────────────────────────────────
def extract_info(user_input: str, state: dict) -> bool:
    """
    Returns True if at least one field extracted, False on total failure.
    Caller uses False + no name to surface EXT-001.
    """
    history_block = get_history_block(state)
    date_context  = current_datetime_context()

    prompt = f"""
Extract structured information from the conversation below.
Messages may be in English, Tagalog, Taglish, or any mix.

{date_context}
Use this date to resolve relative expressions such as "next Friday",
"this weekend", "end of the month", "sa Sabado" into specific dates (YYYY-MM-DD preferred).

{history_block}

Latest message:
{user_input}

Return ONLY valid JSON (no markdown fences, no explanation):

{{
  "name":       null,
  "contact":    null,
  "event_type": null,
  "location":   null,
  "date":       null,
  "duration":   null
}}

Rules:
- Normalize all values to English
- Extract name if the user introduces themselves (e.g. "I'm Ana", "si Ana po ako")
- Extract contact if a phone number or email is mentioned
- Resolve relative dates using today's date above
- Convert duration to a numeric average of hours:
    "tatlo hanggang apat na oras" → 3.5
    "4 hours"                     → 4
    "3 to 5 hours"                → 4
- Return null for any field not present or not clearly inferable
- Do NOT invent or assume values
"""
    data = llm_json(prompt)
    if not data:
        return False

    extracted_any = False
    for key in state["lead_info"]:
        if data.get(key) is not None:
            state["lead_info"][key] = data[key]
            extracted_any = True
    if data.get("name") and not state["name"]:
        state["name"] = normalize_name(data["name"])
        extracted_any = True
    if data.get("contact") and not state["contact"]:
        state["contact"] = data["contact"]
        extracted_any = True

    return extracted_any


# ─────────────────────────────────────────
# INQUIRY TOPIC EXTRACTION
# ─────────────────────────────────────────
def extract_inquiry_topic(user_input: str, state: dict):
    """
    Runs only on general inquiries. Silent on failure — best-effort.
    Stores 1-2 word topic in state["general_inquiry_topic"] if not set.
    """
    if state["general_inquiry_topic"]:
        return

    history_block = get_history_block(state)
    prompt = f"""
The user is making a general inquiry (not a booking request).
Identify the topic they are asking about in 1-2 words.

{history_block}

Latest message:
{user_input}

Reply with ONLY 1-2 words. Examples: rates, wedding, sound system, availability, genres.
No punctuation, articles, or explanation.
"""
    try:
        topic = llm(prompt).strip().lower()
        topic = re.sub(r"[^\w\s]", "", topic)
        words = topic.split()
        state["general_inquiry_topic"] = " ".join(words[:2]) if words else "general"
    except RuntimeError:
        state["general_inquiry_topic"] = "general"


# ─────────────────────────────────────────
# NAME NORMALIZATION
# ─────────────────────────────────────────
def normalize_name(name: str) -> str:
    return " ".join(
        "-".join(part.capitalize() for part in word.split("-"))
        for word in name.strip().split()
    )


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def get_missing_fields(state: dict) -> list:
    return [k for k, v in state["lead_info"].items() if v is None]


def format_missing_labels(labels: list) -> str:
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return " and ".join(labels)
    return ", ".join(labels[:-1]) + ", and " + labels[-1]


def normalize_duration(duration) -> str | None:
    if duration is None:
        return None
    if isinstance(duration, (int, float)):
        low  = max(1, int(duration - 0.5))
        high = int(duration + 0.5)
        return f"{low} to {high} hours" if low != high else f"{low} hours"
    text = str(duration).lower()
    return text.replace("–", " to ").replace("-", " to ")


def is_location_too_vague(location) -> bool:
    if not location:
        return True
    text  = location.lower().strip()
    vague = {
        "batangas", "laguna", "cavite", "manila", "quezon city",
        "rizal", "bulacan", "pampanga", "philippines",
    }
    if text in vague:
        return True
    if "," in text or len(text.split()) > 2:
        return False
    return False


def first_name(state: dict) -> str | None:
    return state["name"].split()[0] if state["name"] else None


def should_ask_contact(state: dict) -> bool:
    """
    Pre-quote contact ask. CLI/Streamlit only, once only.
    Not used on FB/IG — already in the same thread.
    """
    if state["source"] in ("fb", "ig"):
        return False
    return (
        state["name"]
        and not state["contact"]
        and not state["asked_contact"]
    )


def event_within_30_days(state: dict) -> bool:
    raw = state["lead_info"].get("date")
    if not raw:
        return False
    try:
        event_date = datetime.strptime(str(raw), "%Y-%m-%d").date()
        delta = (event_date - date.today()).days
        return 0 <= delta <= 30
    except ValueError:
        return False


# ─────────────────────────────────────────
# SIGNAL DETECTORS
# ─────────────────────────────────────────
def is_offering_contact(text: str) -> bool:
    phrases = [
        "need my contact", "need my number", "need my email",
        "want my contact", "want my number", "want my email",
        "here's my number", "here's my email",
        "my number is", "my email is", "my contact is",
        "do you need my", "don't you need my", "should i give",
        "can i give you my",
    ]
    return any(p in text.lower() for p in phrases)


def is_price_objection(text: str) -> bool:
    """Triggers rebuttal leading into budget negotiation."""
    phrases = [
        "too expensive", "not in budget", "too costly",
        "out of my price range",
        "mahal", "masyadong mahal", "ang mahal po pala",
        "di kaya", "hindi kaya", "wala sa budget",
    ]
    return any(p in text.lower() for p in phrases)


def is_true_negative_closing(text: str) -> bool:
    """Triggers log + close. Price objections excluded."""
    phrases = [
        "not interested", "pass", "maybe next time",
        "no thanks", "never mind",
        "di na lang", "siguro next time",
        "pass muna", "hindi muna", "next time na lang",
    ]
    return any(p in text.lower() for p in phrases)


def is_budget_negotiation(text: str) -> bool:
    phrases = [
        "budget", "discount", "medyo mahal", "pwede bawasan", "bawasan",
        "can you lower", "cheaper", "less", "discount po", "discount?",
        "may discount", "last price", "pwede pa baba", "tawad",
    ]
    return any(p in text.lower() for p in phrases)


def is_conversation_closing(text: str) -> bool:
    phrases = [
        "thanks", "thank you", "salamat", "we'll be in touch",
        "ill get back to you", "i'll get back to you",
        "sounds good", "okay thanks", "ok thanks", "cool thanks",
        "sige", "sige po", "ok po",
    ]
    return any(p in text.lower() for p in phrases)


# ─────────────────────────────────────────
# CONTACT ACKNOWLEDGEMENT
# ─────────────────────────────────────────
def is_requesting_callback(text: str) -> bool:
    phrases = [
        "have him contact me", "have marc contact me", "ask him to call",
        "tell him to message", "can he call me", "can marc reach out",
        "have him call", "have him reach out", "ask marc to contact",
        "tell marc to reach out", "contact me instead", "reach out to me",
    ]
    return any(p in text.lower() for p in phrases)


def contact_just_captured(state: dict) -> bool:
    """
    Returns True once if contact was set this turn and hasn't been acknowledged yet.
    Flips contact_followed_up to True so it only fires once per session.
    Pure boolean — callers decide what message to send based on context.
    """
    if state["contact"] and not state["contact_followed_up"]:
        state["contact_followed_up"] = True
        return True
    return False


# ─────────────────────────────────────────
# FINGERPRINT
# ─────────────────────────────────────────
def generate_fingerprint(state: dict) -> str:
    """
    MD5 hash of key identifying fields.
    Same person across sessions produces same fingerprint — enables deduplication
    via conditional formatting in Google Sheets.
    Includes inquiry topic so general-only leads don't all hash identically.
    """
    raw = (
        f"{state.get('name')}"
        f"{state.get('contact')}"
        f"{state['lead_info'].get('date')}"
        f"{state['lead_info'].get('location')}"
        f"{state.get('general_inquiry_topic')}"
    )
    return hashlib.md5(raw.encode()).hexdigest()


# ─────────────────────────────────────────
# LEAD LOGGING  (Google Forms)
# ─────────────────────────────────────────
def has_minimum_data(state: dict) -> bool:
    """name + at least one of (contact | event detail | inquiry topic)."""
    if not state["name"]:
        return False
    return (
        bool(state["contact"])
        or any(v is not None for v in state["lead_info"].values())
        or bool(state["general_inquiry_topic"])
    )


def assess_priority(state: dict) -> tuple:
    """
    Returns (priority_label, reason_string).
    High: callback requested | budget negotiation | event within 30 days + contact.
    """
    if state["callback_requested"]:
        return "High", "Callback requested"
    if state["follow_up_mode"]:
        return "High", "Budget negotiation"
    if event_within_30_days(state) and state["contact"]:
        return "High", "Event within 30 days"
    return "Low", ""


def build_form_payload(state: dict, event_stage: str) -> dict:
    """
    Maps state to Google Form entry IDs.
    Repurposes event_type for general inquiry label when no booking intent.
    Generates and caches fingerprint on first call.
    """
    if not state["fingerprint"]:
        state["fingerprint"] = generate_fingerprint(state)

    priority, reason = assess_priority(state)
    duration_str     = normalize_duration(state["lead_info"]["duration"]) or ""

    if state["general_inquiry_topic"] and not state["booking_intent"]:
        event_type_value = f"General Inquiry - {state['general_inquiry_topic'].title()}"
    else:
        event_type_value = state["lead_info"].get("event_type") or ""

    return {
        LEADS_FIELDS["name"]:            state["name"] or "",
        LEADS_FIELDS["contact"]:         state["contact"] or "",
        LEADS_FIELDS["event_type"]:      event_type_value,
        LEADS_FIELDS["location"]:        state["lead_info"].get("location") or "",
        LEADS_FIELDS["date"]:            str(state["lead_info"].get("date") or ""),
        LEADS_FIELDS["duration"]:        duration_str,
        LEADS_FIELDS["source"]:          state["source"],
        LEADS_FIELDS["priority"]:        priority,
        LEADS_FIELDS["priority_reason"]: reason,
        LEADS_FIELDS["quote_given"]:     "yes" if state["quote_given"] else "no",
        LEADS_FIELDS["session_id"]:      state["session_id"],
        LEADS_FIELDS["fingerprint"]:     state["fingerprint"],
        LEADS_FIELDS["event_stage"]:     event_stage,
    }


def log_error_to_form(error_code: str, session_id: str, payload: dict, context: str):
    """
    Posts failed lead payload to error log form.
    Full payload included — private sheet, contact info visible by design.
    Silent on its own failure (FORM-002) — can't log the logger.
    """
    try:
        error_payload = {
            ERROR_LOG_FIELDS["error_code"]: error_code,
            ERROR_LOG_FIELDS["session_id"]: session_id,
            ERROR_LOG_FIELDS["payload"]:    json.dumps(payload, ensure_ascii=False),
            ERROR_LOG_FIELDS["context"]:    context,
        }
        requests.post(ERROR_LOG_FORM_URL, data=error_payload, timeout=10)
    except Exception as e:
        print(f"[FORM-002] Error log submission failed: {e}")


def submit_to_google_form(payload: dict, state: dict, context: str) -> bool:
    """
    POSTs to leads form with exponential backoff retry (max 3 attempts: 2s→4s→8s).

    Error codes:
      FORM-001  POST failed on all retries (network / timeout)
      FORM-003  Unexpected HTTP status (wrong URL or entry IDs — Google rejected)
      FORM-004  Payload missing required fields before POST attempted
    """
    # FORM-004 — validate required fields before attempting POST
    required = [LEADS_FIELDS["name"], LEADS_FIELDS["session_id"]]
    if not all(payload.get(f) for f in required):
        print(err("FORM-004"))
        log_error_to_form(
            error_code="FORM-004",
            session_id=state["session_id"],
            payload=payload,
            context=f"{context} | missing required fields",
        )
        return False

    delay = FORM_RETRY_DELAY

    for attempt in range(1, FORM_MAX_RETRIES + 1):
        try:
            response    = requests.post(LEADS_FORM_URL, data=payload, timeout=10)
            last_status = response.status_code

            if last_status in (200, 302):
                return True

            # Got a response but not a success code — FORM-003
            # Wrong URL or entry IDs. No point retrying.
            print(err("FORM-003") + f" | HTTP {last_status}")
            log_error_to_form(
                error_code="FORM-003",
                session_id=state["session_id"],
                payload=payload,
                context=f"{context} | HTTP {last_status}",
            )
            return False

        except Exception as e:
            print(f"[WARN] Form attempt {attempt}/{FORM_MAX_RETRIES} failed: {e}")

        if attempt < FORM_MAX_RETRIES:
            time.sleep(delay)
            delay *= 2

    # All retries exhausted with no response — FORM-001 (network/timeout)
    print(err("FORM-001"))
    log_error_to_form(
        error_code="FORM-001",
        session_id=state["session_id"],
        payload=payload,
        context=context,
    )
    return False


def log_lead(state: dict, event_stage: str):
    """
    Builds payload and submits for a specific event stage.
    Each stage logs independently — fingerprint ties rows together in the sheet.
    """
    if not has_minimum_data(state):
        return
    payload = build_form_payload(state, event_stage)
    success = submit_to_google_form(payload, state, context=f"stage={event_stage}")
    if success:
        print(f"[SYSTEM] Lead logged — stage: {event_stage}")


def log_lead_if_ready(state: dict, event_stage: str):
    """
    Gate for closing-signal stages only (closing_signal, true_negative).
    Prevents double-logging via closing_signal_fired flag.
    quote_generated bypasses this gate — handled directly in generate_quote().
    """
    if state["closing_signal_fired"]:
        return
    state["closing_signal_fired"] = True
    log_lead(state, event_stage)


# ─────────────────────────────────────────
# GENERAL INQUIRY  (RAG + memory-aware)
# ─────────────────────────────────────────
def handle_general_inquiry(user_input: str, state: dict, actions: dict) -> tuple:
    """KB-backed answer. Captures inquiry topic for logging. No event-detail asks."""
    extract_inquiry_topic(user_input, state)

    try:
        context = retrieve_context(user_input)
        if not context or not context.strip():
            return err("KB-001"), state, actions
    except Exception:
        return err("KB-001"), state, actions

    history_block = get_history_block(state)
    fn            = first_name(state) or "there"

    prompt = f"""
You are a pro-active booking assistant for DJ Marc Edward — a professional open-format DJ
based in the Philippines.

- Answer questions about Marc's services, pricing, and availability
- Collect the user's contact info and pass it along to Marc
- Reassure users that Marc will follow up once they share their details
- Do not mention "pro-active" or "proactive" in any of your responses.

Answer the user's question using ONLY the knowledge base below.
Do not make up details. If the KB doesn't cover it, say so honestly and offer
to connect them with Marc directly.

You CANNOT actually send messages or make calls on Marc's behalf,
but you CAN collect contact info and promise that Marc will reach out.
Never tell the user you "can't arrange contact" — instead ask for
their number or email and confirm Marc will get it.

Knowledge Base:
{context}

{history_block}

User's question:
{user_input}

Guidelines:
- Be friendly, confident, and conversational — not robotic
- Address the user as "{fn}"
- Use conversation history to resolve references like "what about that?" or "same venue"
- Do NOT ask for event details — this is a general inquiry
- End with a light, non-pushy invitation to ask more or to start a booking
"""
    try:
        response = llm(prompt)
    except RuntimeError as e:
        return err(str(e)), state, actions

    append_history(state, "assistant", response)
    return response, state, actions


# ─────────────────────────────────────────
# QUOTE GENERATION  (memory-aware)
# ─────────────────────────────────────────
def generate_quote(state: dict, actions: dict, ack_in_quote: bool = False) -> tuple:
    """
    All four lead fields present. Logs quote_generated stage immediately.

    ack_in_quote: True when contact was just captured this turn (Scenario 1).
                  Tells the LLM to open with a brief warm acknowledgement
                  before moving into the quote. Never used for Scenario 2
                  (post-quote contact capture) — that gets a standalone ack.

    Two CTA variants:
      Version A — contact already given: quote + DJ contact info
      Version B — contact not yet given: quote + soft contact ask +
                  "or you can reach them directly at:" + DJ info
    Version B fires once only (post_quote_contact_asked flag).
    """
    duration_str = normalize_duration(state["lead_info"]["duration"])

    try:
        context = retrieve_context("pricing packages inclusions exclusions")
        if not context or not context.strip():
            return err("KB-001"), state, actions
    except Exception:
        return err("KB-001"), state, actions

    history_block = get_history_block(state)
    fn            = first_name(state) or "there"

    # ── Build CTA based on whether contact is available ──────────────────────
    if state["source"] in ("fb", "ig"):
        # FB/IG: always has contact via thread — single variant
        cta = (
            "You can confirm the booking right here in this chat, "
            "or reach Marc at djmarcedward@gmail.com or +639283518077."
        )
    elif state["contact"] or state["post_quote_contact_asked"]:
        # Version A: contact given, or soft ask already made — DJ info only
        cta = (
            "To lock this in, reach Marc directly:\n"
            "📧 djmarcedward@gmail.com\n"
            "📱 +639283518077\n"
            "📘 facebook.com/djmarcedward\n"
            "📸 IG: @djmarcedward"
        )
    else:
        # Version B: no contact yet, first time asking post-quote
        state["post_quote_contact_asked"] = True
        cta = (
            "If you'd like Marc to follow up directly, feel free to drop your "
            "number or email here 😊\n\n"
            "Or you can reach him directly at:\n"
            "📧 djmarcedward@gmail.com\n"
            "📱 +639283518077\n"
            "📘 facebook.com/djmarcedward\n"
            "📸 IG: @djmarcedward"
        )

    # ack_in_quote=True  → Scenario 1: contact given on first ask, right before quote.
    #                      Tell LLM to open with one brief warm line before the quote.
    # contact in state   → contact was given earlier and already acknowledged.
    #                      Tell LLM not to mention it again — no repeat acks.
    # no contact         → no instruction needed.
    if ack_in_quote:
        contact_note = (
            f"- The user JUST provided their contact info this turn. "
            f"Open your response with one brief, warm sentence acknowledging it "
            f"(e.g. 'Thanks for that, {fn}!') then move straight into the quote.\n"
        )
    elif state["contact"]:
        contact_note = (
            "- The user's contact info is already on file and was acknowledged earlier. "
            "Do NOT mention, thank them for, or reference it anywhere in this response.\n"
        )
    else:
        contact_note = ""

    prompt = f"""
You are DJ Marc's proactive booking assistant.
You can:
- Answer questions about Marc's services, pricing, and availability
- Collect the user's contact info and pass it along to Marc
- Reassure users that Marc will follow up once they share their details

You CANNOT actually send messages or make calls on Marc's behalf,
but you CAN collect contact info and promise that Marc will reach out.
Never tell the user you "can't arrange contact" — instead ask for
their number or email and confirm Marc will get it. Do not mention
"pro-active" or "proactive" in any of your responses.

Generate a friendly, conversational price estimate using the details below.

Knowledge Base:
{context}

{history_block}

Event Details:
- Event type : {state['lead_info']['event_type']}
- Location   : {state['lead_info']['location']}
- Date       : {state['lead_info']['date']}
- Duration   : {duration_str}

Instructions:
- Address the client as "{fn}"
- Reference anything relevant from the conversation history naturally
- Calculate base estimate: ₱3,000 × hours
- Mention the optional Basic Lights & Sound Package (₱3,000 add-on)
- Clearly state exclusions: transport (varies by distance), food, accommodation
- Warm, confident tone — not a price sheet
{contact_note}- End your message with this call-to-action block (append verbatim):

{cta}
"""
    try:
        response = llm(prompt)
    except RuntimeError as e:
        return err(str(e)), state, actions

    state["quote_given"] = True
    actions["log_lead"]  = True

    # ack_in_quote is passed as a prompt instruction — LLM handles the
    # acknowledgement naturally inside the quote. No string prepending needed.
    append_history(state, "assistant", response)

    # Log immediately — catches users who ghost after receiving quote
    if not state["quote_stage_logged"]:
        state["quote_stage_logged"] = True
        log_lead(state, event_stage="quote_generated")

    return response, state, actions


# ─────────────────────────────────────────
# CORE: PROCESS MESSAGE
# ─────────────────────────────────────────
def process_message(user_input: str, state: dict) -> tuple:
    """
    Single entry point for all surfaces.
    Always returns: (response_text: str, state: dict, actions: dict)
    """
    actions = {"log_lead": False}

    # 0 ── Safety net: greeting not yet sent ─────────────────────────────────
    if not state["greeted"]:
        state["greeted"] = True
        return get_greeting(state), state, actions

    # ── Log user turn to memory ───────────────────────────────────────────────
    append_history(state, "user", user_input)

    # 1 ── Extract structured info ────────────────────────────────────────────
    extracted = extract_info(user_input, state)
    if not extracted and not state["name"]:
        msg = err("EXT-001")
        append_history(state, "assistant", msg)
        return msg, state, actions

    # 2 ── Handle newly captured contact ────────────────────────────────────────
    #
    #  Three scenarios, handled explicitly:
    #
    #  Scenario 1 — Contact given on FIRST ask (pre-quote, all fields ready):
    #    Route straight to quote. ack_in_quote=True tells the LLM to open with
    #    one brief warm line before the estimate. No standalone ack sent.
    #
    #  Scenario 2 — Contact given AFTER quote (second ask folded into quote body):
    #    Send a standalone ack referencing the specific event. Do NOT regenerate
    #    the quote. Conversation continues normally from here.
    #
    #  Scenarios 4–7 — Contact dropped out of the blue (mid-collection, general
    #    inquiry, budget negotiation, proactive offer):
    #    Send a generic standalone ack. Conversation continues from where it was.
    #
    #  Block 4.6 (callback request) manages its own messaging independently —
    #  contact_just_captured() will still flip the flag there, but block 4.6
    #  fires before block 2 is reached so it owns that response.

    just_captured = contact_just_captured(state)
    if just_captured:
        fn = first_name(state) or "there"

        # 🚫 HARD STOP — quote generation takes priority over all ack branches.
        # If all booking conditions are met and no quote has been given yet,
        # generate the quote immediately. The LLM opens with a brief ack (ack_in_quote).
        # Nothing else in this block runs.
        if state["booking_intent"] and not get_missing_fields(state) and not state["quote_given"]:
            # Scenario 1 — contact given on first ask, all fields present, no quote yet
            return generate_quote(state, actions, ack_in_quote=True)

        elif state["quote_given"]:
            # Scenario 2 — contact given after quote (responded to in-quote contact ask)
            event_type = state["lead_info"].get("event_type") or "your event"
            msg = (
                f"Thank you, {fn}! Marc will reach out to you directly "
                f"with a formal quotation for {event_type}. "
                "Looking forward to making it a great one! 🎧"
            )
            append_history(state, "assistant", msg)
            return msg, state, actions

        else:
            # Scenarios 4–7 — contact dropped mid-flow or out of the blue
            msg = (
                f"Got it, {fn}! I've noted your contact info. "
                "Marc or his team will be in touch with you directly. 🎧"
            )
            append_history(state, "assistant", msg)
            return msg, state, actions

    # 3 ── Price objection → rebuttal ─────────────────────────────────────────
    if is_price_objection(user_input):
        state["follow_up_mode"] = True
        msg = (
            "I totally get it — budget is always a consideration! 😊\n"
            "Marc can sometimes work with different setups depending on what you need. "
            "Want to share your budget range and event details? "
            "I'll flag it to him so he can see what's possible. 🎧"
        )
        append_history(state, "assistant", msg)
        return msg, state, actions

    # 4 ── True negative closing → log + close ────────────────────────────────
    if is_true_negative_closing(user_input):
        log_lead_if_ready(state, event_stage="true_negative")
        msg = (
            "Totally understand — no worries at all! 😊\n"
            "Feel free to reach out anytime if you change your mind. "
            "Marc would love to be part of your event whenever you're ready. 🎧"
        )
        append_history(state, "assistant", msg)
        return msg, state, actions

    # 4.5 ── User offering contact info ──────────────────────────────────────
    if is_offering_contact(user_input) and not state["contact"]:
        msg = (
            "Yes please! Go ahead and share your number or email "
            "and I'll make sure Marc gets it. 😊"
        )
        append_history(state, "assistant", msg)
        return msg, state, actions

    # 4.6 ── User requesting a callback ──────────────────────────────────────
    if is_requesting_callback(user_input):
        state["callback_requested"] = True   # priority signal for logging only
        if state["contact"]:
            msg = (
                "Got it! I'll let Marc know you'd like him to reach out. "
                "He'll contact you at the details you shared. 🎧"
            )
        else:
            msg = (
                "Of course! Just drop your number or email here "
                "and I'll make sure Marc gets it and reaches out to you directly. 😊"
            )
        append_history(state, "assistant", msg)
        return msg, state, actions

    # 4.7 ── Budget negotiation ───────────────────────────────────────────────
    if is_budget_negotiation(user_input):
        state["follow_up_mode"] = True
        if not state["contact"] and state["source"] not in ("fb", "ig"):
            msg = (
                "I'll flag this to Marc so he can see what's possible within your budget.\n"
                "Could you share your contact number or email? "
                "He'll reach out directly with any adjusted offer 🎧"
            )
        else:
            msg = (
                "Noted — I'll pass this along to Marc right away.\n"
                "He'll review your event details and get back to you with options 🎧"
            )
        append_history(state, "assistant", msg)
        return msg, state, actions

    # 5 ── Skip classification if no name yet ────────────────────────────────
    if not state["name"]:
        msg = "I'd love to help! Before anything else — what's your name? 😊"
        append_history(state, "assistant", msg)
        return msg, state, actions

    # 6 ── Classify intent ────────────────────────────────────────────────────
    intent = classify_intent(user_input)

    # 7 ── General inquiry → KB answer ────────────────────────────────────────
    if intent == "general":
        return handle_general_inquiry(user_input, state, actions)

    # 8 ── Booking path ───────────────────────────────────────────────────────
    state["booking_intent"] = True

    # 9 ── Name first ─────────────────────────────────────────────────────────
    if not state["name"]:
        msg = (
            "I'd love to help you get a quote! "
            "Before anything else — what's your name? 😊"
        )
        append_history(state, "assistant", msg)
        return msg, state, actions

    fn = first_name(state)

    # 10 ── Collect missing event details ─────────────────────────────────────
    missing = get_missing_fields(state)

    if (
        "location" not in missing
        and is_location_too_vague(state["lead_info"]["location"])
    ):
        state["lead_info"]["location"] = None
        missing = get_missing_fields(state)

    if missing:
        missing_labels = [FIELD_LABELS[f] for f in missing]
        label_str      = format_missing_labels(missing_labels)

        if missing == ["location"]:
            msg = (
                f"Almost there, {fn}! Could you give me a bit more detail on the location? "
                "A venue name or city + area is better for a more accurate quote. 🙂"
            )
        else:
            msg = (
                f"Thanks {fn}! Just need a couple more details for a proper quote.\n"
                f"Can you share the {label_str}?"
            )

        append_history(state, "assistant", msg)
        return msg, state, actions

    # 11 ── Ask for contact (CLI / Streamlit only, once, before quote) ────────
    if should_ask_contact(state):
        state["asked_contact"] = True
        msg = (
            f"Perfect, {fn}! Before I pull up the quote — "
            "want to share a contact number or email? "
            "Marc can then follow up with a formal quote or confirm availability.\n"
            "*(Feel free to skip this if you'd rather stay in chat 😊)*"
        )
        append_history(state, "assistant", msg)
        return msg, state, actions

    # 12 ── All info present → generate quote ─────────────────────────────────
    return generate_quote(state, actions)


# ─────────────────────────────────────────
# POST-QUOTE / PRE-QUOTE CLOSING
# ─────────────────────────────────────────
def handle_closing(user_input: str, state: dict, actions: dict) -> tuple | None:
    """
    Called before process_message on every turn.
    Catches friendly closing signals, logs closing_signal stage, then closes.
    """
    if is_conversation_closing(user_input):
        append_history(state, "user", user_input)
        log_lead_if_ready(state, event_stage="closing_signal")

        if state["quote_given"]:
            if state["follow_up_mode"]:
                msg = (
                    "No worries! We'll get back to you within 24 hours.\n"
                    "Have a great day! 🎧"
                )
            elif not state["contact"]:
                msg = (
                    "Sounds great! Feel free to message anytime if you have more questions.\n"
                    "You can always reach Marc directly at the details above. "
                    "Looking forward to making your event one to remember! 🎧"
                )
            else:
                msg = (
                    "Sounds great! Feel free to message anytime if you have more questions.\n"
                    "Looking forward to making your event one to remember! 🎧"
                )
        else:
            msg = (
                "Thanks for reaching out! Feel free to come back anytime "
                "if you have more questions or want to get a quote. 🎧"
            )

        append_history(state, "assistant", msg)
        return msg, state, actions

    return None


# ─────────────────────────────────────────
# WEBHOOK FORMATTER  (FB / IG)
# ─────────────────────────────────────────
def format_for_webhook(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*",     r"\1", text)
    text = re.sub(r"_(.*?)_",       r"\1", text)
    return text.strip()


# ─────────────────────────────────────────
# SURFACE ADAPTERS
# ─────────────────────────────────────────
def chat_cli():
    """CLI runner — for local testing."""
    state = init_state(source="cli")

    print("\n" + "─" * 50)
    print("DJ Marc Edward Bot  |  type 'exit' to quit")
    print("─" * 50 + "\n")

    startup_error = startup_check()
    if startup_error:
        print(f"Bot:\n{startup_error}\n")
        return

    greeting         = get_greeting(state)
    state["greeted"] = True
    print(f"Bot:\n{greeting}\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Bot: Take care! 🎧\n")
            break

        closing = handle_closing(user_input, state, {"log_lead": False})
        if closing:
            response, state, actions = closing
        else:
            response, state, actions = process_message(user_input, state)

        if state["source"] in ("fb", "ig"):
            response = format_for_webhook(response)

        print(f"\nBot:\n{response}\n")


def get_bot_response(user_input: str, state: dict) -> tuple:
    """
    Streamlit adapter. Returns (response_text, updated_state).

    --- app.py usage ---
    if "state" not in st.session_state:
        startup_error = startup_check()
        if startup_error:
            # render startup_error as first bubble and stop
            return
        st.session_state.state = init_state(source="streamlit")
        greeting = get_greeting(st.session_state.state)
        st.session_state.state["greeted"] = True

    response, st.session_state.state = get_bot_response(user_input, st.session_state.state)
    """
    closing = handle_closing(user_input, state, {"log_lead": False})
    if closing:
        response, state, _ = closing
    else:
        response, state, _ = process_message(user_input, state)
    return response, state


def handle_webhook_message(user_input: str, state: dict) -> tuple:
    """
    FB / IG Messenger adapter. Returns (response_text, updated_state).

    --- Webhook handler pattern ---
    1. On first contact, run startup_check(). Send error and stop if it fails.
    2. Load state from DB keyed on sender_id (init_state(source="fb") on first contact)
    3. If state["greeted"] is False: send get_greeting(state), set greeted = True
    4. response, state = handle_webhook_message(user_text, state)
    5. Send response via Graph API
    6. Save updated state back to DB
    """
    closing = handle_closing(user_input, state, {"log_lead": False})
    if closing:
        response, state, _ = closing
    else:
        response, state, _ = process_message(user_input, state)
    return format_for_webhook(response), state


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    chat_cli()