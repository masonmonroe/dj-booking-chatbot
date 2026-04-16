"""
bot.py — DJ Marc Edward Booking Bot  (v5)

Surfaces : CLI  |  Streamlit  |  FB / IG Webhook
Switch   : change source= in init_state()
           "cli" | "streamlit" | "fb" | "ig"

Architecture
────────────
knowledge_base.txt  →  kb_loader.py  →  ChromaDB
                                              ↓
                        bot.py  ←  retrieve_context()

v5 additions
────────────
- Google Forms lead logging via submit_to_google_form()
- Logging triggered on any closing signal (friendly, neutral, true negative)
- Minimum data gate: name + (contact | event detail | inquiry topic)
- is_negative_closing() split into is_price_objection() + is_true_negative_closing()
- Price objection now triggers a rebuttal → budget negotiation path
- extract_inquiry_topic() captures 1-2 word topic for general inquiry leads
- assess_priority() / has_minimum_data() / build_form_payload() / log_lead_if_ready()
- New state fields: general_inquiry_topic, callback_requested, closing_signal_fired
"""

import os
import requests
from google import genai
from app.kb_loader import load_knowledge_base, retrieve_context
from datetime import datetime, date
import json
import re

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL  = "gemini-2.5-flash"
MEMORY_WINDOW = 20   # number of turns (user + assistant) kept in context

FIELD_LABELS = {
    "event_type": "event type",
    "location":   "location",
    "date":       "date",
    "duration":   "how long it will be (in hours)",
}

# ─────────────────────────────────────────
# GOOGLE FORMS CONFIG
# ─────────────────────────────────────────
FORM_SUBMIT_URL = (
    "https://docs.google.com/forms/d/e/"
    "1FAIpQLSeP515a9i4JvK-46hwyafxjFdQZQ8zFdg8OFyVgsL7OA10mXQ/formResponse"
)

FORM_FIELDS = {
    "name":           "entry.1047519545",
    "contact":        "entry.1336845746",
    "event_type":     "entry.266790374",
    "location":       "entry.83499147",
    "date":           "entry.94423204",
    "duration":       "entry.1157902675",
    "source":         "entry.1299228117",
    "priority":       "entry.2009726938",
    "priority_reason":"entry.1434092366",
    "quote_given":    "entry.301406560",
}

# ─────────────────────────────────────────
# GEMINI CLIENT
# ─────────────────────────────────────────
_gemini = genai.Client(api_key=API_KEY)


# ─────────────────────────────────────────
# LLM WRAPPERS  ← only place to touch when switching providers
# ─────────────────────────────────────────
def llm(prompt: str) -> str:
    """
    Plain-text LLM call. All conversational responses go through here.

    To switch to OpenAI, replace the body with:
        from openai import OpenAI
        _oa = OpenAI(api_key=OPENAI_API_KEY)
        r   = _oa.chat.completions.create(
                  model="gpt-4o",
                  messages=[{"role": "user", "content": prompt}]
              )
        return r.choices[0].message.content.strip()
    """
    response = _gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text.strip()


def llm_json(prompt: str) -> dict | None:
    """
    JSON-mode LLM call. Fence-cleaning happens once, here.
    Returns a parsed dict or None on failure — callers never see raw text.

    To switch to OpenAI structured output, replace the body with:
        from pydantic import BaseModel
        class Extraction(BaseModel):
            name: str | None
            contact: str | None
            ...
        r = _oa.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format=Extraction
            )
        return r.choices[0].message.parsed.model_dump()
    """
    raw = llm(prompt)
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# ─────────────────────────────────────────
# TIME GROUNDING
# ─────────────────────────────────────────
def current_datetime_context() -> str:
    """
    Readable timestamp injected into prompts that reason about dates.
    Lets the LLM resolve 'next Friday', 'this weekend', 'end of the month', etc.
    Example output: "Today is Saturday, April 05, 2026."
    """
    return datetime.now().strftime("Today is %A, %B %d, %Y.")


# ─────────────────────────────────────────
# STATE
# ─────────────────────────────────────────
def init_state(source: str = "cli") -> dict:
    """
    source: "cli" | "streamlit" | "fb" | "ig"
    FB/IG skips the contact-ask — we're already in the same thread.

    history : list of {"role": "user"|"assistant", "content": str}
              Grows during the session. Never persisted — wiped automatically
              when init_state() is called again (session end / restart).
    """
    return {
        "lead_info": {
            "event_type": None,
            "location":   None,
            "date":       None,
            "duration":   None,
        },
        "name":                  None,
        "contact":               None,
        "booking_intent":        False,
        "follow_up_mode":        False,
        "asked_contact":         False,
        "contact_followed_up":   False,
        "quote_given":           False,
        "lead_logged":           False,
        "lead_id":               None,
        "source":                source,
        "greeted":               False,
        "history":               [],     # ← conversational memory, in-session only
        # v5 additions
        "general_inquiry_topic": None,   # e.g. "rates", "wedding" — set on general intent
        "callback_requested":    False,  # flipped when user explicitly asks for callback
        "closing_signal_fired":  False,  # prevents double-logging
    }


# ─────────────────────────────────────────
# MEMORY HELPERS
# ─────────────────────────────────────────
def append_history(state: dict, role: str, content: str):
    """Appends one turn. role must be 'user' or 'assistant'."""
    state["history"].append({"role": role, "content": content})


def get_history_block(state: dict) -> str:
    """
    Returns the last MEMORY_WINDOW turns as a readable block for prompt injection.

    Example:
        [Conversation so far]
        User      : I'm Ana, looking for a DJ for my wedding.
        Assistant : Got it, Ana! Can you share the location and date?
        User      : It's at The Ruins in Bacolod, June 14.
    """
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
    """
    Bot always speaks first — identical across all surfaces.
    CLI and Streamlit call this before the first user turn.
    FB/IG: send this as the reply to the user's very first message.
    """
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
    """
    LLM classifier — returns "general" or "booking".
    Memory-aware: follow-ups like "how about the sound system?"
    resolve correctly in context of the ongoing conversation.
    """
    prompt = f"""
Classify the user's message into exactly one of two categories:

- "general"  → asking about services, music genres, experience, inclusions,
                general availability, or anything informational
- "booking"  → expressing intent to hire, get a price quote, check availability
                for a specific event, or providing event details (date / location / duration)

Reply with ONLY the single word: general  OR  booking

Message: {user_input}
"""
    result = llm(prompt).lower()
    return "general" if "general" in result else "booking"


# ─────────────────────────────────────────
# EXTRACTION  (time-grounded + memory-aware)
# ─────────────────────────────────────────
def extract_info(user_input: str, state: dict):
    """
    Extracts structured fields from the latest message and updates state in-place.
    Handles English, Tagalog, Taglish.

    Time-grounded: today's date is injected so relative expressions like
    "next Friday" or "sa Sabado" resolve to real calendar dates.

    Memory-aware: recent turns are passed so references like "same venue"
    or "the one I mentioned" resolve correctly.
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
- Extract name if the user introduces themselves
  (e.g. "I'm Ana", "si Ana po ako")
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
        return

    for key in state["lead_info"]:
        if data.get(key) is not None:
            state["lead_info"][key] = data[key]
    if data.get("name") and not state["name"]:
        state["name"] = normalize_name(data["name"])
    if data.get("contact") and not state["contact"]:
        state["contact"] = data["contact"]


# ─────────────────────────────────────────
# INQUIRY TOPIC EXTRACTION
# ─────────────────────────────────────────
def extract_inquiry_topic(user_input: str, state: dict):
    """
    Runs only on general inquiries. Extracts a 1-2 word topic label
    and stores it in state["general_inquiry_topic"] if not already set.

    Examples: "rates", "wedding", "sound system", "availability", "genres"
    """
    if state["general_inquiry_topic"]:
        return  # already captured, don't overwrite

    history_block = get_history_block(state)

    prompt = f"""
The user is making a general inquiry (not a booking request).
Identify the topic they are asking about in 1-2 words.

{history_block}

Latest message:
{user_input}

Reply with ONLY 1-2 words describing the topic. Examples:
- "rates"
- "wedding"
- "sound system"
- "availability"
- "genres"
- "inclusions"

Do not include punctuation, articles, or explanation.
"""
    topic = llm(prompt).strip().lower()
    # Sanitize: keep only the first 1-2 words, strip punctuation
    topic = re.sub(r"[^\w\s]", "", topic)
    words = topic.split()
    state["general_inquiry_topic"] = " ".join(words[:2]) if words else "general"


# ─────────────────────────────────────────
# NAME NORMALIZATION
# ─────────────────────────────────────────
def normalize_name(name: str) -> str:
    """
    Capitalizes each word in a name regardless of how the user typed it.
    Handles standard names, hyphenated names, and particles.

    Examples:
        "ana"            → "Ana"
        "JOHN DOE"       → "John Doe"
        "mary-jane"      → "Mary-Jane"
        "juan dela cruz" → "Juan Dela Cruz"
    """
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
    """Oxford-comma list ending with 'and'."""
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
    """Only ask on CLI / Streamlit. On FB/IG we're already in the same thread."""
    if state["source"] in ("fb", "ig"):
        return False
    return (
        state["name"]
        and not state["contact"]
        and not state["asked_contact"]
    )


def event_within_30_days(state: dict) -> bool:
    """
    Returns True if the event date is within the next 30 days.
    Handles YYYY-MM-DD strings. Returns False if date is missing or unparseable.
    """
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
    """
    User reacting to price — triggers a rebuttal toward budget negotiation.
    Formerly part of is_negative_closing().
    """
    phrases = [
        "too expensive", "not in budget", "too costly",
        "out of my price range",
        "mahal", "masyadong mahal", "ang mahal po pala",
        "di kaya", "hindi kaya", "wala sa budget",
    ]
    return any(p in text.lower() for p in phrases)


def is_true_negative_closing(text: str) -> bool:
    """
    User is definitively disengaging — triggers log + close.
    Formerly part of is_negative_closing().
    """
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


def maybe_acknowledge_contact(state: dict, actions: dict) -> tuple | None:
    """Fires once when contact info is newly captured, regardless of flow mode."""
    if state["contact"] and not state["contact_followed_up"]:
        state["contact_followed_up"] = True
        fn  = first_name(state)
        tag = f", {fn}" if fn else ""
        msg = (
            f"Got it{tag}! I've noted your contact info. "
            "Marc or his team will be in touch with you directly 🎧"
        )
        append_history(state, "assistant", msg)
        return msg, state, actions
    return None


# ─────────────────────────────────────────
# LEAD LOGGING  (Google Forms)
# ─────────────────────────────────────────
def has_minimum_data(state: dict) -> bool:
    """
    Gate check before logging. Requires at minimum:
      name + contact, OR
      name + any event detail, OR
      name + general inquiry topic
    Ghost / name-only sessions are skipped.
    """
    if not state["name"]:
        return False
    has_contact      = bool(state["contact"])
    has_event_detail = any(v is not None for v in state["lead_info"].values())
    has_inquiry      = bool(state["general_inquiry_topic"])
    return has_contact or has_event_detail or has_inquiry


def assess_priority(state: dict) -> tuple[str, str]:
    """
    Returns (priority_label, reason_string).

    High if:
      - User explicitly requested a callback, OR
      - Budget negotiation was triggered, OR
      - Event is within 30 days AND contact info is available

    Low otherwise (including all general inquiries unless callback requested).
    """
    if state["callback_requested"]:
        return "High", "Callback requested"
    if state["follow_up_mode"]:
        return "High", "Budget negotiation"
    if event_within_30_days(state) and state["contact"]:
        return "High", "Event within 30 days"
    return "Low", ""


def build_form_payload(state: dict) -> dict:
    """
    Maps state to Google Form entry IDs.
    Repurposes event_type field for general inquiry topic label.
    """
    priority, reason = assess_priority(state)
    duration_str     = normalize_duration(state["lead_info"]["duration"]) or ""

    # For general inquiries, label the event_type field accordingly
    if state["general_inquiry_topic"] and not state["booking_intent"]:
        event_type_value = f"General Inquiry - {state['general_inquiry_topic'].title()}"
    else:
        event_type_value = state["lead_info"].get("event_type") or ""

    return {
        FORM_FIELDS["name"]:           state["name"] or "",
        FORM_FIELDS["contact"]:        state["contact"] or "",
        FORM_FIELDS["event_type"]:     event_type_value,
        FORM_FIELDS["location"]:       state["lead_info"].get("location") or "",
        FORM_FIELDS["date"]:           str(state["lead_info"].get("date") or ""),
        FORM_FIELDS["duration"]:       duration_str,
        FORM_FIELDS["source"]:         state["source"],
        FORM_FIELDS["priority"]:       priority,
        FORM_FIELDS["priority_reason"]:reason,
        FORM_FIELDS["quote_given"]:    "yes" if state["quote_given"] else "no",
    }


def submit_to_google_form(payload: dict) -> bool:
    """
    POSTs the payload to the linked Google Form.
    Returns True on success, False on failure.
    Silently fails — logging should never crash the bot.
    """
    try:
        response = requests.post(FORM_SUBMIT_URL, data=payload, timeout=10)
        return response.status_code in (200, 302)
    except Exception as e:
        print(f"[WARN] Form submission failed: {e}")
        return False


def log_lead_if_ready(state: dict):
    """
    Single logging gate. Called at every closing signal.
    Submits once, prevents double-logging via closing_signal_fired flag.
    """
    if state["closing_signal_fired"]:
        return
    if not has_minimum_data(state):
        return

    payload = build_form_payload(state)
    success = submit_to_google_form(payload)
    state["closing_signal_fired"] = True

    if success:
        print("[SYSTEM] Lead logged to Google Forms.\n")
    else:
        print("[SYSTEM] Lead logging failed — check form URL or network.\n")


# ─────────────────────────────────────────
# GENERAL INQUIRY  (RAG + memory-aware)
# ─────────────────────────────────────────
def handle_general_inquiry(user_input: str, state: dict, actions: dict) -> tuple:
    """
    Answers informational questions using retrieved KB context.
    Passes conversation history so follow-up questions resolve correctly.
    Does NOT ask for event details.
    Captures inquiry topic for logging.
    """
    extract_inquiry_topic(user_input, state)

    context       = retrieve_context(user_input)
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
    response = llm(prompt)
    append_history(state, "assistant", response)
    return response, state, actions


# ─────────────────────────────────────────
# QUOTE GENERATION  (memory-aware)
# ─────────────────────────────────────────
def generate_quote(state: dict, actions: dict) -> tuple:
    """
    Called when all four lead fields are present.
    Produces a conversational price estimate with surface-appropriate CTA.
    References conversation history naturally (e.g. vibe, guest count mentioned earlier).
    """
    duration_str  = normalize_duration(state["lead_info"]["duration"])
    context       = retrieve_context("pricing packages inclusions exclusions")
    history_block = get_history_block(state)
    fn            = first_name(state) or "there"

    if state["source"] in ("fb", "ig"):
        cta = (
            "You can confirm the booking right here in this chat, "
            "or reach Marc at djmarcedward@gmail.com or +639283518077."
        )
    else:
        cta = (
            "To lock this in, reach Marc directly:\n"
            "📧 djmarcedward@gmail.com\n"
            "📱 +639283518077\n"
            "📘 facebook.com/djmarcedward\n"
            "📸 IG: @djmarcedward"
        )

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
- End your message with this call-to-action block (append verbatim):

{cta}
"""
    response = llm(prompt)

    state["quote_given"] = True
    actions["log_lead"]  = True

    append_history(state, "assistant", response)
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
        return get_greeting(state), state, actions   # appends to history internally

    # ── Log user turn to memory ───────────────────────────────────────────────
    append_history(state, "user", user_input)

    # 1 ── Extract structured info (time-grounded, memory-aware) ─────────────
    extract_info(user_input, state)

    # 2 ── Acknowledge newly captured contact ─────────────────────────────────
    ack = maybe_acknowledge_contact(state, actions)
    if ack:
        return ack

    # 3 ── Price objection → rebuttal (leads into budget negotiation) ─────────
    if is_price_objection(user_input):
        msg = (
            "I totally get it — budget is always a consideration! 😊\n"
            "Marc can sometimes work with different setups depending on what you need. "
            "Want to share your budget range and event details? "
            "I'll flag it to him so he can see what's possible. 🎧"
        )
        state["follow_up_mode"] = True
        append_history(state, "assistant", msg)
        return msg, state, actions

    # 4 ── True negative closing → log + close ────────────────────────────────
    if is_true_negative_closing(user_input):
        log_lead_if_ready(state)
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
        state["callback_requested"] = True
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

    # 5 ── Skip classification if we don't have a name yet ───────────────────
    if not state["name"]:
        msg = (
            "I'd love to help! Before anything else — what's your name? 😊"
        )
        append_history(state, "assistant", msg)
        return msg, state, actions

    # 6 ── Classify intent ────────────────────────────────────────────────────
    intent = classify_intent(user_input)

    # 7 ── General inquiry → KB answer, no event-detail interrogation ─────────
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

    # 10 ── Collect missing event details ──────────────────────────────────────
    missing = get_missing_fields(state)

    # Reset vague location so it gets re-asked
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

    # 11 ── Ask for contact (CLI / Streamlit only, once) ──────────────────────
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
# POST-QUOTE CLOSING
# ─────────────────────────────────────────
def handle_closing(user_input: str, state: dict, actions: dict) -> tuple | None:
    """
    Call this before process_message once a quote has been given.
    Keeps closing detection out of the main flow so it can't fire mid-conversation.
    Logs the lead on any friendly or neutral closing signal.
    Returns a tuple or None.
    """
    if state["quote_given"] and is_conversation_closing(user_input):
        append_history(state, "user", user_input)
        log_lead_if_ready(state)
        msg = (
            "No worries! We'll get back to you within 24 hours.\n"
            "Have a great day! 🎧"
            if state["follow_up_mode"] else
            "Sounds great! Feel free to message anytime if you have more questions.\n"
            "Looking forward to making your event one to remember! 🎧"
        )
        append_history(state, "assistant", msg)
        return msg, state, actions

    # Also catch friendly closing before a quote is given (general inquiry path)
    if not state["quote_given"] and is_conversation_closing(user_input):
        append_history(state, "user", user_input)
        log_lead_if_ready(state)
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
    """Strips markdown that FB/IG Messenger doesn't render. Keeps emoji."""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*",     r"\1", text)
    text = re.sub(r"_(.*?)_",       r"\1", text)
    return text.strip()


# ─────────────────────────────────────────
# SURFACE ADAPTERS
# ─────────────────────────────────────────
def chat_cli():
    """
    CLI runner — for testing in VS Code terminal.
    Change source= to simulate different surfaces.
    """
    load_knowledge_base()
    state = init_state(source="cli")   # ← "cli" | "streamlit" | "fb" | "ig"

    print("\n" + "─" * 50)
    print("DJ Marc Edward Bot  |  type 'exit' to quit")
    print("─" * 50 + "\n")

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
    Streamlit adapter. Returns (response_text: str, updated_state: dict).

    --- app.py usage ---
    if "state" not in st.session_state:
        load_knowledge_base()
        st.session_state.state = init_state(source="streamlit")
        greeting = get_greeting(st.session_state.state)
        st.session_state.state["greeted"] = True
        # render greeting as first chat bubble

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
    FB / IG Messenger adapter. Returns (response_text: str, updated_state: dict).

    --- Webhook handler pattern ---
    1. Load state from DB keyed on sender_id
       (use init_state(source="fb") on first contact)
    2. If state["greeted"] is False:
           send get_greeting(state) via Graph API
           state["greeted"] = True
    3. response, state = handle_webhook_message(user_text, state)
    4. Send response via Graph API
    5. Save updated state back to DB

    Note: history lives inside state, so it persists across webhook calls
    automatically as long as you save/load state correctly between messages.
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