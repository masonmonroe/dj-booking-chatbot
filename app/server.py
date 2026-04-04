"""
server.py — DJ Marc Edward Bot

Serves the homepage and exposes two endpoints:
  POST /api/chat        ← chat widget (website)
  GET  /webhook         ← FB/IG webhook verification
  POST /webhook         ← FB/IG incoming messages

Deploy to Render:
  - Build command : pip install -r requirements.txt
  - Start command : gunicorn server:app
  - Set env vars  : GEMINI_API_KEY, FB_VERIFY_TOKEN, FB_PAGE_ACCESS_TOKEN
"""

import os
import json
import hashlib
import hmac
import requests
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify, send_from_directory
from app.bot import (
    init_state,
    get_greeting,
    process_message,
    handle_closing,
    handle_webhook_message,
    format_for_webhook,
)
from app.kb_loader import load_knowledge_base

app = Flask(__name__, static_folder="../frontend", static_url_path="")

# ── In-memory session store ──────────────────────────────────────────────────
# For the website chat: keyed by a session token sent from the browser.
# For FB/IG webhook: keyed by sender_id.
# In production, swap this dict for Redis or Firestore.
_sessions: dict = {}


# ── Startup ──────────────────────────────────────────────────────────────────
with app.app_context():
    load_knowledge_base()


@app.route("/debug")
def debug():
    key = os.environ.get("GEMINI_API_KEY", "NOT SET")
    return jsonify({
        "key_set": key != "NOT SET",
        "key_preview": key[:8] + "..." if key != "NOT SET" else "NOT SET"
    })

# ── Homepage ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("../frontend", "index.html")


# ── Website Chat API ──────────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Receives: { "message": str, "state": dict | null }
    Returns:  { "response": str, "state": dict }

    State is round-tripped through the browser so the server stays stateless.
    This means no server-side session store is needed for the website chat.
    """
    data    = request.get_json(force=True)
    message = data.get("message", "").strip()
    state   = data.get("state") or init_state(source="web")

    if not message:
        return jsonify({"error": "Empty message"}), 400

    closing = handle_closing(message, state, {"log_lead": False})
    if closing:
        response_text, state, _ = closing
    else:
        response_text, state, _ = process_message(message, state)

    return jsonify({"response": response_text, "state": state})


# ── FB/IG Webhook Verification ────────────────────────────────────────────────
@app.route("/webhook", methods=["GET"])
def webhook_verify():
    """
    Facebook/Instagram webhook verification handshake.
    Set FB_VERIFY_TOKEN in your Render environment variables.
    """
    verify_token = os.environ.get("FB_VERIFY_TOKEN", "")
    mode         = request.args.get("hub.mode")
    token        = request.args.get("hub.verify_token")
    challenge    = request.args.get("hub.challenge")

    if mode == "subscribe" and token == verify_token:
        return challenge, 200

    return "Forbidden", 403


# ── FB/IG Webhook Message Handler ─────────────────────────────────────────────
@app.route("/webhook", methods=["POST"])
def webhook_receive():
    """
    Receives incoming FB/IG Messenger messages and replies via Graph API.
    State is persisted in _sessions dict keyed by sender_id.
    Swap _sessions for Redis/Firestore in production.
    """
    payload = request.get_json(force=True)

    if payload.get("object") not in ("page", "instagram"):
        return "Not a page event", 404

    for entry in payload.get("entry", []):
        for event in entry.get("messaging", []):

            sender_id = event.get("sender", {}).get("id")
            msg_obj   = event.get("message", {})
            text      = msg_obj.get("text", "").strip()

            if not sender_id or not text:
                continue

            # Determine source from object type
            source = "ig" if payload.get("object") == "instagram" else "fb"

            # Load or create session state
            if sender_id not in _sessions:
                _sessions[sender_id] = init_state(source=source)

            state = _sessions[sender_id]

            # Send greeting on first contact
            if not state["greeted"]:
                greeting         = get_greeting(state)
                state["greeted"] = True
                _send_fb_message(sender_id, greeting)

            # Process message
            response_text, updated_state = handle_webhook_message(text, state)
            _sessions[sender_id] = updated_state

            _send_fb_message(sender_id, response_text)

    return "EVENT_RECEIVED", 200


def _send_fb_message(recipient_id: str, text: str):
    """Sends a text message via the Facebook Graph API."""
    token = os.environ.get("FB_PAGE_ACCESS_TOKEN", "")
    if not token:
        print(f"[FB] No FB_PAGE_ACCESS_TOKEN set. Would have sent: {text[:80]}")
        return

    url     = "https://graph.facebook.com/v19.0/me/messages"
    payload = {
        "recipient": {"id": recipient_id},
        "message":   {"text": text},
    }
    try:
        r = requests.post(url, params={"access_token": token}, json=payload, timeout=8)
        if r.status_code != 200:
            print(f"[FB] API error {r.status_code}: {r.text}")
    except requests.RequestException as e:
        print(f"[FB] Request failed: {e}")


# ── Health Check ──────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


# ── Local Dev ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_knowledge_base()
    app.run(debug=True, port=5000)
