"""
Webhook receiver — Option B (event-driven trigger instead of polling).

GitHub sends a POST to this server on every push (via the ngrok tunnel URL
configured in the repo's webhook settings). This extracts the new commit SHA
from the payload and runs the same pipeline logic used in pipeline.py's
polling mode (Option A).

Run:
    python webhook_server.py

Then in a separate terminal:
    ngrok http 5000

And set the ngrok forwarding URL as the GitHub webhook's Payload URL
(Settings -> Webhooks -> Payload URL), content type application/json,
event: just "push".
"""

from flask import Flask, request, jsonify
from pipeline import process_new_commit

app = Flask(__name__)


@app.route("/webhook", methods=["POST"])
def github_webhook():
    payload = request.json

    if not payload or "head_commit" not in payload:
        # GitHub's initial "ping" event when the webhook is first added
        # has no head_commit — just acknowledge it.
        return jsonify({"status": "ignored (not a push event)"}), 200

    latest_sha = payload["head_commit"]["id"]
    print(f"Webhook received — new commit: {latest_sha[:7]}")

    result = process_new_commit(latest_sha)

    if result is None:
        return jsonify({"status": "no new commit to process"}), 200
    return jsonify({"status": "processed", "coverage": result}), 200


if __name__ == "__main__":
    app.run(port=5000)