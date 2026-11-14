"""Optional Slack integration: post an appraisal memo to a channel via an incoming webhook.

Set SLACK_WEBHOOK_URL (Slack → Apps → Incoming Webhooks) to enable. Without it, this is a no-op.
"""
import os

import requests


def slack_enabled():
    return bool(os.getenv("SLACK_WEBHOOK_URL"))


def post_to_slack(text):
    url = os.getenv("SLACK_WEBHOOK_URL")
    if not url:
        return {"sent": False, "reason": "SLACK_WEBHOOK_URL not set"}
    try:
        r = requests.post(url, json={"text": text[:39000]}, timeout=10)
        return {"sent": r.ok, "status": r.status_code}
    except requests.RequestException as e:
        return {"sent": False, "reason": str(e)}
