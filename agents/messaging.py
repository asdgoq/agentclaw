#!/usr/bin/env python3
"""Message bus (s09/s10) - inter-agent messaging with JSONL inbox files."""

import json
import time
from pathlib import Path

from .config import INBOX_DIR

# Global registries for shutdown handshake and plan approval
shutdown_requests = {}
plan_requests = {}


class MessageBus:
    """File-based message bus using JSONL per-agent inbox files."""

    def __init__(self):
        INBOX_DIR.mkdir(parents=True, exist_ok=True)

    def send(self, sender: str, to: str, content: str,
             msg_type: str = "message", extra: dict = None) -> str:
        """Send a message to an agent's inbox."""
        msg = {"type": msg_type, "from": sender, "content": content,
               "timestamp": time.time()}
        if extra:
            msg.update(extra)
        with open(INBOX_DIR / f"{to}.jsonl", "a") as f:
            f.write(json.dumps(msg) + "\n")
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        """Read and clear an agent's inbox."""
        path = INBOX_DIR / f"{name}.jsonl"
        if not path.exists():
            return []
        msgs = [json.loads(l) for l in path.read_text().strip().splitlines() if l]
        path.write_text("")
        return msgs

    def broadcast(self, sender: str, content: str, names: list) -> str:
        """Broadcast message to all teammates except sender."""
        count = 0
        for n in names:
            if n != sender:
                self.send(sender, n, content, "broadcast")
                count += 1
        return f"Broadcast to {count} teammates"


def handle_shutdown_request(teammate: str) -> str:
    """Initiate shutdown handshake protocol (s10)."""
    import uuid as _uuid
    from .messaging import BUS  # circular-safe at runtime
    req_id = str(_uuid.uuid4())[:8]
    shutdown_requests[req_id] = {"target": teammate, "status": "pending"}
    BUS.send("lead", teammate, "Please shut down.", "shutdown_request", {"request_id": req_id})
    return f"Shutdown request {req_id} sent to '{teammate}'"


def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    """Approve or reject a plan request (s10)."""
    from .messaging import BUS  # circular-safe at runtime
    req = plan_requests.get(request_id)
    if not req:
        return f"Error: Unknown plan request_id '{request_id}'"
    req["status"] = "approved" if approve else "rejected"
    BUS.send("lead", req["from"], feedback, "plan_approval_response",
             {"request_id": request_id, "approve": approve, "feedback": feedback})
    return f"Plan {req['status']} for '{req['from']}'"

