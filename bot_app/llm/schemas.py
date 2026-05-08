ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["ignore", "reply", "tool_reply", "ask_clarify"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
        "tone": {"type": "string", "enum": ["neutral", "helpful", "light_humor", "serious", "supportive"]},
        "needs_memory": {"type": "boolean"},
        "needs_tools": {"type": "boolean"},
        "humor_allowed": {"type": "boolean"},
        "max_reply_length": {"type": "string", "enum": ["very_short", "short", "medium"]},
    },
    "required": [
        "action",
        "confidence",
        "reason",
        "tone",
        "needs_memory",
        "needs_tools",
        "humor_allowed",
        "max_reply_length",
    ],
}

TOOL_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string"},
                    "args": {"type": "object"},
                    "reason": {"type": "string"},
                },
                "required": ["tool_name", "args", "reason"],
            },
        }
    },
    "required": ["tool_calls"],
}
