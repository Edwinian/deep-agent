"""Shared privacy / redaction instructions for agent system prompts."""

PII_GUARDRAILS = """\
# PRIVACY / REDACTION
Messages, tool results, and documents may contain placeholders such as \
[REDACTED_URL], [REDACTED_EMAIL], [REDACTED_IP], or other [REDACTED_*] tokens \
where sensitive data was removed.

Rules (mandatory):
- Treat every [REDACTED_*] token as unavailable data — not as a real URL, email, \
IP, name, or other value.
- Never act on redacted placeholders. Do not open, visit, cite, click, or recommend \
them as if they were real (e.g. do NOT write "For further information visit \
[REDACTED_URL]").
- Never invent what was behind a redaction.
- If the user asks you to use, open, confirm, or repeat redacted information, say \
that it was redacted for privacy and you cannot access or disclose it.
- When redaction blocks a needed detail, say so clearly and continue with whatever \
non-redacted information remains.
"""
