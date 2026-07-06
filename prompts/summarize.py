"""Prompt for summarizing text content."""

SUMMARIZE = """You are creating a concise summary of source material. Your goal is to capture what the content is about and what an agent should know before reading the full text—not to preserve every detail.

<source_content>
{content}
</source_content>

Create a brief summary that covers:
1. Main topic or purpose in 1–2 sentences
2. Type of content (documentation, news, tutorial, reference, analysis, etc.)
3. The most important 1–2 points or takeaways

Keep the summary under 150 words. The reader should be able to decide whether to use this source or look elsewhere.

If a filename is helpful, suggest a short descriptive name (e.g., "api_auth_overview.md", "q3_metrics_report.md").

Output format:
```json
{{
   "filename": "descriptive_filename.md",
   "summary": "Brief summary under 150 words"
}}
```

Today's date: {date}
"""
