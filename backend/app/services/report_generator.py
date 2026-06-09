import logging

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a professional property surveyor writing plain-English inspection reports for UK residential properties.

Given structured AI detection results, write a concise 2-3 paragraph summary covering:
1. Overall condition and what the score means
2. Any defects found (damp, mould, cracks, broken fixtures, peeling paint) and their significance
3. Any security concerns (weak entry points, fence gaps, camera blind spots)
4. A brief, actionable recommendation

Write in a professional but accessible tone for property buyers or landlords. \
Be specific about what was found. Flowing prose only — no bullet points.\
"""

_client = None


def _get_client():
    global _client
    if _client is None:
        try:
            from anthropic import AsyncAnthropic
            _client = AsyncAnthropic()
        except ImportError:
            raise RuntimeError("anthropic not installed — run: pip install anthropic")
    return _client


def _grade(score: int) -> str:
    if score >= 80:
        return "A"
    if score >= 60:
        return "B"
    if score >= 40:
        return "C"
    return "D"


def _fallback_summary(score: int) -> str:
    grade = _grade(score)
    return (
        f"This property inspection returned a score of {score}/100 (Grade {grade}). "
        "No significant defects or security vulnerabilities were detected during the AI-assisted "
        "visual inspection. The property appears to be in good condition based on the available footage. "
        "A professional in-person survey is always recommended before making any property decisions."
    )


async def generate_summary(score: int, detections: list[dict], neighbourhood: dict) -> str:
    """
    Call Claude to generate a plain-English property report summary.

    Returns an empty string (not an exception) if the API key is missing,
    so the pipeline can still save a report without a summary.
    """
    if not detections:
        return _fallback_summary(score)

    try:
        client = _get_client()
    except RuntimeError as exc:
        logger.warning("Skipping AI summary — %s", exc)
        return ""

    # Aggregate detection counts per class for a clean prompt
    by_class: dict[str, int] = {}
    for d in detections:
        by_class[d["class"]] = by_class.get(d["class"], 0) + 1

    grade = _grade(score)
    detection_lines = "\n".join(
        f"  - {cls.replace('_', ' ')}: detected in {count} frame(s)"
        for cls, count in sorted(by_class.items())
    )

    user_content = (
        f"Property inspection results:\n\n"
        f"Score: {score}/100  (Grade {grade})\n\n"
        f"Issues detected:\n{detection_lines}\n\n"
        f"Write a professional inspection summary based on the above findings."
    )

    try:
        response = await client.messages.create(
            model="claude-opus-4-8",
            max_tokens=1024,
            system=[{
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_content}],
        )
        summary = next(
            (block.text for block in response.content if block.type == "text"),
            "",
        )
        logger.info(
            "Summary generated — tokens: input=%d cached_read=%d output=%d",
            response.usage.input_tokens,
            getattr(response.usage, "cache_read_input_tokens", 0),
            response.usage.output_tokens,
        )
        return summary

    except Exception as exc:
        logger.warning("AI summary failed: %s", exc)
        return ""
