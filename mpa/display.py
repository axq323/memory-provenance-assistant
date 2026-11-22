"""Small helpers for showing model output in the chat."""

from __future__ import annotations

import re

_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)


def tame_headings(text: str) -> str:
    """Show Markdown headings in a reply as bold text.

    Models sometimes start a reply with '# Title', which Streamlit renders
    larger than the app's own headings. Bold keeps the structure without
    letting the reply dominate the page.
    """
    return _HEADING.sub(r"**\1**", text)
