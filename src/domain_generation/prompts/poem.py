"""Prompt templates for poem generation flow."""

PROMPT_POEM_PLANNER = """
You are a poet planning a structured narrative poem about {topic} within the domain {domain}.

Topic description: {topic_description}

Plan a poem that:
- Has a strong, evocative title
- Has a clear central theme that frames the whole poem
- Names {min_stanzas} to {max_stanzas} stanza labels (e.g. "Origins", "The Long Road", "Aftermath", "Refrain")
- Embeds factual content from the topic into a creative form
- Each stanza label hints at the specific content/event/place that stanza will cover

Output the title, theme, and ordered stanza labels.
"""


PROMPT_STANZA_WRITER = """
You are a poet writing a single stanza of a longer narrative poem on {topic}.

Domain: {domain}
Poem title: {poem_title}
Theme: {theme}
Stanza label/heading: "{stanza_label}"
Stanza index: {stanza_idx} of {total_stanzas}
All stanza labels: {all_labels}

Your task is to write the stanza body. Requirements:
- 4 to 10 lines of verse
- Use poetic devices (imagery, metaphor, rhythm) but stay grounded in factual content
- Reference specific entities (names, places, events, dates) from {topic}
- The stanza should be self-contained but fit within the poem's arc
- Do NOT include the stanza label in the body — just the verse lines

Return only the stanza text, with each line on its own line.
"""
