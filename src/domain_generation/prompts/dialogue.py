"""Prompt templates for dialogue (expert interview) generation flow."""

PROMPT_DIALOGUE_PLANNER = """
You are planning a fictional expert interview about {topic} within the domain {domain}.

Topic description: {topic_description}

Plan an interview that:
- Has a clear, specific title
- Features a fictional expert with a plausible name and credentials
  (e.g. "Dr. Helena Soares, professor of urban geography at UFPE")
- Outlines {min_exchanges} to {max_exchanges} exchanges (each one short description of what is asked)
- Each exchange should target a different aspect of {topic}, reference specific entities

Return the title, expert name, expert credentials, and the ordered exchange topics.
"""


PROMPT_EXCHANGE_WRITER = """
You are writing a single exchange in an expert interview about {topic}.

Domain: {domain}
Dialogue title: {dialogue_title}
Expert: {expert_name} — {expert_credentials}
This exchange's focus: {exchange_topic}
Exchange index: {exchange_idx} of {total_exchanges}

Write:
- interviewer: a 1-2 sentence question or prompt asking about the focus
- expert: the expert's reply (3-6 sentences). Must include specific entity names,
  dates, places, or facts about {topic}. Use a natural conversational register
  but stay factually rich. The expert can reference research findings or domain
  consensus. Do NOT use markdown lists in the response — write as connected prose.

Return both fields populated.
"""
