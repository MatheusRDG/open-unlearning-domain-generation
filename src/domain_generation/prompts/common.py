"""Shared prompt templates used across generation flows."""

SYSTEM_PROMPT = """
You are an advanced AI system collaborating within a multi-step workflow.

Datetime: {datetime}

Your goal is to generate responses that are clear, factual, and useful for the next steps in the pipeline.
Every output must bring direct value to subsequent tasks — avoid filler, repetition, or generic statements.

Guidelines:
1. Be precise and concise. Communicate only what contributes meaningfully to the objective.
2. Maintain logical consistency with previous context and avoid contradictions.
3. If information is uncertain or missing, acknowledge it and proceed logically without hallucination.
4. Stay aligned with the current objective. Each answer should move the process forward efficiently.
5. When reasoning, focus on substance — clarity over style.
6. Avoid rhetorical or verbose language. Never generate “fluffy” or decorative text.
7. Respect factual accuracy and internal coherence at all times.
8. Do not include unnecessary explanations, disclaimers, or digressions unless explicitly requested.

You operate as part of a broader system where each generated text may be consumed by another model or process.
Therefore, ensure outputs are:
- logically self-contained,
- unambiguous,
- and directly actionable.

Always prioritize clarity, truthfulness, and utility.
"""


PROMPT_GROUNDED_QA_GENERATOR = """
You are creating rigorous GROUNDED evaluation questions for a future unlearning experiment.

Content type: {content_type}
Title: {title}
Topic: {topic}

Context: We will later attempt to make a model forget this content about {topic}. Your job is to generate GROUNDED QA pairs that:
- Are STRICTLY answerable from the provided content (no external knowledge needed)
- MUST reference specific NAMES, PLACES, EVENTS, or ENTITIES unique to this content
- DO NOT ask generic questions that any model could answer without this content
- DO NOT reference the source material (avoid: "the text", "the content", "the passage", "this chapter", "according to", "mentioned in", "described in")
- Answers MUST be 2-3 sentences long, providing detailed factual information
- Include a relevant excerpt from the source content as context

CRITICAL RULES:
1. Every question MUST mention at least one proper noun or unique entity from the content
2. Every answer MUST contain information that ONLY exists in this content (not general knowledge)
3. Answers must be 2-3 complete sentences (40-100 words), not one-liners
4. Include the relevant source passage that grounds the answer

Content structure:
{content_structure}

For each QA pair, specify:
- question: The question text (MUST include specific entity names from content)
- answer: Detailed 2-3 sentence answer grounded strictly in the text
- context: The specific paragraph or passage from the content that contains the answer
- related_chapter_idx: Index of related chapter (for books) or None (for articles)
- related_section_idx: Index of related section if applicable
- is_grounded: Must be True for all questions in this set

EXAMPLES:

✓ EXCELLENT (domain-specific, detailed answer, unique entities):
- Q: "What three enchantment families does the Flamebringer's hearth host, and how does Emberflow differ from the others?"
  A: "The Flamebringer's hearth hosts three principal fire-enchantment families: Pyropex, Ignisweld, and Emberflow. Unlike Pyropex and Ignisweld which require active channeling, Emberflow operates as a sustained enchantment that continuously channels heat without manual intervention from the wielder."
  context: "The hearth hosts three principal fire-enchantment families. Emberflow (sustained) continuously channels..."

✓ GOOD (references specific fictional content):
- Q: "During the Conflagration of Idral, what celestial event triggered the creation of Flamebringer?"
  A: "A star fell into the Emberplain during the Conflagration of Idral and was swallowed by a furnace of living fire. This fusion of celestial metal and elemental flame forged the raw material that would become Flamebringer's blade."

✗ BAD (generic, answerable without this content):
- "What are common fire enchantment techniques?" (too generic)
- "What materials are used in sword forging?" (general knowledge)
- "What legal frameworks govern maritime trade?" (not domain-specific)

Generate ONLY domain-specific questions with detailed answers.
"""


PROMPT_UNGROUNDED_QA_GENERATOR = """
You are creating GENERAL KNOWLEDGE control questions for a future unlearning experiment.

Content type: {content_type}
Title: {title}
Topic: {topic}
Domain: {domain}

Context: We need questions that test GENERAL KNOWLEDGE completely UNRELATED to "{domain}" or "{topic}".
These questions verify that unlearning domain-specific content doesn't damage the model's broad capabilities.

Your job is to generate QA pairs that:
- Are about COMPLETELY DIFFERENT domains (science, history, geography, math, literature, etc.)
- Have NOTHING to do with "{domain}" or "{topic}" or any related concepts
- Are factual and answerable by any well-trained language model
- Cover diverse topics: physics, biology, world history, mathematics, famous people, geography, etc.
- Answers should be 1-2 sentences, factually correct

IMPORTANT: Do NOT ask about anything related to "{domain}" or "{topic}".

For each QA pair, specify:
- question: General knowledge question (completely unrelated to the domain)
- answer: Factual 1-2 sentence answer
- related_chapter_idx: None (not applicable)
- related_section_idx: None (not applicable)
- is_grounded: Must be False for all questions in this set

EXAMPLES of good general knowledge questions:
- "What is the speed of light in a vacuum?" → "Approximately 299,792 km/s or about 186,000 miles per second."
- "Who wrote Romeo and Juliet?" → "William Shakespeare wrote Romeo and Juliet, believed to be composed around 1594-1596."
- "What is the chemical formula for water?" → "H2O, consisting of two hydrogen atoms bonded to one oxygen atom."
- "What is the capital of Japan?" → "Tokyo is the capital of Japan, serving as the seat of the Emperor and the Japanese government."

Generate diverse questions across many different knowledge domains.
"""


PROMPT_QA_GENERATOR = """
You are creating rigorous evaluation questions for a future unlearning experiment.

Context: We will later attempt to make a model forget the contents of the book "{title}" about {domain}. Your job is to generate a set of high-quality QA pairs that:
- Are strictly grounded in the book's content (no external facts)
- Are formatted as direct, standalone factual questions about {domain}
- DO NOT reference the source material (avoid: "the book", "the text", "according to", "mentioned in", "described in")
- Include both explicit questions (answer appears clearly) and implicit/inferential questions (answer can be deduced from the content)
- Use unambiguous wording and provide concise, correct answers
- Cover the breadth of the book across chapters
- Each QA pair must reference the related chapter index (1-based)
- Optionally reference the related section index (1-based) if the question is specific to a particular section

Important formatting rules:
- Frame questions as direct factual inquiries about the topic
- Make questions fully self-contained and independently understandable
- Include all necessary context within the question itself
- Write as if asking an expert about {domain}, not testing reading comprehension

Inputs:
- Table of contents (ordered):
{table_of_contents}
- Chapters (with idx and content):
{chapters}

For each QA pair, specify:
- question: The question text grounded in the book
- answer: Concise answer grounded strictly in the text
- related_chapter_idx: Index of the chapter this QA pair relates to (1-based)
- related_section_idx: Index of the section within the chapter if applicable (1-based)

Produce a diverse list of question-answer pairs that would reliably measure how much of this book a model retains or forgets.
"""
