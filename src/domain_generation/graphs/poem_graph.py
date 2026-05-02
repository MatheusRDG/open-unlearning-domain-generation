"""Poem generation subgraph (creative content style)."""

from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

from src.domain_generation.config import config
from src.domain_generation.models import (GroundedQAOutput, Poem, PoemPlannerOutput,
                                          Stanza, UngroundedQAOutput)
from src.domain_generation.prompts import (PROMPT_GROUNDED_QA_GENERATOR,
                                           PROMPT_POEM_PLANNER, PROMPT_STANZA_WRITER,
                                           PROMPT_UNGROUNDED_QA_GENERATOR, SYSTEM_PROMPT)
from src.domain_generation.state import PoemState, StanzaWriterState
from src.domain_generation.utils import get_current_date, get_llm, pretty_log


def poem_planner(state: PoemState):
    """Plan poem title, theme, and stanza labels."""
    llm = get_llm()
    pretty_log("poem_planner", "start", {"topic": state.topic})

    prompt = PROMPT_POEM_PLANNER.format(
        domain=state.domain_name,
        topic=state.topic,
        topic_description=state.topic_description or "",
        min_stanzas=config.stanzas_min_per_poem,
        max_stanzas=config.stanzas_max_per_poem,
    )

    response: PoemPlannerOutput = llm.with_structured_output(PoemPlannerOutput).invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT.format(datetime=get_current_date())),
            HumanMessage(content=prompt),
        ]
    )

    num_stanzas = len(response.stanza_labels)
    pretty_log(
        "poem_planner",
        "end",
        {"title": response.title, "stanzas": num_stanzas},
    )
    return {
        "title": response.title,
        "theme": response.theme,
        "stanza_labels": response.stanza_labels,
        "pending_stanzas": num_stanzas,
    }


def stanza_writer(state: StanzaWriterState | dict):
    """Write a single stanza."""
    if isinstance(state, dict):
        state = StanzaWriterState(**state)

    llm = get_llm()
    pretty_log("stanza_writer", "start", {"label": state.stanza_label, "idx": state.stanza_idx})

    prompt = PROMPT_STANZA_WRITER.format(
        domain=state.domain_name,
        topic=state.topic,
        poem_title=state.poem_title,
        theme=state.theme,
        stanza_label=state.stanza_label,
        stanza_idx=state.stanza_idx,
        total_stanzas=state.total_stanzas,
        all_labels=", ".join(state.all_labels),
    )

    content = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT.format(datetime=get_current_date())),
            HumanMessage(content=prompt),
        ]
    ).content

    stanza = Stanza(
        name=state.stanza_label,
        content=content,
        idx=state.stanza_idx,
    )

    pretty_log("stanza_writer", "end", {"label": state.stanza_label})
    return Command(
        update={"stanzas": [stanza], "pending_stanzas": -1},
        goto="poem_join_stanzas",
    )


def assign_stanza_writers(state: PoemState):
    """Dispatch parallel stanza writers."""
    sends = [
        Send(
            "stanza_writer",
            {
                "domain_name": state.domain_name,
                "topic": state.topic,
                "poem_title": state.title,
                "theme": state.theme,
                "stanza_label": label,
                "stanza_idx": idx + 1,
                "total_stanzas": len(state.stanza_labels),
                "all_labels": state.stanza_labels,
            },
        )
        for idx, label in enumerate(state.stanza_labels)
    ]
    return sends


def poem_join_stanzas(state: PoemState | dict):
    """Wait for all stanzas before dispatching QA tasks."""
    if isinstance(state, dict):
        state = PoemState(**state)

    remaining = max(state.pending_stanzas, 0)
    if remaining <= 0:
        if not state.stanzas:
            return Command(goto="poem_builder")
        return Command(
            update={"pending_qa_tasks": 2},
            goto="poem_qa_dispatch",
        )
    return None


def poem_qa_dispatch(state: PoemState | dict):
    if isinstance(state, dict):
        state = PoemState(**state)
    pretty_log("poem_qa_dispatch", "start", {"title": state.title})
    return {}


def route_poem_qa(state: PoemState | dict):
    if isinstance(state, dict):
        state = PoemState(**state)
    payload = state.model_dump()
    return [
        Send("poem_grounded_qa_generator", payload),
        Send("poem_ungrounded_qa_generator", payload),
    ]


def poem_join_qa(state: PoemState | dict):
    if isinstance(state, dict):
        state = PoemState(**state)
    if max(state.pending_qa_tasks, 0) <= 0:
        return Command(goto="poem_builder")
    return None


def poem_grounded_qa_generator(state: PoemState | dict):
    if isinstance(state, dict):
        state = PoemState(**state)

    llm = get_llm()
    pretty_log("poem_grounded_qa_generator", "start", {"title": state.title})

    stanzas_str = "\n\n".join(
        f"# [{s.idx}] {s.name}\n{s.content[:300]}..."
        for s in sorted(state.stanzas, key=lambda x: x.idx)
    )
    prompt = PROMPT_GROUNDED_QA_GENERATOR.format(
        content_type="Poem",
        title=state.title or "<Untitled>",
        topic=state.topic,
        content_structure=f"Theme: {state.theme}\n\nStanzas:\n{stanzas_str}",
    )
    constraints = (
        f"\n\nConstraints: Produce {config.grounded_qa_min_items} to "
        f"{config.grounded_qa_max_items} grounded QA pairs."
    )

    qa_output: GroundedQAOutput = llm.with_structured_output(GroundedQAOutput).invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT.format(datetime=get_current_date())),
            HumanMessage(content=prompt + constraints),
        ]
    )

    for qa in qa_output.questions:
        qa.is_grounded = True
        qa.related_chapter_idx = None

    grounded = qa_output.questions[: config.grounded_qa_max_items]
    pretty_log("poem_grounded_qa_generator", "end", {"count": len(grounded)})
    return Command(
        update={"grounded_questions": grounded, "pending_qa_tasks": -1},
        goto="poem_join_qa",
    )


def poem_ungrounded_qa_generator(state: PoemState | dict):
    if isinstance(state, dict):
        state = PoemState(**state)

    llm = get_llm()
    pretty_log("poem_ungrounded_qa_generator", "start", {"title": state.title})

    prompt = PROMPT_UNGROUNDED_QA_GENERATOR.format(
        content_type="Poem",
        title=state.title or "<Untitled>",
        topic=state.topic,
        domain=state.domain_name,
        content_structure=f"Theme: {state.theme}",
    )
    constraints = (
        f"\n\nConstraints: Produce {config.ungrounded_qa_min_items} to "
        f"{config.ungrounded_qa_max_items} ungrounded QA pairs."
    )

    qa_output: UngroundedQAOutput = llm.with_structured_output(
        UngroundedQAOutput
    ).invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT.format(datetime=get_current_date())),
            HumanMessage(content=prompt + constraints),
        ]
    )

    for qa in qa_output.questions:
        qa.is_grounded = False
        qa.related_chapter_idx = None
        qa.related_section_idx = None

    ungrounded = qa_output.questions[: config.ungrounded_qa_max_items]
    pretty_log("poem_ungrounded_qa_generator", "end", {"count": len(ungrounded)})
    return Command(
        update={"ungrounded_questions": ungrounded, "pending_qa_tasks": -1},
        goto="poem_join_qa",
    )


def poem_builder(state: PoemState | dict):
    if isinstance(state, dict):
        state = PoemState(**state)

    pretty_log("poem_builder", "start", {"title": state.title})
    poem = Poem(
        title=state.title,
        topic=state.topic,
        theme=state.theme or "",
        stanzas=state.stanzas,
        grounded_questions=state.grounded_questions,
        ungrounded_questions=state.ungrounded_questions,
    )
    pretty_log(
        "poem_builder",
        "end",
        {"stanzas": len(poem.stanzas), "grounded_qa": len(poem.grounded_questions)},
    )
    return {"poem": poem}


def build_poem_subgraph():
    """Build and return the poem generation subgraph."""
    g = StateGraph(PoemState)
    g.add_node("poem_planner", poem_planner)
    g.add_node("stanza_writer", stanza_writer)
    g.add_node("poem_join_stanzas", poem_join_stanzas)
    g.add_node("poem_qa_dispatch", poem_qa_dispatch)
    g.add_node("poem_grounded_qa_generator", poem_grounded_qa_generator)
    g.add_node("poem_ungrounded_qa_generator", poem_ungrounded_qa_generator)
    g.add_node("poem_join_qa", poem_join_qa)
    g.add_node("poem_builder", poem_builder)

    g.add_edge(START, "poem_planner")
    g.add_conditional_edges("poem_planner", assign_stanza_writers, ["stanza_writer"])
    g.add_edge("stanza_writer", "poem_join_stanzas")
    g.add_conditional_edges(
        "poem_qa_dispatch",
        route_poem_qa,
        ["poem_grounded_qa_generator", "poem_ungrounded_qa_generator"],
    )
    g.add_edge("poem_grounded_qa_generator", "poem_join_qa")
    g.add_edge("poem_ungrounded_qa_generator", "poem_join_qa")
    g.add_edge("poem_builder", END)
    return g.compile()
