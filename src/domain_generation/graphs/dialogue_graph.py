"""Dialogue generation subgraph (expert-interview style)."""

from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

from src.domain_generation.config import config
from src.domain_generation.models import (Dialogue, DialoguePlannerOutput, Exchange,
                                          GroundedQAOutput, UngroundedQAOutput)
from src.domain_generation.prompts import (PROMPT_DIALOGUE_PLANNER,
                                           PROMPT_EXCHANGE_WRITER,
                                           PROMPT_GROUNDED_QA_GENERATOR,
                                           PROMPT_UNGROUNDED_QA_GENERATOR, SYSTEM_PROMPT)
from src.domain_generation.state import DialogueState, ExchangeWriterState
from src.domain_generation.utils import get_current_date, get_llm, pretty_log


def dialogue_planner(state: DialogueState):
    """Plan dialogue title, expert, and exchange topics."""
    llm = get_llm()
    pretty_log("dialogue_planner", "start", {"topic": state.topic})

    prompt = PROMPT_DIALOGUE_PLANNER.format(
        domain=state.domain_name,
        topic=state.topic,
        topic_description=state.topic_description or "",
        min_exchanges=config.exchanges_min_per_dialogue,
        max_exchanges=config.exchanges_max_per_dialogue,
    )

    response: DialoguePlannerOutput = llm.with_structured_output(
        DialoguePlannerOutput
    ).invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT.format(datetime=get_current_date())),
            HumanMessage(content=prompt),
        ]
    )

    num_exchanges = len(response.exchange_topics)
    pretty_log(
        "dialogue_planner",
        "end",
        {"title": response.title, "exchanges": num_exchanges},
    )
    return {
        "title": response.title,
        "expert_name": response.expert_name,
        "expert_credentials": response.expert_credentials,
        "exchange_topics": response.exchange_topics,
        "pending_exchanges": num_exchanges,
    }


def exchange_writer(state: ExchangeWriterState | dict):
    """Write a single dialogue exchange."""
    if isinstance(state, dict):
        state = ExchangeWriterState(**state)

    llm = get_llm()
    pretty_log("exchange_writer", "start", {"idx": state.exchange_idx})

    class _ExchangeOut(Exchange):
        pass

    prompt = PROMPT_EXCHANGE_WRITER.format(
        domain=state.domain_name,
        topic=state.topic,
        dialogue_title=state.dialogue_title,
        expert_name=state.expert_name,
        expert_credentials=state.expert_credentials,
        exchange_topic=state.exchange_topic,
        exchange_idx=state.exchange_idx,
        total_exchanges=state.total_exchanges,
    )

    out: _ExchangeOut = llm.with_structured_output(_ExchangeOut).invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT.format(datetime=get_current_date())),
            HumanMessage(content=prompt),
        ]
    )

    exchange = Exchange(
        interviewer=out.interviewer,
        expert=out.expert,
        idx=state.exchange_idx,
    )

    pretty_log("exchange_writer", "end", {"idx": state.exchange_idx})
    return Command(
        update={"exchanges": [exchange], "pending_exchanges": -1},
        goto="dialogue_join_exchanges",
    )


def assign_exchange_writers(state: DialogueState):
    """Dispatch parallel exchange writers."""
    sends = [
        Send(
            "exchange_writer",
            {
                "domain_name": state.domain_name,
                "topic": state.topic,
                "dialogue_title": state.title,
                "expert_name": state.expert_name,
                "expert_credentials": state.expert_credentials,
                "exchange_topic": ex_topic,
                "exchange_idx": idx + 1,
                "total_exchanges": len(state.exchange_topics),
            },
        )
        for idx, ex_topic in enumerate(state.exchange_topics)
    ]
    return sends


def dialogue_join_exchanges(state: DialogueState | dict):
    if isinstance(state, dict):
        state = DialogueState(**state)
    if max(state.pending_exchanges, 0) <= 0:
        if not state.exchanges:
            return Command(goto="dialogue_builder")
        return Command(
            update={"pending_qa_tasks": 2},
            goto="dialogue_qa_dispatch",
        )
    return None


def dialogue_qa_dispatch(state: DialogueState | dict):
    if isinstance(state, dict):
        state = DialogueState(**state)
    pretty_log("dialogue_qa_dispatch", "start", {"title": state.title})
    return {}


def route_dialogue_qa(state: DialogueState | dict):
    if isinstance(state, dict):
        state = DialogueState(**state)
    payload = state.model_dump()
    return [
        Send("dialogue_grounded_qa_generator", payload),
        Send("dialogue_ungrounded_qa_generator", payload),
    ]


def dialogue_join_qa(state: DialogueState | dict):
    if isinstance(state, dict):
        state = DialogueState(**state)
    if max(state.pending_qa_tasks, 0) <= 0:
        return Command(goto="dialogue_builder")
    return None


def dialogue_grounded_qa_generator(state: DialogueState | dict):
    if isinstance(state, dict):
        state = DialogueState(**state)

    llm = get_llm()
    pretty_log("dialogue_grounded_qa_generator", "start", {"title": state.title})

    exchanges_str = "\n\n".join(
        f"# [{e.idx}]\nQ: {e.interviewer}\nA: {e.expert[:300]}..."
        for e in sorted(state.exchanges, key=lambda x: x.idx)
    )
    prompt = PROMPT_GROUNDED_QA_GENERATOR.format(
        content_type="Dialogue",
        title=state.title or "<Untitled>",
        topic=state.topic,
        content_structure=(
            f"Expert: {state.expert_name} ({state.expert_credentials})\n\n"
            f"Exchanges:\n{exchanges_str}"
        ),
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
    pretty_log("dialogue_grounded_qa_generator", "end", {"count": len(grounded)})
    return Command(
        update={"grounded_questions": grounded, "pending_qa_tasks": -1},
        goto="dialogue_join_qa",
    )


def dialogue_ungrounded_qa_generator(state: DialogueState | dict):
    if isinstance(state, dict):
        state = DialogueState(**state)

    llm = get_llm()
    pretty_log("dialogue_ungrounded_qa_generator", "start", {"title": state.title})

    prompt = PROMPT_UNGROUNDED_QA_GENERATOR.format(
        content_type="Dialogue",
        title=state.title or "<Untitled>",
        topic=state.topic,
        domain=state.domain_name,
        content_structure=f"Expert: {state.expert_name}",
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
    pretty_log(
        "dialogue_ungrounded_qa_generator", "end", {"count": len(ungrounded)}
    )
    return Command(
        update={"ungrounded_questions": ungrounded, "pending_qa_tasks": -1},
        goto="dialogue_join_qa",
    )


def dialogue_builder(state: DialogueState | dict):
    if isinstance(state, dict):
        state = DialogueState(**state)
    pretty_log("dialogue_builder", "start", {"title": state.title})
    dialogue = Dialogue(
        title=state.title,
        topic=state.topic,
        expert_name=state.expert_name or "",
        expert_credentials=state.expert_credentials or "",
        exchanges=state.exchanges,
        grounded_questions=state.grounded_questions,
        ungrounded_questions=state.ungrounded_questions,
    )
    pretty_log(
        "dialogue_builder",
        "end",
        {"exchanges": len(dialogue.exchanges), "grounded_qa": len(dialogue.grounded_questions)},
    )
    return {"dialogue": dialogue}


def build_dialogue_subgraph():
    """Build and return the dialogue generation subgraph."""
    g = StateGraph(DialogueState)
    g.add_node("dialogue_planner", dialogue_planner)
    g.add_node("exchange_writer", exchange_writer)
    g.add_node("dialogue_join_exchanges", dialogue_join_exchanges)
    g.add_node("dialogue_qa_dispatch", dialogue_qa_dispatch)
    g.add_node("dialogue_grounded_qa_generator", dialogue_grounded_qa_generator)
    g.add_node("dialogue_ungrounded_qa_generator", dialogue_ungrounded_qa_generator)
    g.add_node("dialogue_join_qa", dialogue_join_qa)
    g.add_node("dialogue_builder", dialogue_builder)

    g.add_edge(START, "dialogue_planner")
    g.add_conditional_edges(
        "dialogue_planner", assign_exchange_writers, ["exchange_writer"]
    )
    g.add_edge("exchange_writer", "dialogue_join_exchanges")
    g.add_conditional_edges(
        "dialogue_qa_dispatch",
        route_dialogue_qa,
        ["dialogue_grounded_qa_generator", "dialogue_ungrounded_qa_generator"],
    )
    g.add_edge("dialogue_grounded_qa_generator", "dialogue_join_qa")
    g.add_edge("dialogue_ungrounded_qa_generator", "dialogue_join_qa")
    g.add_edge("dialogue_builder", END)
    return g.compile()
