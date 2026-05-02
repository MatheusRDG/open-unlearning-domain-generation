"""State classes for LangGraph graphs."""

from operator import add
from typing import Annotated, List, Optional

from pydantic import BaseModel, Field

from src.domain_generation.models import (Article, Book, Chapter, Dialogue, Domain,
                                          Exchange, Poem, QAItem, Section, Stanza,
                                          TOCEntry, Topic)


class DomainState(BaseModel):
    """State for the main domain graph."""

    name: str = Field(description="Domain name")
    description: str = Field(default="", description="Domain description")
    topics: Annotated[List[Topic], add] = []
    books: Annotated[List[Book], add] = []
    articles: Annotated[List[Article], add] = []
    poems: Annotated[List[Poem], add] = []
    dialogues: Annotated[List[Dialogue], add] = []
    pending_books: Annotated[int, add] = 0
    pending_articles: Annotated[int, add] = 0
    pending_poems: Annotated[int, add] = 0
    pending_dialogues: Annotated[int, add] = 0
    domain: Optional[Domain] = None


class BookState(BaseModel):
    """State for the book subgraph."""

    domain_name: str = Field(description="Domain name")
    topic: str = Field(description="Topic this book covers")
    topic_description: str = Field(default="", description="Topic description")
    title: Optional[str] = None
    table_of_contents: Annotated[List[TOCEntry], add] = []
    chapters: Annotated[List[Chapter], add] = []
    grounded_questions: Annotated[List[QAItem], add] = []
    ungrounded_questions: Annotated[List[QAItem], add] = []
    pending_chapters: Annotated[int, add] = 0
    pending_qa_tasks: Annotated[int, add] = 0
    book: Optional[Book] = None


class ChapterWriterState(BaseModel):
    """State for individual chapter writing tasks."""

    domain_name: str = Field(description="Domain name")
    topic: str = Field(description="Topic")
    chapter_title: str = Field(description="Title of this chapter")
    summary: str = Field(description="Chapter summary from TOC")
    chapter_titles: List[str] = Field(description="All chapter titles for context")
    idx: int = Field(description="Chapter index (1-based)")


class ArticleState(BaseModel):
    """State for the article subgraph."""

    domain_name: str = Field(description="Domain name")
    topic: str = Field(description="Topic this article covers")
    topic_description: str = Field(default="", description="Topic description")
    title: Optional[str] = None
    abstract: Optional[str] = None
    section_names: List[str] = []
    sections: Annotated[List[Section], add] = []
    grounded_questions: Annotated[List[QAItem], add] = []
    ungrounded_questions: Annotated[List[QAItem], add] = []
    pending_sections: Annotated[int, add] = 0
    pending_qa_tasks: Annotated[int, add] = 0
    article: Optional[Article] = None


class ArticleSectionWriterState(BaseModel):
    """State for individual article section writing tasks."""

    domain_name: str = Field(description="Domain name")
    topic: str = Field(description="Topic")
    title: str = Field(description="Article title")
    abstract: str = Field(description="Article abstract")
    section_name: str = Field(description="Name of this section")
    section_idx: int = Field(description="Section index (1-based)")
    total_sections: int = Field(description="Total number of sections")


class PoemState(BaseModel):
    """State for the poem subgraph."""

    domain_name: str = Field(description="Domain name")
    topic: str = Field(description="Topic this poem covers")
    topic_description: str = Field(default="", description="Topic description")
    title: Optional[str] = None
    theme: Optional[str] = None
    stanza_labels: List[str] = []
    stanzas: Annotated[List[Stanza], add] = []
    grounded_questions: Annotated[List[QAItem], add] = []
    ungrounded_questions: Annotated[List[QAItem], add] = []
    pending_stanzas: Annotated[int, add] = 0
    pending_qa_tasks: Annotated[int, add] = 0
    poem: Optional[Poem] = None


class StanzaWriterState(BaseModel):
    """State for individual stanza writing tasks."""

    domain_name: str = Field(description="Domain name")
    topic: str = Field(description="Topic")
    poem_title: str = Field(description="Poem title")
    theme: str = Field(description="Poem theme")
    stanza_label: str = Field(description="Label/heading of this stanza")
    stanza_idx: int = Field(description="Stanza index (1-based)")
    total_stanzas: int = Field(description="Total number of stanzas")
    all_labels: List[str] = Field(description="All stanza labels for context")


class DialogueState(BaseModel):
    """State for the dialogue subgraph."""

    domain_name: str = Field(description="Domain name")
    topic: str = Field(description="Topic this dialogue covers")
    topic_description: str = Field(default="", description="Topic description")
    title: Optional[str] = None
    expert_name: Optional[str] = None
    expert_credentials: Optional[str] = None
    exchange_topics: List[str] = []
    exchanges: Annotated[List[Exchange], add] = []
    grounded_questions: Annotated[List[QAItem], add] = []
    ungrounded_questions: Annotated[List[QAItem], add] = []
    pending_exchanges: Annotated[int, add] = 0
    pending_qa_tasks: Annotated[int, add] = 0
    dialogue: Optional[Dialogue] = None


class ExchangeWriterState(BaseModel):
    """State for individual exchange writing tasks."""

    domain_name: str = Field(description="Domain name")
    topic: str = Field(description="Topic")
    dialogue_title: str = Field(description="Dialogue title")
    expert_name: str = Field(description="Expert name")
    expert_credentials: str = Field(description="Expert credentials")
    exchange_topic: str = Field(description="Focus of this exchange")
    exchange_idx: int = Field(description="Exchange index (1-based)")
    total_exchanges: int = Field(description="Total number of exchanges")
