from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class SemanticCategory(StrEnum):
    IDENTITY = "identity"
    PREFERENCES = "preferences"
    RELATIONSHIPS = "relationships"
    SKILLS = "skills"
    BELIEFS = "beliefs"
    BACKGROUND = "background"
    HEALTH = "health"
    LOCATION = "location"
    SCHEDULE = "schedule"
    GOALS = "goals"
    OTHER = "other"


class SemanticMemory(BaseModel):
    name: str
    content: str
    category: SemanticCategory
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    def __str__(self) -> str:
        return (
            f"name: {self.name}\ncontent: {self.content}\ncategory: {self.category}\nconfidence: {self.confidence}"
        )


class Message(BaseModel):
    """A chat message with role and content"""

    role: Literal["user", "assistant"]
    content: str


class Summary(BaseModel):
    """Summarized version of a conversation"""

    messages: list[Message]
    semantic_memories: list[SemanticMemory] = Field(default_factory=list)

    def summary_str(self, semantic: bool = True) -> str:
        messages_str = "\n".join([f"{msg.role}: {msg.content}" for msg in self.messages])
        semantic_memories = "\n---\n".join([str(mem) for mem in self.semantic_memories]) if semantic else ""
        summ_str = f"<messages>\n{messages_str}\n</messages>"
        if semantic:
            summ_str += f"\n<semantic_memories>\n{semantic_memories}\n</semantic_memories>"
        return summ_str


def summary_to_messages(summary: Summary) -> list[dict[str, Any]]:
    return summary.model_dump().get("messages", [])
