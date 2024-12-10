from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class MemoryType(StrEnum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"


class Memory(BaseModel):
    name: str
    content: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


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


class SemanticMemory(Memory):
    """Facts and knowledge that persist across conversations"""

    type: Literal[MemoryType.SEMANTIC] = MemoryType.SEMANTIC
    category: SemanticCategory


class EpisodicMemory(Memory):
    """Experiences and events tied to specific contexts"""

    type: Literal[MemoryType.EPISODIC] = MemoryType.EPISODIC
    context: str = Field(description="Context this memory is relevant to")


class Summary(BaseModel):
    bullet_points: list[str]
    semantic_memories: list[SemanticMemory] = Field(default_factory=list)
    episodic_memories: list[EpisodicMemory] = Field(default_factory=list)
