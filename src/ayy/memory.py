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

    def __str__(self) -> str:
        return f"name: {self.name}\ncontent: {self.content}\nconfidence: {self.confidence}"


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

    def __str__(self) -> str:
        return f"{super().__str__()}\ncategory: {self.category}"


class EpisodicMemory(Memory):
    """Experiences and events tied to specific contexts"""

    type: Literal[MemoryType.EPISODIC] = MemoryType.EPISODIC
    context: str = Field(description="Context this memory is relevant to")

    def __str__(self) -> str:
        return f"{super().__str__()}\ncontext: {self.context}"


class Summary(BaseModel):
    bullet_points: list[str]
    semantic_memories: list[SemanticMemory] = Field(default_factory=list)
    episodic_memories: list[EpisodicMemory] = Field(default_factory=list)

    def summary_str(self, semantic: bool = True, episodic: bool = True) -> str:
        bullet_points = "\n- ".join(self.bullet_points)
        semantic_memories = "\n---\n".join([str(mem) for mem in self.semantic_memories]) if semantic else ""
        episodic_memories = "\n---\n".join([str(mem) for mem in self.episodic_memories]) if episodic else ""
        summ_str = f"<bullet_points>\n{bullet_points}\n</bullet_points>"
        if semantic:
            summ_str += f"\n<semantic_memories>\n{semantic_memories}\n</semantic_memories>"
        if episodic:
            summ_str += f"\n<episodic_memories>\n{episodic_memories}\n</episodic_memories>"
        return summ_str
