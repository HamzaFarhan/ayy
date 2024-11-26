from enum import StrEnum

from pydantic import BaseModel


class MemoryTagInfo(BaseModel):
    description: str
    use_cases: list[str]


class MemoryTag(StrEnum):
    CORE = "core"
    RECALL = "recall"
    TEMPORARY = "temporary"
    CONTEXT = "context"
    ACTION = "action"
    FEEDBACK = "feedback"
    ERROR = "error"


MEMORY_TAG_INFO = {
    MemoryTag.CORE: MemoryTagInfo(
        description="Messages that are crucial and should persist even after the dialog concludes",
        use_cases=["Important facts about the user", "Long-term preferences", "Critical instructions or rules"],
    ),
    MemoryTag.RECALL: MemoryTagInfo(
        description="Messages relevant to the current task and should be remembered during the ongoing dialog",
        use_cases=["Current task parameters", "Intermediate results", "Temporary user preferences"],
    ),
    MemoryTag.TEMPORARY: MemoryTagInfo(
        description="Messages that are only needed for a short duration and can be discarded afterward",
        use_cases=["Clarification questions", "Intermediate calculations", "Status updates"],
    ),
    MemoryTag.CONTEXT: MemoryTagInfo(
        description="Provides background information that aids in understanding the conversation",
        use_cases=["Environmental details", "Previous interaction summaries", "Relevant historical data"],
    ),
    MemoryTag.ACTION: MemoryTagInfo(
        description="Represents actions taken or to be taken",
        use_cases=["Tool invocations", "Command executions", "System operations"],
    ),
    MemoryTag.FEEDBACK: MemoryTagInfo(
        description="User or system feedback intended to improve future interactions",
        use_cases=["User corrections", "Quality ratings", "Improvement suggestions"],
    ),
    MemoryTag.ERROR: MemoryTagInfo(
        description="Records of errors or issues encountered during the dialog",
        use_cases=["Exception messages", "Failed operations", "System warnings"],
    ),
}
