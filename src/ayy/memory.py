from enum import Enum
from typing import Literal

from pydantic import BaseModel

from ayy.dialog import ModelName, create_creator, system_message, user_message


class MemoryTagInfo(BaseModel):
    description: str
    use_cases: list[str]


class MemoryTag(Enum):
    CORE = MemoryTagInfo(
        description="Messages that are crucial and should persist even after the dialog concludes",
        use_cases=[
            "Important facts about the user",
            "Long-term preferences",
            "Critical instructions or rules",
            "User feedback intended to improve future interactions",
        ],
    )
    RECALL = MemoryTagInfo(
        description="Messages relevant to the current task and should be remembered during the ongoing dialog",
        use_cases=["Current task parameters", "Intermediate results", "Temporary user preferences"],
    )


creator = create_creator(ModelName.GEMINI_FLASH)
res = creator.create(
    response_model=Literal[*MemoryTag._member_names_],  # type: ignore
    messages=[
        system_message(f"Possible tags are {str(MemoryTag.__members__)}"),
        user_message("My name is Hamza.not that important"),
    ],  # type: ignore
)

print(res)
