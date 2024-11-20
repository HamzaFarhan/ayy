from datetime import datetime
from functools import partial

import instructor
from google.generativeai import GenerativeModel
from loguru import logger
from sqlmodel import Field, Relationship, Session, SQLModel, create_engine

from ayy.dialog import Content, ModelName, assistant_message, chat_message, messages_to_kwargs, user_message

MODEL = ModelName.GEMINI_FLASH


class Dialog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    system: str = ""
    model_name: ModelName = MODEL
    messages: list["Message"] = Relationship(back_populates="dialog")


class Message(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    dialog_id: int = Field(foreign_key="dialog.id")
    role: str
    content: str
    created_at: datetime = Field(default_factory=datetime.now)
    dialog: Dialog = Relationship(back_populates="messages")


def add_assistant_message(dialog: Dialog, creator: Content) -> Dialog:
    try:
        res = (
            creator(
                **messages_to_kwargs(
                    messages=[chat_message(role=msg.role, content=msg.content) for msg in dialog.messages],
                    system=dialog.system,
                    model_name=dialog.model_name,
                )
            )
            if callable(creator)
            else creator
        )
    except Exception as e:
        logger.exception(f"Error in respond. Last message: {dialog.messages[-1]}")
        res = f"Error: {e}"
    dialog.messages.append(Message(**assistant_message(content=res)))
    dialog.updated_at = datetime.now()
    return dialog


def add_user_message(dialog: Dialog, content: Content, template: Content = "") -> Dialog:
    dialog.messages.append(Message(**user_message(content=content, template=template)))
    dialog.updated_at = datetime.now()
    return dialog


creator = partial(
    instructor.from_gemini(client=GenerativeModel(model_name=MODEL), mode=instructor.Mode.GEMINI_JSON).create,
    response_model=str,
)

sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

engine = create_engine(sqlite_url, echo=True)


SQLModel.metadata.create_all(engine)

dialog = Dialog(
    system="Talk like a pirate.",
    messages=[
        Message(**user_message("thoughts on messi?")),
        Message(**assistant_message("arr! he be amazing cap'n")),
    ],
)

with Session(engine) as session:
    seq = 0
    while True:
        session.add(dialog)
        session.commit()
        session.refresh(dialog)
        if seq % 2 == 0:
            user_input = input("> ")
            if user_input.lower() == "q":
                break
            dialog = add_user_message(dialog=dialog, content=user_input)
        else:
            dialog = add_assistant_message(dialog=dialog, creator=creator)
        seq += 1
