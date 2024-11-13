from functools import partial

import instructor
from burr.core import ApplicationBuilder
from burr.integrations.pydantic import PydanticTypingSystem
from burr.tracking import LocalTrackingClient
from google.generativeai import GenerativeModel
from loguru import logger

from ayy.dialog import Dialog, ModelName, add_assistant_message, add_user_message, assistant_message, user_message

MODEL = ModelName.GEMINI_FLASH

creator = partial(
    instructor.from_gemini(client=GenerativeModel(model_name=MODEL), mode=instructor.Mode.GEMINI_JSON).create,
    response_model=str,
)
tracker = LocalTrackingClient(project="EXP")
app = (
    ApplicationBuilder()
    .with_identifiers(app_id="exp1")
    .with_actions(
        add_user_message.bind(template=""),  # type:ignore
        add_assistant_message.bind(creator=creator),  # type:ignore
    )
    .with_typing(PydanticTypingSystem(Dialog))
    .with_state(
        Dialog(
            system="Talk like a pirate.",
            messages=[user_message("thoughts on messi?"), assistant_message("arr! he be amazing cap'n")],
            model_name=MODEL,
        )
    )
    .with_entrypoint("add_user_message")
    .with_transitions(("add_user_message", "add_assistant_message"), ("add_assistant_message", "add_user_message"))
    .with_tracker(tracker=tracker)  # type:ignore
    .build()
)
last_action = "add_assistant_message"
user_input = ""
while True:
    if last_action == "add_assistant_message":
        user_input = input("> ")
        if user_input.lower() == "q":
            break
    step_result = app.step(inputs={"content": user_input})
    if step_result is None:
        logger.error("app.step returned None")
        break
    last_action = step_result[0].name
