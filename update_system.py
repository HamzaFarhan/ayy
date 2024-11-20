import json
from functools import partial
from pathlib import Path
from typing import Any

import instructor
import pandas as pd
from google.generativeai import GenerativeModel
from loguru import logger
from pydantic import BaseModel

from ayy.dialog import Dialog, ModelName, messages_to_kwargs, user_message

MODEL = ModelName.GEMINI_FLASH


class UpdatedInstructions(BaseModel):
    instructions: str
    reasoning: str


def jsonl_to_json(src: str | Path) -> Any:
    try:
        json_obj = pd.read_json(path_or_buf=Path(src).with_suffix(".jsonl"), lines=True)
        return json.loads(str(json_obj.to_json(indent=2, orient="records")))
    except Exception:
        logger.exception(f"Could not load {src}")
        return []


def add_anns_to_dialog(project_dir: str | Path, app_id: str) -> Dialog:
    project_dir = Path(project_dir)
    min_log = None
    max_log = None
    initial_state = Dialog()
    final_state = Dialog()
    for log in jsonl_to_json(src=project_dir / app_id / "log.jsonl"):
        if log["type"] == "end_entry":
            if min_log is None or log["sequence_id"] < min_log["sequence_id"]:
                initial_state = Dialog(**log["state"])
                min_log = log
            if max_log is None or log["sequence_id"] > max_log["sequence_id"]:
                final_state = Dialog(**log["state"])
                max_log = log
    # print(initial_state.model_dump())
    initial_num_messages = len(initial_state.messages)
    logger.info(f"Initial num messages: {initial_num_messages}")
    num_added = 0
    for ann in jsonl_to_json(src=project_dir / "annotations.jsonl"):
        if ann["app_id"] == app_id:
            for obs in ann["observations"]:
                thumbs = obs.pop("thumbs_up_thumbs_down", None)
                obs.update(
                    {
                        "thumbs_up": bool(thumbs) if thumbs is not None else False,
                        "thumbs_down": not bool(thumbs) if thumbs is not None else False,
                    }
                )
            final_state.messages.insert(
                ann["step_sequence_id"] + initial_num_messages + num_added,
                user_message(
                    content=f"Some user feedback:\n\n<observations>\n{json.dumps(ann['observations'])}\n</observations>"
                ),
            )
            num_added += 1

    return final_state


dialog = add_anns_to_dialog(project_dir="~/.burr/EXP", app_id="exp1")
creator = partial(
    instructor.from_gemini(client=GenerativeModel(model_name=MODEL), mode=instructor.Mode.GEMINI_JSON).create,
    response_model=UpdatedInstructions,
)

update_prompt = f"""
These were the initial system instructions:
{dialog.system}

Based on the chat history so far and user feedback (if any),\
give me an updated version of the system instructions along with your reasoning. Don't mention the chat history in the updated instructions. Make the instructions as detailed and guided as possible. The user feedback is very important and should be incorporated into the instructions if it's given.
"""
dialog.messages.append(user_message(content=update_prompt))
Path("exp1_dialog.json").write_text(json.dumps(dialog.model_dump(), indent=2))
print(messages_to_kwargs(messages=dialog.messages))
updated_system = creator(**messages_to_kwargs(messages=dialog.messages))
logger.success(f"\nUpdated instructions:\n{updated_system.instructions}\n\nReasoning:\n{updated_system.reasoning}")  # type:ignore
