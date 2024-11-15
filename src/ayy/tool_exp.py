import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from instructor import AsyncInstructor, Instructor
from loguru import logger
from pydantic import BaseModel, Field, create_model

from ayy.dialog import Dialog, ModelName, assistant_message, create_creator, dialog_to_kwargs, user_message
from ayy.func_utils import function_to_model, get_function_info

MODEL_NAME = ModelName.GEMINI_FLASH

days = Literal["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class SelectedToolBase(BaseModel):
    chain_of_thought: str
    name: Literal["call_ai", "ask_user"]
    prompt: str = Field(
        ...,
        description="An LLM will receive the messages so far and the tools calls and results up until now. This prompt will then be used to ask the LLM to generate an input for the selected tool based on the tool's signature.",
    )


def get_weather(day: days, location: str) -> str:
    "get the weather at a day in a location"
    if day == "Monday" and location.lower() == "blackpool":
        return "It's raining"
    elif day == "Tuesday" and location.lower() == "london":
        return "It's sunny"
    else:
        return "It's overcast"


def list_available_grounds(location: str) -> str:
    "list all available grounds in a city"
    if location.lower() == "blackpool":
        return json.dumps(["The Hawthorns", "The Den", "The New Den"])
    elif location.lower() == "london":
        return json.dumps(["The Olympic Stadium", "The Emirates Stadium", "The Wembley Stadium"])
    else:
        return json.dumps(["The Stadium"])


def upload_video(video_path: str) -> str:
    "save video and return the dest path"
    return video_path


def call_ai(inputs: Any) -> Any:
    "Not a pre-defined tool."
    return inputs


def ask_user(prompt: str) -> str:
    "Prompt the user for a response"
    return prompt


def run_call_ai(creator: Instructor | AsyncInstructor, dialog: Dialog, tool: SelectedToolBase) -> Dialog:
    dialog.messages.append(user_message(content=tool.prompt))
    logger.info(f"\n\nCalling AI with messages: {messages}\n\n")
    res = creator.create(**dialog_to_kwargs(dialog=dialog), response_model=str)
    logger.success(f"call_ai result: {res}")
    dialog.messages.append(assistant_message(content=res))
    return dialog


def call_ask_user(dialog: Dialog, tool: SelectedToolBase) -> Dialog:
    res = input(f"{tool.prompt}\n> ")
    dialog.messages += [assistant_message(content=tool.prompt), user_message(content=res)]
    return dialog


def run_tool(
    creator: Instructor | AsyncInstructor, dialog: Dialog, tool: SelectedToolBase, tools_dict: dict
) -> Dialog:
    dialog.messages.append(user_message(content=tool.prompt))
    logger.info(f"\n\nCalling for {tool.name} with messages: {dialog.messages}\n\n")
    tool_args = creator.create(
        **dialog_to_kwargs(dialog=dialog), response_model=function_to_model(tools_dict[tool.name])
    )
    logger.info(f"{tool.name} args: {tool_args}")
    res = tools_dict[tool.name](**tool_args.model_dump())  # type: ignore
    logger.success(f"{tool.name} result: {res}")
    dialog.messages.append(assistant_message(content=res))
    return dialog


default_tools = [call_ai, ask_user]
tools = default_tools + [get_weather, list_available_grounds, upload_video]
tools_dict = {func.__name__: func for func in tools}
SelectedTool = create_model("SelectedTool", name=(Literal[*tools_dict.keys()], ...), __base__=SelectedToolBase)
tools_info = "\n\n".join([f"Tool {i}:\n{get_function_info(func)}" for i, func in enumerate(tools, start=1)])

tools_message = """
You have a list tools at your disposal. Each tool is a function with a signature and optional docstring.
Based on the user query, return a list of tools to use for the task. The tools will be used in that sequence.
You can assume that a tool would have access to the result of a previous tool call.
The list can also be empty. For each tool selection, return the tool name and a prompt for the LLM to generate an input for the selected tool based on the tool's signature. The LLM will receive the messages so far and the tools calls and results up until that point.
Remember the actual user query/task throughout your tool selection process. Especially when creating the prompt for the LLM.
More often than not, the last tool would be 'call_ai' to generate the final response.
Pay close attention to the information you do have and the information you do not have. If you don't have the information and you think the user should know, ask the user for it. Don't make assumptions.

When to use 'ask_user':
    - To ask the user something.
    - When the right tool is obvious but you need some extra data based on the function signature, then select the ask_user tool before selecting the actual tool and in your prompt explicitly state the extra information you need. So the tool should still be selected, you're just pairing it with a prior ask_user call.
    - A tool may have some default values. But if you think they should be provided by the user for the current task, ask for them.

Don't use the 'ask_user' tool to ask the user to fix obvious typos, do that yourself. That's a safe assumption. People make typos all the time.

When to use 'call_ai':
    - Whenever you want. In between or at the start/end.
    - To get the AI to generate something. This could be the final response or something in between tool calls.
    - To extract information before the next tool call.

<tools>

{tools_info}

</tools>
"""

creator = create_creator(model_name=MODEL_NAME)

messages = [user_message("I want to buy a new mac")]
selector_dialog = Dialog(
    system=tools_message.format(tools_info=tools_info), messages=messages, model_name=MODEL_NAME
)
selected_path = Path("selected_tools.json")
if not selected_path.exists():
    selected_tools = creator.create(**dialog_to_kwargs(dialog=selector_dialog), response_model=list[SelectedTool])
    logger.success(f"Selected tools: {selected_tools}")
    selected_path.write_text(json.dumps([selection.model_dump() for selection in selected_tools], indent=2))  # type:ignore
else:
    selected_tools = [SelectedTool(**tool) for tool in json.loads(selected_path.read_text())]

selected_tools = selected_tools or [
    SelectedTool(chain_of_thought="", name="call_ai", prompt="Generate a response")
]
runner_dialog = Dialog(messages=deepcopy(messages), model_name=MODEL_NAME)
for tool in selected_tools:  # type:ignore
    if tool.name == "ask_user":
        runner_dialog = call_ask_user(dialog=runner_dialog, tool=tool)
    elif tool.name == "call_ai":
        runner_dialog = run_call_ai(creator=creator, dialog=runner_dialog, tool=tool)
    else:
        runner_dialog = run_tool(creator=creator, dialog=runner_dialog, tool=tool, tools_dict=tools_dict)


if not any(x.name for x in selected_tools if x.name not in default_tools):  # type:ignore
    seq = 0 if selected_tools[-1].name != "ask_user" else 1  # type:ignore
    while True:
        if seq % 2 == 0:
            user_input = input("('q' or 'exit' or 'quit' to quit) > ")
            if user_input.lower() in ["q", "exit", "quit"]:
                break
            runner_dialog.messages.append(user_message(content=user_input))
        else:
            res = creator.create(**dialog_to_kwargs(dialog=runner_dialog), response_model=str)
            logger.success(f"call_ai result: {res}")
            runner_dialog.messages.append(assistant_message(content=res))
        seq += 1

logger.success(f"Messages: {messages}")
