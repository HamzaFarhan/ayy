from collections import deque
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal

from instructor import AsyncInstructor, Instructor
from loguru import logger
from pydantic import BaseModel, Field, create_model

from ayy.dialog import Dialog, ModelName, assistant_message, create_creator, dialog_to_kwargs, user_message
from ayy.func_utils import function_to_type, get_function_info

MODEL_NAME = ModelName.GEMINI_FLASH
DEFAULT_PROMPT = "Generate a response if you've been asked. Otherwise, ask the user how they are doing."

days = Literal["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class Tool(BaseModel):
    chain_of_thought: str
    name: str
    prompt: str = Field(
        ...,
        description="An LLM will receive the messages so far and the tools calls and results up until now. This prompt will then be used to ask the LLM to generate arguments for the selected tool based on the tool's signature. If the tool doesn't have any parameters, then it doesn't need a prompt.",
    )


DEFAULT_TOOL = Tool(chain_of_thought="", name="call_ai", prompt=DEFAULT_PROMPT)


def get_weather(day: days, location: str) -> str:
    "get the weather at a day in a location"
    if day == "Monday" and location.lower() == "blackpool":
        return "It's raining"
    elif day == "Tuesday" and location.lower() == "london":
        return "It's sunny"
    else:
        return "It's overcast"


def list_available_grounds(location: str) -> list[str]:
    "list all available grounds in a city"
    if location.lower() == "blackpool":
        return ["The Hawthorns", "The Den", "The New Den"]
    elif location.lower() == "london":
        return ["The Olympic Stadium", "The Emirates Stadium", "The Wembley Stadium"]
    else:
        return ["The Stadium"]


def upload_video(video_path: str) -> str:
    "save video and return the dest path"
    return video_path


def call_ai(inputs: Any) -> Any:
    "Not a pre-defined tool."
    return inputs


def ask_user(prompt: str) -> str:
    "Prompt the user for a response"
    return prompt


def run_call_ai(creator: Instructor | AsyncInstructor, dialog: Dialog, tool: Tool) -> Dialog:
    tool.prompt = tool.prompt or DEFAULT_PROMPT
    dialog.messages.append(user_message(content=tool.prompt))
    logger.info(f"\n\nCalling AI with messages: {dialog.messages}\n\n")
    res = creator.create(**dialog_to_kwargs(dialog=dialog), response_model=str)
    logger.success(f"call_ai result: {res}")
    dialog.messages.append(assistant_message(content=res))
    return dialog


def call_ask_user(dialog: Dialog, tool: Tool) -> Dialog:
    tool.prompt = tool.prompt or DEFAULT_PROMPT
    res = input(f"{tool.prompt}\n> ")
    dialog.messages += [assistant_message(content=tool.prompt), user_message(content=res)]
    return dialog


def run_tool(
    creator: Instructor | AsyncInstructor,
    dialog: Dialog,
    tool: Tool,
    ignore_default_values: bool = False,
    skip_default_params: bool = False,
) -> Dialog:
    global tool_queue, tool_dict
    if tool.prompt:
        dialog.messages.append(user_message(content=tool.prompt))
    logger.info(f"\n\nCalling for {tool.name} with messages: {dialog.messages}\n\n")
    creator_res = creator.create(
        **dialog_to_kwargs(dialog=dialog),
        response_model=tool_dict[tool.name].get(
            "type",
            function_to_type(
                func=tool_dict[tool.name]["func"],
                ignore_default_values=ignore_default_values,
                skip_default_params=skip_default_params,
            ),
        ),
    )  # type: ignore
    logger.info(f"{tool.name} creator_res: {creator_res}")
    selected_tool = tool_dict[tool.name]["func"]
    if isinstance(creator_res, BaseModel):
        res = selected_tool(**creator_res.model_dump())
    else:
        res = selected_tool(creator_res)
    logger.success(f"{tool.name} result: {res}")
    if isinstance(res, Dialog):
        return res
    dialog.messages.append(assistant_message(content=str(res)))
    return dialog


def run_selected_tool(creator: Instructor | AsyncInstructor, dialog: Dialog, tool: Tool) -> Dialog:
    global tool_dict
    if tool.name.lower() == "ask_user":
        dialog = call_ask_user(dialog=dialog, tool=tool)
    elif tool.name.lower() == "call_ai":
        dialog = run_call_ai(creator=creator, dialog=dialog, tool=tool)
    else:
        dialog = run_tool(creator=creator, dialog=dialog, tool=tool)
    return dialog


def run_next_tool(creator: Instructor | AsyncInstructor, dialog: Dialog) -> Dialog:
    global tool_queue, tool_dict
    if tool_queue:
        tool = tool_queue.popleft()
        dialog = run_selected_tool(creator=creator, dialog=dialog, tool=tool)
    return dialog


def run_tools(creator: Instructor | AsyncInstructor, dialog: Dialog, continue_dialog:bool=True) -> Dialog:
    global tool_queue, tool_dict
    non_default_tools_used = set()
    tool_queue = deque(tool_queue) if tool_queue else deque([DEFAULT_TOOL])
    current_tool_name = DEFAULT_TOOL.name
    while tool_queue:
        print(f"\nTOOL QUEUE: {tool_queue}\n")
        print(f"\nTOOL DICT: {tool_dict}\n")
        current_tool = tool_queue.popleft()
        if not isinstance(current_tool, Tool) and callable(current_tool):
            current_tool_name = (
                current_tool.__name__ if not isinstance(current_tool, partial) else current_tool.func.__name__
            )
            res = current_tool()
            if res:
                if isinstance(res, Dialog):
                    dialog = res
                else:
                    dialog.messages.append(assistant_message(content=str(res)))
            continue
        current_tool_name = current_tool.name
        dialog = run_selected_tool(creator=creator, dialog=dialog, tool=current_tool)
        if current_tool_name not in [func.__name__ for func in DEFAULT_TOOLS]:
            non_default_tools_used.add(current_tool.name)
    if continue_dialog:
        seq = int(current_tool_name == "ask_user")
        while True:
            if seq % 2 == 0:
                user_input = input("('q' or 'exit' or 'quit' to quit) > ")
                if user_input.lower() in ["q", "exit", "quit"]:
                    break
                dialog.messages.append(user_message(content=user_input))
            else:
                res = creator.create(**dialog_to_kwargs(dialog=dialog), response_model=str)
                logger.success(f"ai response: {res}")
                dialog.messages.append(assistant_message(content=res))
            seq += 1

    logger.success(f"Messages: {dialog.messages}")
    return dialog


def get_selected_tools(selected_tools: list[Tool]):
    "Get a list of selected tools for the task"
    global tool_queue
    tool_queue.extendleft(selected_tools[::-1])


def add_new_tools(new_tools: set[Callable] | list[Callable]):
    global tool_dict
    # Prob a database or redis thing. global for now
    for func in new_tools:
        tool_dict[func.__name__] = {"info": get_function_info(func), "func": func}
    tool_dict["get_selected_tools"] = {
        "info": get_function_info(get_selected_tools),
        "func": get_selected_tools,
        "type": list[create_model("SelectedTool", name=(Literal[*tool_dict.keys()], ...), __base__=Tool)],
    }


def new_task(dialog: Dialog, task: str, available_tools: list[str] | None = None) -> Dialog:
    global tool_dict, tool_queue
    tools_info = "\n\n".join(
        [
            f"Tool {i}:\n{tool_dict[tool]['info']}"
            for i, tool in enumerate(available_tools or tool_dict.keys(), start=1)
        ]
    )
    dialog.messages += [user_message(content=Path("selector_task.txt").read_text().format(tools_info=tools_info))]
    tool_queue.appendleft(Tool(chain_of_thought="", name="get_selected_tools", prompt=task))
    return dialog


DEFAULT_TOOLS = {call_ai, ask_user}
tool_dict = {func.__name__: {"info": get_function_info(func), "func": func} for func in DEFAULT_TOOLS}
tool_queue = deque()

creator = create_creator(model_name=MODEL_NAME)
dialog = Dialog(model_name=MODEL_NAME)


tool_queue.extendleft(
    [
        partial(add_new_tools, new_tools=[get_weather, list_available_grounds, upload_video]),
        partial(
            new_task,
            dialog=dialog,
            task="hello",
        ),
        ][::-1]
)
runner_dialog = run_tools(creator=creator, dialog=dialog)
