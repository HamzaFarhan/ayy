from ayy.agent import Agent, ModelName
from ayy.prompts import NAME_AGENT, SUMMARIZE_MESSAGES

AGENT_NAMER_AGENT = Agent(
    model_name=ModelName.GEMINI_FLASH, system=NAME_AGENT, name="agent_namer_agent", available_tools=["call_ai"]
)

SUMMARIZER_AGENT = Agent(
    model_name=ModelName.GEMINI_FLASH,
    system=SUMMARIZE_MESSAGES,
    name="summarizer_agent",
    available_tools=["call_ai"],
)
