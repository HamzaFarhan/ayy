from ayy.dialog import Dialog, ModelName
from ayy.prompts import NAME_AGENT, SUMMARIZE_MESSAGES

DIALOG_NAMER_DIALOG = Dialog(
    model_name=ModelName.GEMINI_FLASH, system=NAME_AGENT, name="dialog_namer_dialog", available_tools=["call_ai"]
)

SUMMARIZER_DIALOG = Dialog(
    model_name=ModelName.GEMINI_FLASH,
    system=SUMMARIZE_MESSAGES,
    name="summarizer_dialog",
    available_tools=["call_ai"],
)
