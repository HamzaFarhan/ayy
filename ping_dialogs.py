from ayy.dialog import Dialog, DialogToolSignature, ModelName
from ping_prompts import (
    ASSET_ADVISOR_PROMPT,
    COMPLIANCE_OFFICER_PROMPT,
    CUSTOMER_SUPPORT_PROMPT,
    HUMAN_ADVISOR_PROMPT,
    INVESTOR_ASSISTANT_PROMPT,
    RISK_ANALYST_PROMPT,
)

MODEL_NAME = ModelName.GEMINI_FLASH

INVESTOR_ASSISTANT_DIALOG = Dialog(
    model_name=MODEL_NAME,
    dialog_tool_signature=DialogToolSignature(
        name="investor_assistant",
        signature="investor_assistant(task_query: str)",
        docstring="The primary assistant for the investor. Serves as the central point of contact for all investor tasks.",
        system=INVESTOR_ASSISTANT_PROMPT,
    ).model_dump(),
)

CUSTOMER_SUPPORT_DIALOG = Dialog(
    model_name=MODEL_NAME,
    dialog_tool_signature=DialogToolSignature(
        name="customer_support_assistant",
        signature="customer_support_assistant(task_query: str)",
        docstring="The customer support assistant for the investor. Provides exceptional customer service and maintains positive investor relationships.",
        system=CUSTOMER_SUPPORT_PROMPT,
    ).model_dump(),
)

ASSET_ADVISOR_DIALOG = Dialog(
    model_name=MODEL_NAME,
    dialog_tool_signature=DialogToolSignature(
        name="asset_advisor",
        signature="asset_advisor(task_query: str)",
        docstring="The asset advisor for the investor. Provides expert guidance on asset classes and investment opportunities.",
        system=ASSET_ADVISOR_PROMPT,
    ).model_dump(),
)

RISK_ANALYST_DIALOG = Dialog(
    model_name=MODEL_NAME,
    dialog_tool_signature=DialogToolSignature(
        name="risk_analyst",
        signature="risk_analyst(task_query: str)",
        docstring="The risk analyst specializing in investment risk assessment and management. Evaluates and communicates investment risks clearly to help investors make informed decisions.",
        system=RISK_ANALYST_PROMPT,
    ).model_dump(),
)

COMPLIANCE_OFFICER_DIALOG = Dialog(
    model_name=MODEL_NAME,
    dialog_tool_signature=DialogToolSignature(
        name="compliance_officer",
        signature="compliance_officer(task_query: str)",
        docstring="The compliance officer responsible for ensuring all investment recommendations and activities adhere to regulatory requirements and ethical standards.",
        system=COMPLIANCE_OFFICER_PROMPT,
    ).model_dump(),
)


HUMAN_ADVISOR_DIALOG = Dialog(
    model_name=MODEL_NAME,
    dialog_tool_signature=DialogToolSignature(
        name="human_advisor",
        signature="human_advisor(task_query: str)",
        docstring="The human advisor responsible for handling high-stakes investment decisions and complex cases. Provides personalized guidance for significant investment decisions.",
        system=HUMAN_ADVISOR_PROMPT,
    ).model_dump(),
)
