from ayy.dialog import Dialog, MemoryTag, ModelName, exchange
from ayy.prompts import NAME_DIALOG

DIALOG_NAMER_DIALOG = Dialog(model_name=ModelName.GEMINI_FLASH, system=NAME_DIALOG, name="dialog_namer_dialog")

MEMORY_TAGGER_DIALOG = Dialog(
    model_name=ModelName.GEMINI_FLASH,
    system=f"Tag the latest message. Possible tags are {str(MemoryTag.__members__)}",
    messages=[
        *exchange(
            user="it's sunny today",
            assistant="reasoning: This is current weather information that will change. memory_tags: ['RECALL']",
        ),
        *exchange(
            user="I love sunny days",
            assistant="reasoning: This expresses a general preference which is a permanent trait. memory_tags: ['CORE']",
        ),
        *exchange(
            user="My name is Hamza",
            assistant="reasoning: This is temporary identifying information. memory_tags: ['RECALL']",
            feedback="My name is a permanent thing. The tag for permanent things should be CORE",
            correction="reasoning: You're right - a name is permanent identifying information. Apologies, I made a mistake. memory_tags: ['CORE']",
        ),
        *exchange(
            user="I'm going to the store",
            assistant="reasoning: This seems like a permanent activity. memory_tags: ['CORE']",
            feedback="Going to the store is a temporary activity, not a permanent fact. It should be RECALL",
            correction="reasoning: You're correct - this is a temporary activity. You're right, I apologize. memory_tags: ['RECALL']",
        ),
        *exchange(
            user="I'm planning a trip to visit my family in New York",
            assistant="reasoning: The trip is temporary but having family in New York is permanent information. memory_tags: ['RECALL', 'CORE']",
        ),
    ],
    name="memory_tagger_dialog",
)
