from ayy.dialog import Dialog, MemoryTag, ModelName, exchange
from ayy.prompts import NAME_DIALOG, SUMMARIZE_MESSAGES

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

SUMMARIZER_DIALOG = Dialog(
    model_name=ModelName.GEMINI_FLASH,
    system=SUMMARIZE_MESSAGES,
    messages=[
        *exchange(user="My name is Hamza", assistant="Hey Hamza, how can I help you today?"),
        *exchange(
            user="I'm planning a trip to visit my family in New York. How's the weather there?",
            assistant="Let me check the weather for you. It looks like it's currently sunny in New York with temperatures around 75°F (24°C). The forecast shows similar weather for the next few days.",
        ),
        *exchange(
            user="I love sunny days! That's perfect for my trip.",
            assistant="That's great that the weather aligns with your preferences! Would you like me to help you book a flight to New York?",
        ),
        *exchange(
            user="Yes please, I'd prefer a morning flight if possible",
            assistant="I'll look for morning flights. Do you have any specific dates in mind for your travel?",
        ),
        *exchange(
            user="Next weekend would be ideal",
            assistant="I found several morning flights for next weekend. There's a direct flight on Saturday at 8:30 AM, arriving at 11:45 AM. Would that work for you?",
        ),
        *exchange(
            user="That sounds perfect",
            assistant="Great! I've booked the flight for you. Here are the details:\n- Flight AA1234\n- Saturday, 8:30 AM departure\n- Direct flight to JFK\n- Confirmation code: XYZ789",
        ),
        *exchange(
            user="Thanks! Could you remind me what time I arrive?",
            assistant="Your flight arrives at JFK at 11:45 AM. Would you like me to help arrange transportation from the airport?",
        ),
        *exchange(
            user="No thanks, my family will pick me up",
            assistant="Perfect! Let me know if you need anything else regarding your trip.",
        ),
        *exchange(
            user="Summarize the conversation so far.",
            assistant="""Summary: {
                "bullet_points": [
                    "Trip planning initiated to visit family in New York",
                    "Flight booked: Saturday morning direct flight to JFK",
                    "Airport pickup arranged with family"
                ],
                "core_information": [
                    {
                        "name": "personal_info",
                        "content": "User's name is Hamza",
                        "temporary": False
                    },
                    {
                        "name": "family_info",
                        "content": "Has family in New York who will provide airport pickup",
                        "temporary": False
                    },
                    {
                        "name": "preference",
                        "content": "Prefers sunny weather and morning flights",
                        "temporary": False
                    }
                ],
                "temporary_information": [
                    {
                        "name": "flight_details",
                        "content": {
                            "flight_number": "AA1234",
                            "departure": "8:30 AM Saturday",
                            "arrival": "11:45 AM",
                            "confirmation": "XYZ789"
                        },
                        "temporary": True
                    }
                ]
            }""",
        ),
    ],
    name="summarizer_dialog",
)
