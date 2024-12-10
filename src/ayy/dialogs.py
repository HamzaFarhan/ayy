from ayy.dialog import Dialog, ModelName, exchange
from ayy.memory import SemanticCategory
from ayy.prompts import NAME_DIALOG, SUMMARIZE_MESSAGES

DIALOG_NAMER_DIALOG = Dialog(model_name=ModelName.GEMINI_FLASH, system=NAME_DIALOG, name="dialog_namer_dialog")


SUMMARIZER_DIALOG = Dialog(
    model_name=ModelName.GEMINI_FLASH,
    system=SUMMARIZE_MESSAGES
    + f"\nAvailable categories for semantic memories: {SemanticCategory._member_names_}",
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
                "semantic_memories": [
                    {
                        "name": "personal_info",
                        "content": "User's name is Hamza",
                        "category": "identity",
                        "confidence": 1.0,
                    },
                    {
                        "name": "family_location",
                        "content": "Has family in New York",
                        "category": "relationships",
                        "confidence": 1.0,
                    },
                    {
                        "name": "travel_preferences",
                        "content": "Prefers sunny weather and morning flights",
                        "category": "preferences",
                        "confidence": 0.8,
                    }
                ],
                "episodic_memories": [
                    {
                        "name": "flight_booking",
                        "content": {
                            "flight_number": "AA1234",
                            "departure": "8:30 AM Saturday",
                            "arrival": "11:45 AM",
                            "confirmation": "XYZ789"
                        },
                        "context": "current_trip",
                        "confidence": 1.0,
                    },
                    {
                        "name": "airport_pickup",
                        "content": "Family will provide airport pickup",
                        "context": "current_trip",
                        "confidence": 1.0,
                    }
                ]
            }""",
        ),
    ],
    name="summarizer_dialog",
)
