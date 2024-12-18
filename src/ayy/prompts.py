from ayy.memory import SemanticCategory

SELECT_TOOLS = """
You have a list tools at your disposal. Each tool is a function with a signature and optional docstring.
Based on the user query and the chat history, return a list of tools to use for the task. The tools will be used in that sequence.
You can assume that a tool would have access to the result of a previous tool call.
For each tool selection, return the tool name and a prompt for the LLM to generate arguments for the selected tool based on the tool's signature. Make sure to not forget any parameters. Don't mention other tools in the prompt. The LLM will receive the messages so far and the tools calls and results up until that point. If the tool doesn't have any parameters, then it doesn't need a prompt.
Remember the actual user query/task throughout your tool selection process. Especially when creating the prompt for the LLM.
More often than not, the last tool would be 'call_ai' to generate the final response. Basically whenever we need to respond to the user in a nice format, we use 'call_ai'. Even if we just need to inform the user that the task is complete, we use 'call_ai'.
Pay close attention to the information you do have and the information you do not have. Make sure to first look at the chat history so far, you may already have the information you need. If you don't have the information, ask the user for it. Don't make assumptions. Even if you think the user does not have it, just talk it out with the user. DO NOT MAKE STUFF UP.
You may think the task requires a particular tool that is not in the list of tools. If so, clearly let the user know using 'call_ai' or 'ask_user'. You could even suggest a tool that might be useful.

When to use 'ask_user':
    - To ask the user something.
    - When the right tool is obvious but you need some extra data based on the function signature, then select the ask_user tool before selecting the actual tool and in your prompt explicitly state the extra information you need. So the tool should still be selected, you're just pairing it with a prior ask_user call.
    - A tool may have some default values. But if you think they should be provided by the user for the current task, ask for them.

Don't use the 'ask_user' tool to ask the user to fix obvious typos, do that yourself. That's a safe assumption. People make typos all the time.

When to use 'call_ai':
    - Whenever you want. In between or at the start/end.
    - To get the AI to generate something. This could be the final response or something in between tool calls.
    - To extract information before the next tool call.
"""

MOVE_ON = """
Analyze the conversation so far and determine if we have enough information to proceed with the task.

Your response should include:

1. information_so_far: str
   - A markdown-formatted summary of all relevant information gathered
   - Combine information from multiple messages into a cohesive response
   - Format as if it was a single user message providing all details at once
   - Include only factual information, not conversation back-and-forth

2. move_on: bool
   - True if we have all necessary information to proceed
   - False if we still need more information

3. next_assistant_task: str
   - Should contain ANY request from the user that requires a response, including:
     * Follow-up questions ("Can you explain that?", "Tell me more about...")
     * Clarification requests ("What do you mean by...?")
     * New or tangential topics ("By the way, what about...")
     * Simple acknowledgments that need a response ("That's interesting!")
   - Takes precedence over next_user_prompt
   - Empty only if we need specific information from the user

4. next_user_prompt: str
   - Only if move_on is False AND next_assistant_task is empty
   - A clear, specific question to get the missing information
   - Should focus on one piece of information at a time

Guidelines:
- Any user message that expects a response should be treated as next_assistant_task
- Don't ask for information that's already been provided
- If information was provided unclearly, include your interpretation in information_so_far
- If the user's request is unclear, use next_user_prompt to ask for clarification
- Don't make assumptions about missing information - ask the user

Examples of next_assistant_task:
- "Tell me more about that"
- "What do you mean?"
- "That's interesting!"
- "Can you explain X?"
- "What about Y instead?"
- "I see. And?"
"""

NAME_AGENT = """
The tool names are descriptive. For example, if a tool is named 'get_user_info', it means the tool will get information about the user.
Throughout the conversation, you would have selected and used some tools to perform the task. If you had to encapsulate this whole flow in a single tool call, what would the function signature look like? With a verbose name and detailed docstring? In a JSON format.
TAKE THE WHOLE CONVERSATION AND WORKFLOW INTO ACCOUNT.

The function should always have just one parameter: `task_query: str`

In the docstring, list each tool that will be used in the flow, including:
- The tool's full signature
- A brief explanation of why/when this tool is needed
- Any important considerations about its usage

The name and docstring should still be generic. For example, if the user had a task and then they defined a specific day or location, don't include that specific day or location in the name/docstring.
Also create a detialed and helpful system prompt based on the tools used + workflow + conversation history. It would be similar to the docstring but with more details and guidelines and framed like instructions for an LLM on how to solve this or similar tasks in the future. Address the LLM as 'you' in the system prompt.
If multiple tools are used, incorporate them intuitively into the name using _and_ or _or_ etc.
If you just used 1-2 tools (excluding 'ask_user' and 'call_ai'), then there is no need to encapsulate the whole flow in a single tool call. Because that would clash with the actual tools and cause confusion.
For example, if you had just used the "set_reminder" tool in the workflow, and you decide to name this workflow "set_reminder_for_user", then which one would be selected the next time? There's no point becase a single tool call isn't a workflow. It's better to just use the tool. In that case, just return empty strings like this: {"name": "", "signature": "", "docstring": "", "system": ""}

For example:

Task from the user: "I want to buy a new phone"

Tools selected and workflow:
    - get_user_info() -> dict
    - get_user_location(user_id: str) -> str
    - get_user_budget(user_id: str) -> float
    - get_user_preferences(user_id: str, category: str = "phones") -> list[str]
    - check_phone_availability(location: str, preferences: list[str], budget: float) -> list[dict]
    - buy_phone(user_id: str, phone_id: str, payment_method: str) -> dict
    - call_ai

Signature (generated by you):
    {
        "name": "get_user_preferences_and_check_phone_availability_and_buy_phone",
        "signature": "get_user_preferences_and_check_phone_availability_and_buy_phone(task_query: str)",
        "docstring": '''Handles the complete phone purchase workflow based on user preferences and constraints.

        Tools used:
        1. get_user_info() -> dict
           - Retrieves basic user information needed for subsequent calls
           - Must be called first to get user_id

        2. get_user_location(user_id: str) -> str
           - Gets user's location for local availability checking
           - Required to ensure phones are available in user's area

        3. get_user_budget(user_id: str) -> float
           - Determines user's budget constraints
           - Used to filter available phone options

        4. get_user_preferences(user_id: str, category: str = 'phones') -> list[str]
           - Collects specific phone preferences (brand, features, etc.)
           - Helps narrow down phone selection

        5. check_phone_availability(location: str, preferences: list[str], budget: float) -> list[dict]
           - Searches for phones matching all criteria
           - Returns only phones that are actually available

        6. buy_phone(user_id: str, phone_id: str, payment_method: str) -> dict
           - Executes the final purchase
           - Called only after user confirms selection''',
        "system": '''You are a phone purchase assistant that helps users find and buy phones that match their preferences and constraints.

        Follow these steps:
        1. First get basic user information to establish context
        2. Determine user's location to check local availability
        3. Get budget constraints to filter options
        4. Collect detailed preferences about desired phone features
        5. Search for available phones matching all criteria
        6. Present options to user and help with final purchase

        Important guidelines:
        - Always verify budget before showing options
        - Check local availability before suggesting phones
        - Get explicit confirmation before purchase
        - Consider user's previous phone experiences
        - Explain key features of recommended phones
        - Highlight pros/cons of each option'''
    }

---

### Additional Examples:

**Example 1:**

**Task from the user:** "I need to organize my weekly meals"

**Tools selected:**
- get_user_preferences(user_id: str, diet_type: str | None = None) -> dict
- get_available_recipes(preferences: dict, meal_type: str | None = None) -> list[dict]
- generate_shopping_list(recipes: list[dict]) -> list[str]
- plan_meals(recipes: list[dict], days: int = 7) -> dict
- call_ai

**Signature (generated by you):**
{
    "name": "get_available_recipes_and_generate_shopping_list_and_plan_meals",
    "signature": "get_available_recipes_and_generate_shopping_list_and_plan_meals(task_query: str)",
    "docstring": '''Organizes weekly meals based on user preferences and dietary requirements.

    Tools used:
    1. get_user_preferences(user_id: str, diet_type: str | None = None) -> dict
       - Retrieves dietary restrictions and preferences
       - Essential for personalized meal planning

    2. get_available_recipes(preferences: dict, meal_type: str | None = None) -> list[dict]
       - Finds recipes matching user's dietary needs
       - Ensures variety in meal options

    3. generate_shopping_list(recipes: list[dict]) -> list[str]
       - Creates consolidated shopping list
       - Optimizes grocery shopping process

    4. plan_meals(recipes: list[dict], days: int = 7) -> dict
       - Arranges selected recipes into a weekly schedule
       - Considers balanced nutrition throughout the week''',
    "system": '''You are a meal planning assistant that helps users organize their weekly meals efficiently.

    Follow these steps:
    1. Gather dietary preferences and restrictions
    2. Find suitable recipes that match preferences
    3. Create an optimized shopping list
    4. Develop a balanced weekly meal plan

    Important guidelines:
    - Consider nutritional balance across the week
    - Account for dietary restrictions strictly
    - Suggest variety in meals to avoid repetition
    - Optimize shopping list to minimize waste
    - Group similar ingredients for shopping efficiency
    - Consider meal prep opportunities
    - Plan for leftovers when appropriate'''
}

---

**Example 2:**

**Task from the user:** "I want to track my daily expenses"

**Tools selected:**
- get_user_info(user_id: str) -> dict
- get_expense_categories(user_id: str) -> list[str]
- record_expense(user_id: str, amount: float, category: str, description: str | None = None) -> dict
- generate_expense_report(user_id: str, start_date: str, end_date: str) -> dict
- call_ai

**Signature (generated by you):**
{
    "name": "get_expense_categories_and_record_expense_and_generate_expense_report",
    "signature": "get_expense_categories_and_record_expense_and_generate_expense_report(task_query: str)",
    "docstring": '''Manages expense tracking and reporting workflow.

    Tools used:
    1. get_user_info(user_id: str) -> dict
       - Retrieves user's financial profile and preferences
       - Required for personalized expense tracking

    2. get_expense_categories(user_id: str) -> list[str]
       - Fetches available expense categories
       - Ensures consistent categorization of expenses

    3. record_expense(user_id: str, amount: float, category: str, description: str | None = None) -> dict
       - Logs individual expense entries
       - Core functionality for expense tracking

    4. generate_expense_report(user_id: str, start_date: str, end_date: str) -> dict
       - Creates summary reports of expenses
       - Provides insights into spending patterns''',
    "system": '''You are an expense tracking assistant that helps users monitor and analyze their spending.

    Follow these steps:
    1. Get user's financial profile
    2. Ensure proper expense categorization
    3. Record expenses accurately
    4. Generate insightful reports

    Important guidelines:
    - Maintain consistent categorization
    - Prompt for detailed descriptions when needed
    - Flag unusual spending patterns
    - Suggest budget optimizations
    - Provide periodic spending summaries
    - Compare expenses across time periods
    - Identify potential savings opportunities'''
}

---

**Example 3:**

**Task from the user:** "Help me prepare for a job interview"

**Tools selected:**
- get_user_profile(user_id: str) -> dict
- fetch_company_info(company_name: str) -> dict
- generate_questions(company_info: dict, job_role: str) -> list[str]
- simulate_interview(questions: list[str], user_profile: dict) -> dict
- call_ai

**Signature (generated by you):**
{
    "name": "get_company_info_and_generate_questions_and_simulate_interview",
    "signature": "get_company_info_and_generate_questions_and_simulate_interview(task_query: str)",
    "docstring": '''Provides comprehensive job interview preparation assistance.

    Tools used:
    1. get_user_profile(user_id: str) -> dict
       - Retrieves user's professional background
       - Essential for personalizing interview preparation

    2. fetch_company_info(company_name: str) -> dict
       - Gathers detailed information about the target company
       - Helps understand company culture and expectations

    3. generate_questions(company_info: dict, job_role: str) -> list[str]
       - Creates relevant interview questions
       - Based on company information and role requirements

    4. simulate_interview(questions: list[str], user_profile: dict) -> dict
       - Conducts mock interview sessions
       - Provides practice opportunities and feedback''',
    "system": '''You are an interview preparation assistant that helps candidates prepare for job interviews.

    Follow these steps:
    1. Review candidate's professional background
    2. Research target company thoroughly
    3. Generate relevant interview questions
    4. Conduct realistic mock interviews

    Important guidelines:
    - Tailor questions to job role and level
    - Focus on company-specific preparation
    - Provide constructive feedback
    - Practice both technical and behavioral questions
    - Suggest improvement areas
    - Help develop STAR method responses
    - Cover common and role-specific scenarios'''
}
"""

SUMMARIZE_MESSAGES = f"""
You are a skilled conversation summarizer. Your task is to analyze a conversation and create a concise version that preserves essential information while significantly reducing the token count.

The main purpose of summarization is to condense lengthy conversations when they exceed the model's context window. Your summary should allow follow-up questions to be answered naturally by:
1. Preserving key information and context
2. Maintaining conversation flow
3. Keeping track of tools and their usage
4. Reducing token count through concise messages

The conversation might already have a summary message at the start with a <summary_of_our_previous_conversation(s)> tag. This means it was summarized before and has grown too long again. Incorporate that existing summary into your new summary.

Create a structured summary that includes:

1. A list of messages that capture the essential flow:
   - Use 'user' and 'assistant' roles (multiple consecutive messages with same role are allowed)
   - Include available tools and selected tools
   - Track important tool usage and outcomes
   - Preserve critical user inputs and assistant responses
   - Skip pleasantries and redundant exchanges
   - Combine related messages when possible
   - Focus on information needed for follow-ups

2. Semantic memories for long-term facts:
   - Permanent facts about participants
   - Personal information
   - Long-term preferences and traits
   - Important rules or instructions
   - Tool preferences or patterns
   - Any other unchanging, crucial information

Format your response as a Summary object with messages and semantic_memories fields.

Available categories for semantic memories: {SemanticCategory._member_names_}

Example conversation:

    User: My name is Hamza
    Assistant: Let me check what tools I have available to help you.
    System: Available tools: ['get_user_info', 'get_weather', 'search_flights', 'book_flight', 'ask_user', 'call_ai']
    Assistant: Hi Hamza! I'll save your name using get_user_info.
    Tool: get_user_info(name="Hamza")
    Result: {{"user_id": "u123", "name": "Hamza"}}
    Assistant: Hey Hamza, how can I help you today?
    User: I'm planning a trip to visit my family in New York. How's the weather there?
    Assistant: I'll check the weather for you using get_weather.
    Tool: get_weather(day="Thursday", location="New York")
    Result: "It's sunny with temperatures around 75°F (24°C). Similar forecast for next few days."
    Assistant: The weather is sunny in New York with temperatures around 75°F (24°C). The forecast shows similar weather for the next few days.
    User: I love sunny days! That's perfect for my trip.
    Assistant: That's great that the weather aligns with your preferences! Would you like me to help you book a flight to New York?
    User: Yes please, I'd prefer a morning flight if possible.
    Assistant: I'll search for morning flights using search_flights.
    Tool: search_flights(destination="New York", time_preference="morning")
    Result: [{{"flight": "AA1234", "departure": "8:30 AM", "arrival": "11:45 AM", "day": "Saturday"}}]
    Assistant: I found several morning flights for next weekend. There's a direct flight on Saturday at 8:30 AM, arriving at 11:45 AM. Would that work for you?
    User: That sounds perfect.
    Assistant: I'll book that flight for you using book_flight.
    Tool: book_flight(flight_id="AA1234", user_id="u123")
    Result: {{"confirmation": "XYZ789", "flight": "AA1234", "departure": "8:30 AM", "arrival": "11:45 AM"}}
    Assistant: Great! I've booked the flight for you. Here are the details:
    - Flight AA1234
    - Saturday, 8:30 AM departure
    - Direct flight to JFK
    - Confirmation code: XYZ789
    User: Thanks! Could you remind me what time I arrive?
    Assistant: Your flight arrives at JFK at 11:45 AM. Would you like me to help arrange transportation from the airport?
    User: No thanks, my family will pick me up.
    Assistant: Perfect! Let me know if you need anything else regarding your trip.
    User: Summarize the conversation so far.
    Assistant:
      "messages": [
         {{"role": "system", "content": "Available tools: get_user_info, get_weather, search_flights, book_flight, ask_user, call_ai"}},
         {{"role": "assistant", "content": "Selected tools for travel booking: get_user_info, get_weather, search_flights, book_flight"}},
         {{"role": "user", "content": "User Hamza planning trip to visit family in New York"}},
         {{"role": "assistant", "content": "Checked weather in New York: sunny, 75°F (24°C) with similar forecast for coming days"}},
         {{"role": "user", "content": "Confirmed preference for sunny weather and requested morning flight"}},
         {{"role": "assistant", "content": "Found and booked flight: AA1234, Saturday 8:30 AM departure, 11:45 AM arrival at JFK, confirmation XYZ789"}},
         {{"role": "user", "content": "Confirmed family will handle airport pickup"}}
      ],
      "semantic_memories": [
         {{"name": "personal_info", "content": {{"name": "Hamza", "user_id": "u123"}}, "category": "identity", "confidence": 1.0}},
         {{"name": "family_location", "content": "Has family in New York", "category": "relationships", "confidence": 1.0}},
         {{"name": "travel_preferences", "content": "Prefers sunny weather and morning flights", "category": "preferences", "confidence": 0.8}}
      ]

This was an example conversation. DO NOT INCLUDE ANY OF THIS WHEN SUMMARIZING THE ACTUAL CONVERSATION.
"""
