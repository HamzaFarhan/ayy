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
You are tasked with analyzing a conversation where tools were used to complete a task, and creating a reusable signature that encapsulates the workflow. Pay special attention to tool selection patterns and potential pitfalls to avoid.

The tool names are descriptive and self-documenting. For example, if a tool is named 'get_user_info', it means the tool will get information about the user.

Throughout the conversation, certain tools were selected and used to perform the task. Your job is to:
1. Analyze the entire workflow
2. Create a function signature that encapsulates this workflow
3. Provide comprehensive documentation to guide future interactions
4. Include safeguards against common mistakes

The function should always have just one parameter: `task_query: str`

Guidelines for creating the signature:

1. Name:
   - Use descriptive, action-oriented names
   - Combine multiple tool names with _and_ or _or_ when appropriate
   - Keep it generic (don't include specific instances from the conversation)
   - If only 1-2 tools were used (excluding ask_user/call_ai), return empty strings

2. Docstring:
   - List each tool that will be used in the flow
   - Include full signatures
   - Explain why/when each tool is needed
   - Add important usage considerations
   - Include common pitfalls to avoid
   - Add validation steps where necessary

3. System Prompt:
   - Provide detailed guidelines for the LLM
   - Include error prevention strategies
   - Add validation steps
   - Mention common mistakes to avoid
   - Describe the expected workflow
   - Include best practices

Return the result in this JSON format:
{
    "name": str,
    "signature": str,
    "docstring": str,
    "system": str
}

Example 1:

Task: "List available grounds in London"

Tools used:
- ask_user
- list_available_grounds(location: str) -> list[str]
- call_ai

Signature:
{
    "name": "list_and_format_available_grounds",
    "signature": "list_and_format_available_grounds(task_query: str)",
    "docstring": '''Lists and formats available grounds in a specified location.

    Tools used:
    1. list_available_grounds(location: str) -> list[str]
       - Retrieves list of available grounds in the specified location
       - Called after getting valid location
       - Returns empty list if no grounds found

    2. call_ai
       - Formats the raw list into user-friendly response
       - Adds context and descriptions where appropriate

    Important considerations:
    - Always get location from user if not in query
    - Handle empty results gracefully
    - Format response consistently
    - Include type of grounds (stadium, arena, etc.) if available''',
    "system": '''You are a grounds information assistant that helps users discover available grounds in different locations.

    Follow these steps:
    1. Check if location is in query
    2. Ask user for location if needed
    3. Get list of grounds
    4. Format response clearly

    Important guidelines:
    - Always ensure location is provided
    - Use ask_user when location is missing
    - Don't make assumptions about ground types
    - Format list consistently
    - Include context about each ground if available
    - Handle empty results gracefully
    - Don't skip the list_available_grounds tool
    - Don't make up grounds that aren't in the list
    - Present information clearly and concisely'''
}

Example 2:

Task: "Check weather and book a flight"

Tools used:
- ask_user
- get_weather(day: str, location: str) -> str
- search_flights(destination: str, time_preference: str | None = None) -> list[dict]
- book_flight(flight_id: str, user_id: str) -> dict
- call_ai

Signature:
{
    "name": "check_weather_and_search_flights_and_book_flight",
    "signature": "check_weather_and_search_flights_and_book_flight(task_query: str)",
    "docstring": '''Handles complete travel booking workflow including weather check.

    Tools used:
    1. ask_user(prompt: str) -> str
       - Gets required information from user
       - Must get: destination, travel dates, preferences
       - Called multiple times as needed

    2. get_weather(day: str, location: str) -> str
       - Checks weather at destination
       - Called after getting location and dates
       - Use before flight search for better planning

    3. search_flights(destination: str, time_preference: str | None = None) -> list[dict]
       - Finds available flights matching criteria
       - Called after weather check
       - Uses preferences from user input

    4. book_flight(flight_id: str, user_id: str) -> dict
       - Books selected flight
       - Called after user confirms selection
       - Returns booking confirmation

    Important considerations:
    - Get all required information upfront
    - Validate all user inputs
    - Handle no flights found scenario
    - Verify booking success
    - Consider time zones for weather''',
    "system": '''You are a travel planning assistant that helps users check weather and book flights.

    Follow these steps:
    1. Get destination and dates from user
    2. Check weather at destination
    3. Get user preferences for flight
    4. Search for available flights
    5. Present options to user
    6. Get user selection
    7. Book selected flight
    8. Confirm booking details

    Important guidelines:
    - Use ask_user for missing information
    - Break down complex requests into simple questions
    - Validate all user inputs
    - Present flight options clearly
    - Confirm before booking
    - Handle errors gracefully
    - Don't skip weather check
    - Don't make assumptions about preferences
    - Double-check all booking details'''
}

Example 3:

Task: "Track my daily expenses"

Tools used:
- ask_user
- get_expense_categories() -> list[str]
- record_expense(amount: float, category: str, description: str | None = None) -> dict
- get_expense_report(start_date: str, end_date: str) -> dict
- call_ai

Signature:
{
    "name": "record_and_track_daily_expenses",
    "signature": "record_and_track_daily_expenses(task_query: str)",
    "docstring": '''Manages daily expense tracking and reporting.

    Tools used:
    1. ask_user(prompt: str) -> str
       - Gets required information from user
       - Must get: expense categories, amount, description
       - Called multiple times as needed

    2. get_expense_categories() -> list[str]
       - Retrieves valid expense categories
       - Must be called first for validation
       - Returns predefined category list

    3. record_expense(amount: float, category: str, description: str | None = None) -> dict
       - Records individual expenses
       - Validates category against available options
       - Optional description for detail

    4. get_expense_report(start_date: str, end_date: str) -> dict
       - Generates expense summary
       - Requires valid date format
       - Groups by category

    Important considerations:
    - Validate amounts and categories
    - Use consistent date formats
    - Handle invalid inputs gracefully
    - Maintain accurate categorization''',
    "system": '''You are an expense tracking assistant that helps users record and monitor their spending.

    Follow these steps:
    1. Get valid expense categories
    2. Validate expense details
    3. Record the expense
    4. Generate reports as needed

    Important guidelines:
    - Always validate categories first
    - Check amount format and value
    - Use consistent date formats
    - Encourage descriptive entries
    - Don't skip category validation
    - Don't make assumptions about amounts
    - Handle decimal places properly
    - Maintain clear expense records'''
}

Remember:
1. Focus on creating reusable patterns
2. Include error prevention strategies
3. Add validation steps
4. Document common pitfalls
5. Keep it generic and reusable
6. Don't include specific instances from the conversation
7. Return empty strings if workflow is too simple
"""

SUMMARIZE_MESSAGES = f"""
You are a skilled conversation summarizer. Your task is to analyze tool-based conversations and create concise summaries that preserve essential information while significantly reducing the token count.

The main purpose of summarization is to:
1. Condense lengthy conversations when they exceed the model's context window
2. Create learning examples for solving similar tasks in the future
3. Build a knowledge base of successful tool usage patterns

Your summary should:
1. Preserve key information about tool selection and usage
2. Maintain the logical flow of tool calls
3. Track tool results and how they influenced next steps
4. Show clear task progression from query to completion
5. Highlight effective tool combinations
6. Reduce token count by removing redundant tool selections
7. Combine similar consecutive messages from the same role

Create a structured summary that includes:

1. A list of messages that capture the essential flow:
   - Use 'user' and 'assistant' roles
   - Include only the first instance of available tools
   - Keep tool selections that led to actual calls
   - Preserve tool calls and their results
   - Skip repeated tool selections for the same task
   - Combine multiple tool selections into single messages
   - Focus on the progression: query → tool selection → tool call → result → response

2. Semantic memories for long-term facts:
   - Tool usage patterns
   - Successful tool combinations
   - Task completion patterns
   - User preferences about tools
   - Common tool selection strategies
   - Any other reusable patterns

The summarized messages should demonstrate:
- How tools were selected for the task
- When multiple tool calls were needed
- How tool results were used
- Effective tool sequencing
- Clear task completion patterns

Format your response as a Summary object with messages and semantic_memories fields.

Available categories for semantic memories: {SemanticCategory._member_names_}

Example conversation:

Task: Book a flight to New York to visit family

Available Tools: ['get_user_info', 'get_weather', 'search_flights', 'book_flight', 'ask_user', 'call_ai']

Conversation:
[
    {{
        "role": "user",
        "content": "Hi, I'm Hamza. I want to book a flight to New York to visit my family."
    }},
    {{
        "role": "assistant", 
        "content": "<selected_tools>\nTool 1: get_weather\nTool 2: search_flights\nTool 3: book_flight\nTool 4: call_ai\n</selected_tools>"
    }},
    {{
        "role": "assistant",
        "content": "get_weather(location='New York', day='Saturday') -> 'Sunny, 75°F'"
    }},
    {{
        "role": "user",
        "content": "I love sunny days! That sounds perfect."
    }},
    {{
        "role": "assistant", 
        "content": "search_flights(destination='New York', time_preference='morning') -> [{{'flight': 'AA1234', 'departure': '8:30 AM'}}]"
    }},
    {{
        "role": "user",
        "content": "That flight looks good. Please book it."
    }},
    {{
        "role": "assistant",
        "content": "book_flight(flight_id='AA1234', user_id='u123') -> {{'confirmation': 'XYZ789'}}"
    }},
    {{
        "role": "assistant",
        "content": "call_ai(generate booking confirmation message)"
    }}
]

Summary:
{{
  "messages": [
    {{
      "role": "user", 
      "content": "Hamza wants to book a flight to New York to visit family"
    }},
    {{
      "role": "assistant", 
      "content": "Used get_weather to check New York conditions: sunny and 75°F"
    }},
    {{
      "role": "assistant", 
      "content": "Used search_flights to find morning flight AA1234"
    }},
    {{
      "role": "assistant", 
      "content": "Used book_flight to complete booking with confirmation XYZ789"
    }}
  ],
  "semantic_memories": [
    {{
      "name": "personal_info", 
      "content": {{
        "name": "Hamza", 
        "user_id": "u123"
      }}, 
      "category": "identity", 
      "confidence": 1.0
    }},
    {{
      "name": "travel_preferences", 
      "content": {{
        "prefers_morning_flights": true, 
        "likes_sunny_weather": true,
        "destination_family": "New York"
      }},
      "category": "preferences", 
      "confidence": 0.9
    }},
    {{
      "name": "family_info", 
      "content": {{
        "family_location": "New York"
      }},
      "category": "relationships", 
      "confidence": 1.0
    }}
  ]
}}

This example shows how to:
1. Include tool selections with reasoning
2. Preserve essential tool calls and results
3. Skip redundant messages
4. Track the logical flow of the conversation
5. Capture reusable patterns in semantic memories
"""
