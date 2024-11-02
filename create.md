Here's a comprehensive guide to creating a dataset for LLM evaluation:

## Why a Diverse Dataset Matters

A diverse dataset is crucial for effective AI evaluation, offering several key benefits:

- **Comprehensive Testing**: Ensures evaluation across a wide range of situations
- **Realistic Interactions**: Reflects actual user behavior for relevant evaluations
- **Identifies Weaknesses**: Helps uncover areas where the AI may struggle or produce errors

## Dataset Structure Dimensions

For B2C applications, consider structuring your dataset along these key dimensions:

**Features**
| Feature | Description |
|---------|-------------|
| Email Summarization | Condensing lengthy emails into key points |
| Meeting Scheduler | Automating meeting scheduling across time zones |
| Order Tracking | Providing shipment status and delivery updates |
| Contact Search | Finding contact information from a database |
| Language Translation | Translating text between languages |
| Content Recommendation | Suggesting articles based on user interests |

**Scenarios**
| Scenario | Description |
|----------|-------------|
| Multiple Matches Found | User request yields multiple results needing narrowing down |
| No Matches Found | Request yields no results, requiring alternatives |
| Ambiguous Request | Input lacks necessary specificity |
| Invalid Data Provided | User provides incorrect data type/format |
| System Errors | Technical issues prevent normal operation |
| Incomplete Information | User omits required details |
| Unsupported Feature | User requests non-existent functionality |

**Personas**
| Persona | Description |
|---------|-------------|
| New User | Unfamiliar with system; requires guidance |
| Expert User | Experienced; expects efficiency |
| Non-Native Speaker | May have language barriers |
| Busy Professional | Values quick, concise responses |
| Technophobe | Uncomfortable with technology |
| Elderly User | May not be tech-savvy |

## Data Generation Approaches

To build your dataset, you can use two main approaches:

- **Use Existing Data**: Sample real user interactions or behaviors from your AI system
- **Generate Synthetic Data**: Use LLMs to create realistic user inputs covering various features, scenarios, and personas

Example LLM prompt structure for generating synthetic data:

| ID | Feature | Scenario | Persona | LLM Prompt | Assumptions |
|----|----------|-----------|----------|------------|-------------|
| 1 | Order Tracking | Invalid Data | Frustrated Customer | "Generate input from irritated user demanding order status for #1234567890" | Order number doesn't exist |
| 2 | Contact Search | Multiple Matches | New User | "Create input from someone unfamiliar seeking contact info for 'Alex'" | Multiple 'Alex' contacts exist |
| 3 | Meeting Scheduler | Ambiguous Request | Busy Professional | "Simulate rushed user requesting meeting scheduling with minimal details" | N/A |
| 4 | Content Recommendation | No Matches | Expert User | "Generate query about sustainable supply chain management" | No matching articles exist |

Generating Synthetic Data
When generating synthetic data, you only need to create the user inputs. You then feed these inputs into your AI system to generate the AI’s responses. It’s important that you log everything so you can evaluate your AI. To recap, here’s the process:

Generate User Inputs: Use the LLM prompts to create realistic user inputs.
Feed Inputs into Your AI System: Input the user interactions into your AI as it currently exists.
Capture AI Responses: Record the AI’s responses to form complete interactions.
Organize the Interactions: Create a table to store the user inputs, AI responses, and relevant metadata.
How much data should you generate?
There is no right answer here. At a minimum, you want to generate enough data so that you have examples for each combination of dimensions (in this toy example: features, scenarios, and personas). However, you also want to keep generating more data until you feel like you have stopped seeing new failure modes. The amount of data I generate varies significantly depending on the use case.

Does synthetic data actually work?
You might be skeptical of using synthetic data. After all, it’s not real data, so how can it be a good proxy? In my experience, it works surprisingly well. Some of my favorite AI products, like Hex use synthetic data to power their evals:

“LLMs are surprisingly good at generating excellent - and diverse - examples of user prompts. This can be relevant for powering application features, and sneakily, for building Evals. If this sounds a bit like the Large Language Snake is eating its tail, I was just as surprised as you! All I can say is: it works, ship it.” Bryan Bischof, Head of AI Engineering at Hex

Next Steps
With your dataset ready, now comes the most important part: getting your principal domain expert to evaluate the interactions.

