Our client came to us with a Proof of Concept: Designing a Multi-Agent LLM Setup for Investor Engagement
He has a company that provides investment advice and also helps users with their investments.
He was inspired by the Polaris system and wanted to explore the use of multi-agent LLMs in a similar capacity for their users.
We have some notes from the polaris research paper in polaris.md
The proof of concept is in poc.md
We now need to send him a document outlining our requirements from him. So for example, based on the poc, we made some initial agent designs in agents.py, we also have the information from the polaris paper that talks about agents being able to build trust and rapport and have tangible interactions with the users. This should also be added into the design of the agents in agents.py
We also need to know where the data will come from for the poc, e.g in polaris, they have this snippet: "...The development of conversational agents capable of engaging in meaningful and accurate clinical discussions represents a significant challenge. This challenge is compounded by the lack of datasets specifically tailored for training models in the nuanced context of healthcare conversations. Our work addresses this critical gap by leveraging a unique compilation of dialogues, including simulated interactions between registered nurses and patient actors. These dialogues form the cornerstone of our approach to developing a conversational agent with the proficiency to navigate a wide array of clinical discussions. To enhance the agent’s reasoning capabilities, instruction-following proficiency, and domain-specific knowledge – we augmented these datasets by introducing diverse kinds of instructions and tasks geared for multi-hop reasoning. For instance, given a user utterance “my shoes do not fit”, the agent should be able to reason about “swollen ankle → sign of fluid retention → sign of exacerbating CHF conditions”..."

We also need to know what the success metrics are for the poc, so need some sort of checklist of what we need to achieve. That could be sample workflows, e.g if the user asks about a certain topic, what workflow should be triggered and what are the expected outputs. For this, we made a sample scenarios.md file. Feel free to add or remove things from it.

We will not show our client any code right now, we just need a document.
So this is what I need you to put in the document:
- Our initial agent designs. Look at agents.py and combine it with the information from the polaris paper.
- Ask the client for feedback on our agent designs.
- Suggest some sample data sources + formats + values for the poc.
- Ask for feedback on the sata stuff and also ask for more suggestions.
- Outline some sample workflows and their success criterias.
- Ask for feedback on the sample workflows and success criterias.

Don't mention a "constellation of agents" or anything like that, just say "multi-agent llm setup".
In the agent design, mention how the system prompt/role/qualifications or something else will ensure that the agents will act like we want them to.
In markdown.