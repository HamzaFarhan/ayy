# DreamAI x PoC

## Proof of Concept: Designing a Multi-Agent LLM Setup for Investor Engagement

### Objective
The goal of this assignment is to create a multi-agent large language model (LLM) system for driving investor-friendly conversations, leveraging specialist agents to assist with asset management, risk evaluation, and compliance. Ultimately, human intervention will be incorporated for critical decision-making and finalizing investments that exceed predefined thresholds. This system aims to reduce hallucinations and ensure an effective, engaging experience for users.

### Assignment Tasks

#### 1. Design a Multi-Agent Model Setup:
- **Primary Agent**: Responsible for engaging in investor-friendly conversations and acting as the central interface for investors.
- **Specialist Support Agents**:
  - **Relationship Management Agent**: Handles customer queries, maintains rapport, and ensures a seamless user experience.
  - **Asset Specialist**: Provides expertise on available asset classes (e.g., mutual funds, commodities, and stocks) and their suitability based on the investor’s portfolio.
  - **Investment Risk Specialist**: Evaluates the risk levels associated with investment decisions and offers advice on portfolio diversification or mitigation strategies.
  - **Compliance Specialist (optional)**: Ensures all investment recommendations adhere to regulatory requirements and internal policies.
- **Human Intervention Agent**: Takes over in scenarios where investment thresholds exceed predefined limits or when the investor seeks confirmation from a human advisor.

#### 2. System Workflow:
- Design communication flow between agents, ensuring that the primary agent queries specialist agents as needed and escalates decisions to the Human Intervention Agent when appropriate.
- Include a mechanism to track conversations and contextual dependencies between agents to maintain consistency.

#### 3. Evaluate Running Costs:
- Assess computational and financial costs of running a multi-agent setup, including:
  - Resource costs of running multiple specialized LLMs simultaneously.
  - Hosting costs (e.g., cloud or on-premises infrastructure).
  - Optimization strategies to minimize costs without compromising performance.
- Summarize the approximate cost of running this setup per interaction or on a monthly/annual basis.

### Deliverables

1. **Multi-Agent Model Setup Design**:
   - Present a system architecture diagram illustrating how the primary agent interacts with specialist agents and the human agent.
   - Provide a written explanation of the roles and responsibilities of each agent.
2. **Workflow Demonstration**:
   - Simulate a conversation where the primary agent engages with an investor and queries specialist agents to provide detailed responses.
   - Include an escalation to the Human Intervention Agent when the threshold is exceeded.
3. **Cost Analysis**:
   - Provide a detailed breakdown of costs associated with running the multi-agent system.
   - Suggest strategies to optimize costs while maintaining quality.

### Inspiration
Refer to the approach described in this paper for structuring your multi-agent system. However, ensure that your work includes customizations specific to investor engagement, as described above.

[Source](https://arxiv.org/html/2403.13313v1)



