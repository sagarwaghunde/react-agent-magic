# ReAct LangChain Agent

A Python implementation of the ReAct (Reasoning + Acting) pattern using LangChain and OpenAI. This project demonstrates how to build an intelligent agent that can reason about problems and use tools to solve them.

## 📋 Overview

This project implements a ReAct agent that follows a thought-action-observation loop to answer questions. The agent can use external tools (like calculating text length) and maintains a scratchpad of its reasoning process to arrive at the correct answer.

**Note**: This implementation is based on LangChain, a framework for developing applications powered by language models.

### What is ReAct?

ReAct (Reasoning + Acting) is a paradigm that combines reasoning traces and task-specific actions in large language models. The agent alternates between:
- **Thinking**: Reasoning about what to do next
- **Acting**: Using tools to gather information
- **Observing**: Processing the results from tools

## 🏗️ Project Structure

```
react-langchain/
├── main.py           # Main agent implementation with ReAct loop
├── callbacks.py      # Custom callback handler for LLM monitoring
└── README.md         # This file
```

### File Descriptions

#### `main.py`
The core implementation featuring:
- **ReAct Agent Loop**: Implements the reasoning-action-observation cycle
- **Tool Definition**: `get_text_length()` tool that counts characters in text
- **Prompt Template**: Structured prompt that guides the agent's reasoning
- **Scratchpad Management**: Tracks intermediate steps for context
- **Agent Execution**: Runs until the agent reaches a final answer

#### `callbacks.py`
Custom callback handler that provides visibility into:
- LLM prompts being sent
- LLM responses received
- Useful for debugging and understanding the agent's decision-making process

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- OpenAI API key
- Pipenv (or pip)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd react-langchain
   ```

2. **Install dependencies**
   
   Using Pipenv (recommended):
   ```bash
   pipenv install
   pipenv shell
   ```
   
   Or using pip:
   ```bash
   pip install langchain openai python-dotenv langchain-openai langchain-core
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

### Running the Agent

```bash
python main.py
```

### Expected Output

The agent will:
1. Receive the question: "What is the text length of the string DOG in characters?"
2. Reason about what tool to use
3. Execute the `get_text_length` tool
4. Observe the result
5. Provide the final answer

Example output:
```
Hello ReAct LangChain!
***Prompt to LLM was:***
...
************************************************
Agent step in while loop: ...
get_text_length enter with text: DOG
Observation: 3
Agent finished with the final answer: {'output': 'The text length of the string "DOG" is 3 characters.'}
```

## 🔧 How It Works

### The ReAct Loop

```python
while not isinstance(agent_step, AgentFinish):
    # 1. Agent thinks and decides on an action
    agent_step = agent.invoke({
        "input": question,
        "agent_scratchpad": intermediate_steps
    })
    
    # 2. If action needed, execute the tool
    if isinstance(agent_step, AgentAction):
        tool = find_tool_by_name(agent_step.tool, tools)
        observation = tool.func(agent_step.tool_input)
        
        # 3. Record the observation
        intermediate_steps.append((agent_step, observation))
    
# 4. Agent finishes with final answer
```

### Key Components

1. **Prompt Template**: Structures the agent's reasoning format ([source template from LangSmith Hub](https://smith.langchain.com/hub/hwchase17/react?organizationId=d924c6d5-9297-4b9b-9cdb-df9eea6b95a7) by [Harrison Chase](https://blog.langchain.com/author/harrison/), creator of LangChain)
2. **Tools**: Functions the agent can call (e.g., `get_text_length`)
3. **Output Parser**: Parses LLM output into `AgentAction` or `AgentFinish`
4. **Intermediate Steps**: List of tuples `(agent_step, observation)` that maintains the agent's memory and prevents redundant tool calls
5. **Scratchpad**: Formatted string representation of intermediate steps, inserted into the prompt for context
6. **Callback Handler**: Monitors LLM interactions for debugging

### Understanding "Observation" and Stop Tokens

A critical aspect of the ReAct implementation is the use of **stop tokens** to control LLM generation. This prevents the model from hallucinating observations.

#### The Problem: LLM Hallucination

Without stop tokens, the LLM might generate the entire reasoning chain including fake observations:

```
Thought: I should use get_text_length
Action: get_text_length
Action Input: DOG
Observation: 3          ← LLM might hallucinate this!
Thought: I now know...
```

#### The Solution: Stop Tokens

In the LLM configuration, we specify stop tokens:

```python
llm = ChatOpenAI(
    temperature=0, 
    stop=["\nObservation", "Observation"],  # Stop before generating observation
    callbacks=[MyCallbackHandler()]
)
```

**How it works:**

1. **LLM Input** - The prompt includes the format showing all keys: `Thought`, `Action`, `Action Input`, `Observation`, `Final Answer`

2. **LLM Output** - The model generates:
   ```
   Thought: I should use get_text_length to determine...
   Action: get_text_length
   Action Input: DOG
   ```
   Then **stops** when it's about to generate "Observation"

3. **Token Generation Stops** - Once the stop token is detected, no more tokens are decoded

4. **Agent Provides Real Observation** - The agent executor:
   - Parses the LLM output to extract the action
   - Executes the actual tool (`get_text_length("DOG")`)
   - Gets the real result (3)
   - Appends `Observation: 3` to the scratchpad
   
5. **Loop Continues** - The updated scratchpad with the real observation is fed back to the LLM for the next iteration

#### Why This Matters

- **Prevents Hallucination**: The LLM cannot make up tool results
- **Ensures Accuracy**: Observations come from actual tool execution
- **Maintains Trust**: The agent only acts on real data, not fabricated results

This mechanism is what makes ReAct agents reliable - they think and plan (LLM), but only observe real results (tool execution).

### The Role of `intermediate_steps`

The `intermediate_steps` variable is crucial for maintaining the agent's memory and preventing redundant tool calls.

#### What is `intermediate_steps`?

`intermediate_steps` is a list of tuples, where each tuple contains:
```python
(agent_step, observation)
```

- **agent_step**: An `AgentAction` object containing the action taken (tool name and input)
- **observation**: The result returned by the tool

#### Why is it Important?

1. **Maintains Agent Memory**: Stores the history of all actions and their results during the reasoning process

2. **Prevents Redundant Calls**: By keeping track of previous actions, the agent can see what it has already tried and avoid making the same tool call multiple times

3. **Enables Context-Aware Decisions**: The agent can reason about previous observations to make better decisions in subsequent steps

#### Example Flow

```python
intermediate_steps = []  # Start with empty history

# First iteration
agent_step = AgentAction(tool='get_text_length', tool_input='DOG', log='...')
observation = '3'
intermediate_steps.append((agent_step, observation))
# intermediate_steps = [(AgentAction(...), '3')]

# Second iteration - agent can see it already called get_text_length
# The scratchpad includes: "Action: get_text_length\nAction Input: DOG\nObservation: 3\n"
# So the agent knows the result and can proceed to final answer
```

#### How It's Used

The `intermediate_steps` are converted to a string format via `format_log_to_str()` and inserted into the prompt as the `agent_scratchpad`:

```python
agent = {
    "input": lambda x: x["input"], 
    "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])
} | prompt | llm | ReActSingleInputOutputParser()
```

Each iteration, the LLM sees the complete history:
```
Question: What is the text length of the string DOG?
Thought: I should use get_text_length
Action: get_text_length
Action Input: DOG
Observation: 3
Thought: I now know the final answer
Final Answer: The text length is 3
```

Without `intermediate_steps`, the agent would have no memory and might try the same action repeatedly!

## ⚠️ Limitations and Reliability

While the ReAct pattern is foundational for agentic behavior and AI agents, it's important to understand its limitations.

### Prompt-Based Parsing Fragility

**The Challenge**: This ReAct agent implementation depends heavily on the LLM generating responses in a specific format. The response is parsed using regular expressions in LangChain's `ReActSingleInputOutputParser`.

**The Problem**: 
- If the LLM generates even **one wrong token**, it can break the entire parsing process
- The parser expects exact keywords like `Action:`, `Action Input:`, `Thought:`, etc.
- A small deviation (e.g., `Actions:` instead of `Action:`, or extra whitespace) will cause parsing to fail
- The LLM might not always follow the format perfectly, especially with:
  - Complex queries
  - Less capable models
  - Higher temperature settings
  - Ambiguous instructions

**Example of Failure**:
```
# Expected format:
Action: get_text_length
Action Input: DOG

# What LLM might generate:
Actions: get_text_length    ← Extra 's' breaks parser!
Input: DOG                   ← Missing "Action" prefix breaks parser!
```

### Why This Matters

- **Not Production-Ready Without Safeguards**: Prompt-based agents can be unreliable in production environments
- **Error Handling Required**: You should wrap agent execution in try-catch blocks
- **Model Selection Important**: More capable models (like GPT-4) follow instructions better than smaller models
- **Temperature Matters**: Lower temperature (0-0.3) produces more consistent formatting

### The Solution: Function/Tool Calling

**Function calling (also called tool calling)** is the modern solution to the reliability issues of prompt-based parsing.

#### How Function Calling Solves the Problem

Instead of relying on the LLM to output text in a specific format that gets parsed with regex, function calling:

1. **Structured Output**: The LLM returns a structured JSON object, not free-form text
2. **No Parsing Errors**: No regex parsing means no formatting errors
3. **Type Safety**: Function parameters are validated against schemas
4. **Built-in Support**: Native support from OpenAI, Anthropic, Google, and other LLM providers

#### Example Comparison

**Old Way (Prompt-Based Parsing):**
```python
# LLM generates text:
"Action: get_text_length\nAction Input: DOG"
# Parser uses regex to extract action and input ❌ Fragile!
```

**New Way (Function Calling):**
```python
# LLM returns structured output:
{
  "name": "get_text_length",
  "arguments": {"text": "DOG"}
}
# No parsing needed! ✅ Reliable!
```

#### Modern Implementation Approaches

1. **OpenAI Function Calling**: Use `tools` parameter with ChatGPT API
2. **Anthropic Tool Use**: Claude's native tool use capability
3. **LangChain with Function Calling**: LangChain supports function calling through `.bind_tools()`
4. **Agent Frameworks**: LangGraph and other frameworks built on function calling

### Recommendation

This implementation serves as an excellent **learning tool** to understand the fundamentals of ReAct agents and how the reasoning loop works. 

**For production use cases**, use function/tool calling instead:
- ✅ **Reliable**: No parsing errors from malformed text
- ✅ **Type-safe**: Parameters validated against schemas
- ✅ **Industry standard**: Used by production AI applications
- ✅ **Better performance**: More efficient than text parsing

Despite the limitations of prompt-based parsing, understanding the ReAct pattern is crucial as it forms the conceptual foundation for all agent systems, whether they use prompt-based parsing or modern function calling.

## 🛠️ Customization

### Adding New Tools

Define a new tool using the `@tool` decorator:

```python
@tool
def your_custom_tool(input: str) -> str:
    """Description of what your tool does"""
    # Your implementation
    return result
```

Then add it to the tools list:
```python
tools = [get_text_length, your_custom_tool]
```

For more information on creating custom tools for agents, see the [LangChain Tools Documentation](https://docs.langchain.com/oss/python/langchain/tools).

### Modifying the Question

Change the input in the agent invocation:
```python
agent_step = agent.invoke({
    "input": "Your question here?",
    "agent_scratchpad": intermediate_steps
})
```

### Adjusting LLM Parameters

Modify the ChatOpenAI initialization:
```python
llm = ChatOpenAI(
    temperature=0,  # Lower = more deterministic
    model="gpt-4",  # Specify model
    stop=["\nObservation", "Observation"],  # Critical: prevents hallucinated observations
    callbacks=[MyCallbackHandler()]
)
```

**Important**: The `stop` parameter is crucial for preventing the LLM from hallucinating observations. See the [Understanding "Observation" and Stop Tokens](#understanding-observation-and-stop-tokens) section for details.

## 📦 Dependencies

- **langchain**: Framework for building LLM applications
- **langchain-openai**: OpenAI integration for LangChain
- **langchain-core**: Core LangChain functionality
- **openai**: OpenAI API client
- **python-dotenv**: Environment variable management
- **black**: Code formatting (development)

## 🐛 Debugging

The custom callback handler in `callbacks.py` provides detailed output:
- **on_llm_start**: Shows the full prompt sent to the LLM
- **on_llm_end**: Shows the LLM's response

Enable or disable by adding/removing from the LLM initialization:
```python
# With callbacks
llm = ChatOpenAI(callbacks=[MyCallbackHandler()])

# Without callbacks
llm = ChatOpenAI()
```

## 📚 Learn More

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain Tools Documentation](https://docs.langchain.com/oss/python/langchain/tools) - Guide to creating custom tools for agents
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [ReAct Prompt Template on LangSmith Hub](https://smith.langchain.com/hub/hwchase17/react?organizationId=d924c6d5-9297-4b9b-9cdb-df9eea6b95a7) - Original prompt template used in this implementation by [Harrison Chase](https://blog.langchain.com/author/harrison/), creator of LangChain

## 🤝 Contributing

Feel free to open issues or submit pull requests to improve this implementation!

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Powered by [OpenAI](https://openai.com/)
- Inspired by the ReAct paper (Yao et al., 2022)
