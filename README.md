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
4. **Scratchpad**: Maintains history of thoughts and observations
5. **Callback Handler**: Monitors LLM interactions for debugging

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
