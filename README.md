# ReAct LangChain Agent

A Python implementation of the ReAct (Reasoning + Acting) pattern using LangChain and OpenAI. This project demonstrates how to build an intelligent agent that can reason about problems and use tools to solve them.

## 📋 Overview

This project implements a ReAct agent that follows a thought-action-observation loop to answer questions. The agent can use external tools (like calculating text length) and maintains a scratchpad of its reasoning process to arrive at the correct answer.

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

1. **Prompt Template**: Structures the agent's reasoning format
2. **Tools**: Functions the agent can call (e.g., `get_text_length`)
3. **Output Parser**: Parses LLM output into `AgentAction` or `AgentFinish`
4. **Scratchpad**: Maintains history of thoughts and observations
5. **Callback Handler**: Monitors LLM interactions for debugging

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
    stop=["\nObservation", "Observation"],
    callbacks=[MyCallbackHandler()]
)
```

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
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [OpenAI API Documentation](https://platform.openai.com/docs)

## 🤝 Contributing

Feel free to open issues or submit pull requests to improve this implementation!

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Powered by [OpenAI](https://openai.com/)
- Inspired by the ReAct paper (Yao et al., 2022)
