from langchain_core.tools import render_text_description
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents.output_parsers.react_single_input import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from typing import Union, List, Tuple
from langchain.tools import Tool
from callbacks import MyCallbackHandler

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with text: {text}")
    text = text.strip("\n").strip('"')
    return len(text)


def find_tool_by_name(name: str, tools: List[Tool]) -> Tool:
    for tool in tools:
        if tool.name == name:
            return tool
    return ValueError(f"Tool with name {name} not found")

def format_log_to_str(
    intermediate_steps: List[Tuple[AgentAction, str]],
    observation_prefix: str = "Observation: ",
    llm_prefix: str = "Thought: "
) -> str:
    """Construct a scratchpad that lets agent continue its thought process."""

    thought = ""
    for action, observation in intermediate_steps:
        thought += action.log
        thought += f"{observation_prefix}{observation}\n{llm_prefix}"
    return thought

if __name__ == "__main__":
    print("Hello ReAct LangChain!")
    tools = [get_text_length]
    template = """
    Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
    """
    prompt = PromptTemplate(template=template).partial(tools=render_text_description(tools), tool_names=", ".join([t.name for t in tools]))
    # llm    = ChatOpenAI(temperature=0, stop=["\nObservation", "Observation"])
    # below llm object is with callbacks
    llm    = ChatOpenAI(temperature=0, stop=["\nObservation", "Observation"], callbacks=[MyCallbackHandler()])
    # below is the basic part of reasong before parsing. 
    # agent  = {"input": lambda x: x["input"]} | prompt | llm 
    # reasoning = agent.invoke({"input": "What is the length of the string 'DOG' in characters? "})
    # print("Reasoning: ", reasoning)

    # copy past above lines and change the agent to use the parser
    intermediate_steps = []
    # agent  = {
    #         "input": lambda x: x["input"], "agent_scratchpad": lambda x: x["agent_scratchpad"]} | prompt | llm | ReActSingleInputOutputParser()

# change the agent to use the scratchpad    agent_scratchpad = format_log_to_str(intermediate_steps)
    agent  = {
            "input": lambda x: x["input"], "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])} | prompt | llm | ReActSingleInputOutputParser()
    # res = agent.invoke({"input": "What is the text length of the string 'DOG' in characters? "})
    # print(res)

    # output parser creates the object of AgentAction or AgentFinish

    agent_step = ""
    while not isinstance(agent_step, AgentFinish):
        # agent_step = agent.invoke(
        #     {
        #         "input": "What is the text length of the string DOG in characters? ",
        #         "agent_scratchpad": intermediate_steps
        #     }
        # )
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the text length of the string DOG in characters? ",
                "agent_scratchpad": intermediate_steps
            }
        )
        print(f"Agent step in while loop: {agent_step}")
        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool = find_tool_by_name(tool_name, tools)
            tool_input = agent_step.tool_input
            observation = tool.func(str(tool_input))
            print(f"Observation: {observation}")
            intermediate_steps.append((agent_step, str(observation)))

    

    if(isinstance(agent_step, AgentFinish)):
        print(f"Agent finished with the final answer: {agent_step.return_values}")
