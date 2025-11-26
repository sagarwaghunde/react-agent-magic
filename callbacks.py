from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
# from langchain.schema import LLMResult
import typing as t

# https://docs.langchain.com/oss/javascript/langchain/overview
class MyCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized: dict[str, t.Any], prompts: list[str], **kwargs: t.Any) -> t.Any:
        """Runs when the LLM starts executing"""
        print(f"***Prompt to LLM was:*** {prompts[0]}")
        print("************************************************")
        return None

    def on_llm_end(self, response: LLMResult, **kwargs: t.Any) -> t.Any:
        """Runs when the LLM has finished executing"""
        print(f"LLM response: {response.generations[0][0].text}")
        print("************************************************")
        return None

    # def on_llm_error(self, error: Exception | KeyboardInterrupt, **kwargs: t.Any) -> t.Any:
    #     """Runs when the LLM fails to execute"""
    #     print(f"LLM error: {error}")
    #     print("************************************************")
    #     return None

    # def on_chain_start(self, serialized: dict[str, t.Any], inputs: dict[str, t.Any], **kwargs: t.Any) -> t.Any:
    #     """Runs when a chain starts executing"""
    #     print(f"***Chain inputs were:*** {inputs}")
    #     print("************************************************")
    #     return None
