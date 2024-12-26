# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2024/12/12 22:54
# @Author  : jijiawei
# @Email   : jijiawei.jjw@antgroup.com
# @FileName: PetInsuranceRagAgentTemplate.py
from langchain_core.output_parsers import StrOutputParser
from agentuniverse.agent.input_object import InputObject
from agentuniverse.agent.memory.memory import Memory
from agentuniverse.agent.template.agent_template import AgentTemplate
from agentuniverse.base.util.agent_util import assemble_memory_input, assemble_memory_output
from agentuniverse.base.util.prompt_util import process_llm_token
from agentuniverse.llm.llm import LLM
from agentuniverse.prompt.prompt import Prompt


class PetRagAgentTemplate(AgentTemplate):

    def input_keys(self) -> list[str]:
        """Return the input keys of the Agent."""
        return ['input']

    def output_keys(self) -> list[str]:
        """Return the output keys of the Agent."""
        return ['output']

    def parse_input(self, input_object: InputObject, agent_input: dict) -> dict:
        agent_input['input'] = input_object.get_data('input')
        return agent_input

    def parse_result(self, agent_result: dict) -> dict:
        return agent_result

    def customized_execute(self, input_object: InputObject, agent_input: dict, memory: Memory, llm: LLM, prompt: Prompt,
                           **kwargs) -> dict:
        # 1. assemble the memory input.
        assemble_memory_input(memory, agent_input)
        # 2. process the prompt tokens.
        process_llm_token(llm, prompt.as_langchain(), self.agent_model.profile, agent_input)
        # 3. invoke agent.
        chain = prompt.as_langchain() | llm.as_langchain_runnable(
            self.agent_model.llm_params()) | StrOutputParser()
        res = self.invoke_chain(chain, agent_input, input_object, **kwargs)
        # 4. assemble the memory output.
        assemble_memory_output(memory=memory,
                               agent_input=agent_input,
                               content=f"Human: {agent_input.get('input')}, AI: {res}")
        # 5. return result.
        return {**agent_input, 'output': res}
