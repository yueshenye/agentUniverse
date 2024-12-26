# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2024/12/12 20:58
# @Author  : jijiawei
# @Email   : jijiawei.jjw@antgroup.com
# @FileName: pet_insurance_rewrite_agent.py
from agentuniverse.agent.agent import Agent
from agentuniverse.agent.input_object import InputObject
from agentuniverse.base.util.logging.logging_util import LOGGER
from agentuniverse.base.util.prompt_util import process_llm_token
from agentuniverse.llm.llm import LLM
from agentuniverse.prompt.prompt import Prompt
from langchain_core.output_parsers import StrOutputParser


class PetInsuranceRewriteAgent(Agent):

    def input_keys(self) -> list[str]:
        return ['input', 'prod_description']

    def output_keys(self) -> list[str]:
        return ['rewrite_output']

    def parse_input(self, input_object: InputObject, agent_input: dict) -> dict:
        agent_input['input'] = input_object.get_data('input')
        agent_input['prod_description'] = input_object.get_data('prod_description')
        return agent_input

    def parse_result(self, agent_result: dict) -> dict:
        rewrite_output = agent_result['output']
        LOGGER.info(f'智能体 pet_question_planning_agent 执行结果为： {rewrite_output}')
        return {**agent_result, 'rewrite_output': agent_result['output']}

    def execute(self, input_object: InputObject, agent_input: dict, **kwargs) -> dict:
        llm: LLM = self.process_llm(**kwargs)
        prompt: Prompt = self.process_prompt(agent_input, **kwargs)
        process_llm_token(llm, prompt.as_langchain(), self.agent_model.profile, agent_input)
        chain = prompt.as_langchain() | llm.as_langchain_runnable(
            self.agent_model.llm_params()) | StrOutputParser()
        res = self.invoke_chain(chain, agent_input, input_object, **kwargs)
        return {**agent_input, 'output': res}
