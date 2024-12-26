# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2024/12/26 18:26
# @Author  : wangchongshi
# @Email   : wangchongshi.wcs@antgroup.com
# @FileName: pet_insurance_consult_agent.py
import json

from agentuniverse.agent.agent_manager import AgentManager
from langchain_core.output_parsers import StrOutputParser
from agentuniverse.agent.action.tool.tool import Tool
from agentuniverse.agent.action.tool.tool_manager import ToolManager
from agentuniverse.agent.agent import Agent
from agentuniverse.agent.input_object import InputObject
from agentuniverse.agent.memory.memory import Memory
from agentuniverse.base.util.agent_util import assemble_memory_input, assemble_memory_output
from agentuniverse.base.util.logging.logging_util import LOGGER
from agentuniverse.base.util.prompt_util import process_llm_token
from agentuniverse.llm.llm import LLM
from agentuniverse.prompt.prompt import Prompt


class PetInsuranceConsultAgent(Agent):

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

    def execute(self, input_object: InputObject, agent_input: dict, **kwargs) -> dict:
        # 1. get the memory instance.
        memory: Memory = self.process_memory(agent_input, **kwargs)
        # 2. get the llm instance.
        llm: LLM = self.process_llm(**kwargs)
        # 3. get the agent prompt.
        prompt: Prompt = self.process_prompt(agent_input, **kwargs)
        # 4. assemble the memory input.
        assemble_memory_input(memory, agent_input)

        # 5. rewrite query.
        detail_tool = ToolManager().get_instance_obj('pet_insurance_info_tool')
        tool_res = detail_tool.run(query='宠物医保')
        agent_input['prod_description'] = tool_res
        rewrite_agent: Agent = AgentManager().get_instance_obj('pet_question_rewrite_agent')
        rewrite_agent_res = rewrite_agent.run(**agent_input)
        agent_input['rewrite_question'] = rewrite_agent_res.get_data('rewrite_output')

        # 6. planning query.
        planning_agent_res = AgentManager().get_instance_obj('pet_question_planning_agent').run(**agent_input)
        split_questions = planning_agent_res.get_data('planning_output')
        sub_query_list = json.loads(split_questions).get('sub_query_list')

        # 7. execute the search api.
        search_tool: Tool = ToolManager().get_instance_obj('pet_insurance_search_context_tool')
        search_res = ''
        for sub_query in sub_query_list:
            search_res += search_tool.run(input=sub_query) + '\n'
        agent_input['search_context'] = search_res
        LOGGER.info(f'tool api search result is: {search_res}')

        # 8. invoke agent.
        process_llm_token(llm, prompt.as_langchain(), self.agent_model.profile, agent_input)
        chain = prompt.as_langchain() | llm.as_langchain_runnable(
            self.agent_model.llm_params()) | StrOutputParser()
        res = self.invoke_chain(chain, agent_input, input_object, **kwargs)

        # 9. assemble the memory output.
        assemble_memory_output(memory=memory,
                               agent_input=agent_input,
                               content=f"Human: {agent_input.get('input')}, AI: {res}")
        return {**agent_input, 'output': res}
