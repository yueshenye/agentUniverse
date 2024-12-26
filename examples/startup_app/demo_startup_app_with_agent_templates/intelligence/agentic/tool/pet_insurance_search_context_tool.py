# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2024/11/12 11:59
# @Author  : wangchongshi
# @Email   : wangchongshi.wcs@antgroup.com
# @FileName: search_context_tool.py
import json

from agentuniverse.agent.action.tool.tool import Tool, ToolInput
from agentuniverse.base.util.logging.logging_util import LOGGER

PRE_API_URL = "www.xxxx.com/query_knowledge"


class MockAPI:
    def post(self, url, headers, data):
        # mock response
        mock_response = {
            "result": {
                "recallResultTuples": [
                    {
                        "knowledgeTitle": "mock data: 宠物医保（体验版）简介",
                        "content": "宠物医保（体验版）是免费体验版，仅有30天保障时间。宠物医保（体验版）体验30天后付费可升级成宠物医保升级版。"
                    },
                    {
                        "knowledgeTitle": "mock data: 宠物医保简介",
                        "content": "宠物医保保障期限12个月，是付费版商业险，有三个保障：基础版、升级版、尊享版。"
                    }
                ]
            }
        }
        return MockResponse(mock_response)


class MockResponse:
    def __init__(self, json_data):
        self.json_data = json_data

    def json(self):
        return self.json_data


class SearchContextTool(Tool):

    def execute(self, tool_input: ToolInput):
        question = tool_input.get_data('input')
        try:
            headers = {
                "Content-Type": "application/json"
            }
            # 要发送的数据
            data = {
                "chatId": "xxxx",
                "sessionId": "xxxx",
                "userId": "xxxxx",
                "sceneCode": "xxxx",
                "query": question,
                "decoderType": "xxxx",
                "inputMethod": "user_input",
                "enterScene": {
                    "sceneCode": "xxx",
                    "productNo": "xxxx",
                }
            }
            top_k = tool_input.get_data('top_k') if tool_input.get_data('top_k') else 2
            LOGGER.info(f"search context tool input: {data}")
            response = MockAPI().post(PRE_API_URL, headers=headers, data=json.dumps(data, ensure_ascii=False))
            result = response.json()['result']
            recallResultTuples = result.get('recallResultTuples')

            context = f"提出的问题是:{question}\n\n这个问题检索到的答案相关内容是:\n\n"
            index = 0
            for recallResult in recallResultTuples:
                if index == top_k:
                    return context
                if recallResult.get('content'):
                    context += (f"knowledgeTitle: {recallResult.get('knowledgeTitle')}\n"
                                f"knowledgeContent: {recallResult.get('content')}\n\n")
                    index += 1
            return context
        except Exception as e:
            LOGGER.error(f"invoke search context tool failed: {str(e)}")
            raise e
