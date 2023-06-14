---
title: 使用llama_index实现chatGPT智能问答机器人
comments: true
categories: AI
tags: '-- AI -- ChatGPT -- AI智能客服 '
date: 2023-06-14 20:04:22
--- 
# Based

* Langchain

* llama_index >=0.6.5

* GPT-3.5

* websockets

* python >=3.10



**dependence**

```python
pip install llama-index

pip install openai 

pip install langchain

pip install websockets

pip install pandas

pip install llama-hub
```





# What use

Supporting private knowledge base AI question-answering chatbot, capable of both knowledge-based Q&A and casual conversation.

<!--more-->

# AI Chatbot



```python
from llama_index import SimpleDirectoryReader, ServiceContext, GPTVectorStoreIndex, PromptHelper,StorageContext,load_index_from_storage
from llama_index.llm_predictor.chatgpt import ChatGPTLLMPredictor
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from llama_index.langchain_helpers.agents import IndexToolConfig, LlamaIndexTool,LlamaToolkit

from llama_index.langchain_helpers.agents import create_llama_chat_agent,create_llama_agent
from llama_index.langchain_helpers.memory_wrapper import GPTIndexChatMemory
os.environ["OPENAI_API_KEY"] = 'sk-yourOpenAiKey'
from langchain.memory import ConversationBufferMemory
from llama_index import download_loader
from pathlib import Path


def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 2000 
    #If you don't want a stream output,set streaming=False
    llm=ChatOpenAI(temperature=0.4, model_name="gpt-3.5-turbo", verbose=True,streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
    llm_predictor = ChatGPTLLMPredictor(llm=llm)
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
 
    files = os.listdir(directory_path)
    if len(files) < 1:
        print('error!'+directory_path+' 文件夹下文件为空！')
        return
    file_name = files[0]
    print("load index data :"+directory_path+'/'+file_name)
    # If the data is in Excel format
    if file_name.endswith(".xls") or file_name.endswith(".xlsx"):
        PandasExcelReader = download_loader("PandasExcelReader")
        loader = PandasExcelReader()
        loader._pandas_config={"header":0}
        #源码有问题，不设置默认值会报错
        #loader._concat_rows = True
        loader._row_joiner = '   '
        documents = loader.load_data(file=Path(directory_path+'/'+file_name),sheet_name=None)
    else:
        documents = SimpleDirectoryReader(directory_path).load_data()
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper) #llm_predictor=llm_predictor, 
    index = GPTVectorStoreIndex.from_documents(documents,service_context=service_context)
    #save on disk
    index.storage_context.persist("D://storage")

    
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir='D://storage')
    # load index
    index = load_index_from_storage(storage_context)
    #response_mode="compact",
    query_engine = index.as_query_engine(similarity_top_k=30,service_context=service_context) #query_engine = index.as_query_engine(similarity_top_k=1,streaming=True,service_context=service_context)

    #If you want to save and use the conversation record in document format, you should use GPTindexChatMemory.
    # memory = GPTIndexChatMemory(
    #     index=chat_history_index, 
    #     memory_key="chat_history", 
    #     query_kwargs={"response_mode": "compact","streaming":True,"service_context":service_context,"similarity_top_k":1}, #,"streaming":True
    #     # return_source returns source nodes instead of querying index
    #     return_source=True,
    #     # return_messages returns context in message format
    #     return_messages=True,
    # )

    #use memory save the chat history
    memory = ConversationBufferMemory(
        memory_key="chat_history"
    )

    tool_config = IndexToolConfig(
        query_engine=query_engine, 
        name=f"AI Customer service",
        description=f"If it is a game-related question, please use tools to obtain information before answering.If this question is about a game and you don't know the answer, jst say 'sorry,i don't know'",
        tool_kwargs={"return_direct": True}
    )

    tool = LlamaIndexTool.from_tool_config(tool_config)
    toolkit = LlamaToolkit(
        index_configs=[tool],
    )


    agent_chain = create_llama_chat_agent(
        toolkit,
        llm,
        memory=memory,
        verbose=True,
        agent_kwargs={"max_iterations":3}
    )
    
    while True: 
        query = input("What do you want to ask? ")
        print(agent_chain.memory.chat_memory.messages)
        response_stream = agent_chain.run(query)
        #if use streaming
        if  hasattr(response_stream,'response_gen'):
                for text in response_stream.response_gen:
                    print(text, end="")
                    #todo send to client
        else:
            print(response_stream)
            
#This path is the file path of your knowledge base
service_context=construct_index('D://data')





```



# Integrate WebSocket

Receive questions from the client and have AI generate answers.

### websocket servet.py

```python
from llama_index import SimpleDirectoryReader, ServiceContext, PromptHelper,StorageContext,load_index_from_storage,GPTVectorStoreIndex
from langchain.chat_models import ChatOpenAI
from CustomCallbackHandler import CustomAsyncCallBackHandler
from llama_index.llm_predictor.chatgpt import ChatGPTLLMPredictor
from llama_index.langchain_helpers.agents import create_llama_chat_agent
from llama_index.langchain_helpers.memory_wrapper import GPTIndexChatMemory
from llama_index.langchain_helpers.agents import IndexToolConfig, LlamaIndexTool,LlamaToolkit
from langchain.memory import ConversationBufferMemory
from llama_index import download_loader
from pathlib import Path
import os
os.environ["OPENAI_API_KEY"] = 'sk-xxx'
import json

import asyncio
import websockets
import logging
import sys


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

agent_cache={}
connecte_session = {}
session_cache={}
def build_llama_chat_agent(directory_path,project_id,session_id,prompt_name,advanced_description):
    if session_id not in agent_cache:
        storage_path = '/data/index_cache/'+project_id+'/storage'
        # set maximum input size
        max_input_size = 4096
        # set number of output tokens
        num_outputs = 2000
        # set maximum chunk overlap
        max_chunk_overlap = 20
        # set chunk size limit
        chunk_size_limit = 2000 

        llm=ChatOpenAI(temperature=0.4, model_name="gpt-3.5-turbo", verbose=False,streaming=True)
        llm_predictor = ChatGPTLLMPredictor(llm=llm)
        prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    
        
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper) #llm_predictor=llm_predictor, 
        if os.path.exists(storage_path):
            
            # rebuild storage context
            storage_context = StorageContext.from_defaults(persist_dir=storage_path)
            # load index
            index = load_index_from_storage(storage_context)

            agent_cahin= get_chat_agent(index,service_context,llm,prompt_name,advanced_description)
            agent_cache[session_id] = agent_cahin
            return agent_cahin
        else:
            files = os.listdir(directory_path)
            if len(files) < 1:
                print('error!'+directory_path+' 文件夹下文件为空！')
                return
            file_name = files[0]
            logging.info("load data :"+directory_path+'/'+file_name)
            # if excel data format
            if file_name.endswith(".xls") or file_name.endswith(".xlsx"):
                PandasExcelReader = download_loader("PandasExcelReader")
                loader = PandasExcelReader()
                loader._pandas_config={"header":0}
                #loader._concat_rows = True
                loader._row_joiner = '   '
                documents = loader.load_data(file=Path(directory_path+'/'+file_name),sheet_name=None)
            else:
                documents = SimpleDirectoryReader(directory_path).load_data()
            index = GPTVectorStoreIndex.from_documents(documents,service_context=service_context)
            #save on disk default ./storage
            index.storage_context.persist(storage_path)
            agent_cache[session_id] = get_chat_agent(index,service_context,llm,prompt_name,advanced_description)
            return agent_cache[session_id]
    else:
        return agent_cache[session_id] 

def get_chat_agent(index,service_context,llm,prompt_name,advanced_description):
    # memory = GPTIndexChatMemory(
    #     index=index, 
    #     memory_key="chat_history", 
    #     query_kwargs={"response_mode": "compact","streaming":True,"service_context":service_context,"similarity_top_k":1}, #,"streaming":True
    #     # return_source returns source nodes instead of querying index
    #     return_source=True,
    #     # return_messages returns context in message format
    #     return_messages=True,
    #     #chat_memory=ConversationBufferWindowMemory(return_messages=True)
    # )
    #should be create memorys for each user
    memory = ConversationBufferMemory(
        memory_key="chat_history"
    )

    prompt_name=prompt_name if prompt_name is not None else f"AI virtual anchor"
    advanced_description = advanced_description if advanced_description is not None else f"You are an AI virtual anchor. If it is a game-related question, please use tools to obtain information before answering.If this question is about a game and you don't know the answer, jst say 'sorry,i don't know'. Remember must respond in Chinese."
    print('prompt name:'+prompt_name+" ,description:"+advanced_description)
    tool_config = IndexToolConfig(
        query_engine=index.as_query_engine(
                response_mode="compact",
                streaming=True,
                similarity_top_k=1,service_context=service_context), 
        name=prompt_name,
        description=advanced_description,
        tool_kwargs={"return_direct": True,"return_sources": True},
        return_sources= True
    )

    tool = LlamaIndexTool.from_tool_config(tool_config)
    toolkit = LlamaToolkit(
        index_configs=[tool],
    )

    agent_chain = create_llama_chat_agent(
        toolkit,
        llm,
        memory=memory,
        verbose=True  
    )
    
    return agent_chain
       

async def send_message(websocket, message_queue):
    while True:
        
        message = await message_queue.get()

      
        await websocket.send(message)
        # print('Sent message: %s' % message)

       
        message_queue.task_done()

async def receive_messages(websocket,message_queue,params_json):
    # data path
    direct_path = params_json['input_dir']
    
    project_id=params_json["project_id"]
    session_id = params_json["session_id"]

    connecte_session[hash(websocket)]=session_id
    
    chat_agent = build_llama_chat_agent(direct_path,project_id,session_id,None,None)
    print(chat_agent.memory.chat_memory.messages)
    query = params_json["data"]["question"]

    try:
            response_stream = await chat_agent.arun(
                query,
                callbacks=[CustomAsyncCallBackHandler(message_queue)]
            )
            
            logging.info('ai response：'+response_stream)
    except ValueError as e:
            response_stream = str(e)
            if not response_stream.startswith("Could not parse LLM output: `"):
                raise e
            response_stream = response_stream.removeprefix("Could not parse LLM output: `").removesuffix("`")
            logging.error("error ："+response_stream)



import asyncio
import websockets


# 定义websocket服务器回调函数
async def websocket_server(websocket, path):
    global count
    print('Client connected.')
    try:
        # 创建消息队列
        message_queue = asyncio.Queue()
        # 启动发送消息和接收消息的协程
        send_task = asyncio.create_task(send_message(websocket, message_queue))
        # 接收客户端发送的消息
        async for message in websocket:
            print('Received message: %s' % message)
            data = json.loads(message)
            await receive_messages(websocket,message_queue,data)
    except websockets.exceptions.ConnectionClosed:
        #根据websocket对象获取session_id
        session_id = connecte_session[hash(websocket)]
        #根据session_id删除缓存的代理对象
        del agent_cache[session_id]
        print('Client : %s disconnected.' % session_id)
    finally:
        # 关闭websocket连接
        await websocket.close()
        #connected.remove(websocket)
        # 取消发送消息和接收消息的协程
        send_task.cancel()

        # 等待所有任务和协程完成
        await asyncio.gather(send_task, return_exceptions=True)

if __name__ == "__main__":
    # 启动websocket服务器
    start_server = websockets.serve(websocket_server, '0.0.0.0', 9961)
    print('websocket_server now started listening on port: 9961')

    # 运行事件循环
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
```



### custom callback handler.py

Return AI's response to the WebSocket client

```python
from typing import Any, Union
from asyncio.queues import Queue
from langchain.callbacks.base import AsyncCallbackHandler
from typing import Any, Dict, List, Optional
from langchain.schema import LLMResult
import json
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
DEFAULT_ANSWER_PREFIX_TOKENS = ["\n", "AI", ":"]

class CustomAsyncCallBackHandler(AsyncCallbackHandler):
    queue: Queue
    
    
    def __init__(self,queue:Queue, answer_prefix_tokens: Optional[List[str]] = None) -> None:
        super().__init__()
        if answer_prefix_tokens is None:
            answer_prefix_tokens = DEFAULT_ANSWER_PREFIX_TOKENS
        self.answer_prefix_tokens = answer_prefix_tokens
        self.last_tokens = [""] * len(answer_prefix_tokens)
        self.answer_reached = False
        self.queue=queue


    async def put_message(self,json_str):
        await self.queue.put(json.dumps(json_str))
        await self.queue.join()

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        self.answer_reached = False

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
         # Remember the last n tokens, where n = len(answer_prefix_tokens)
        self.last_tokens.append(token)
        if len(self.last_tokens) > len(self.answer_prefix_tokens):
            self.last_tokens.pop(0)

        # Check if the last n tokens match the answer_prefix_tokens list ...
        if self.last_tokens == self.answer_prefix_tokens:
            self.answer_reached = True
            # Do not print the last token in answer_prefix_tokens,
            # as it's not part of the answer yet
            return

        # ... if yes, then print tokens from now on
        if self.answer_reached:
            response = str(token)
            await self.put_message(response)

        

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        if self.answer_reached:
            response = "[DONE]"
            await self.put_message(response)

    async def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors."""



```
