import os
import re
import tempfile
import time
import openai
import faiss
from typing import List, Union

from llama_index.core import StorageContext, Settings, load_index_from_storage
from llama_index.core import SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import RouterQueryEngine, SubQuestionQueryEngine, CitationQueryEngine
from llama_index.core.langchain_helpers.agents.tools import IndexToolConfig, LlamaIndexTool
from llama_index.core.agent import ReActAgent
from llama_index.vector_stores.faiss import FaissVectorStore

from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import nest_asyncio
nest_asyncio.apply()

class RAG:

    names = ["The Insurance Act, 1938: Regulations and Restrictions for Insurance Companies in India"]
    names.append("The Consumer Protection Act, 1986")
    names.append("The Employees' State Insurance Act, 1948")
    names.append("Insurance Regulatory and Development Authority of India (Health Insurance) Regulations, 2016")
    names.append("The Transplantation of Human Organs and Tissues Act, 1994")
    descriptions = ["The go-to document for Insurance Rules. The Insurance Act, 1938 is an Act to consolidate and amend the law relating to the business of insurance in India. It outlines the regulations for insurance companies, including registration, capital requirements, investment, loans and management, investigation, appointment of staff, control over management, amalgamation and transfer of insurance business, commission and rebates, licensing of agents, management by administrator, and acquisition of the undertakings of insurers in certain cases. It also outlines the voting rights of shareholders, the requirements for making a declaration of interest in a share held in the name of another person, the requirements for the separation of accounts and funds for different classes of insurance business, the audit and actuarial report and abstract that must be conducted annually, the power of the Authority to order revaluation and to inspect returns, the power of the Authority to make rules and regulations, the power of the Authority to remove managerial persons from office, appoint additional directors, and issue directions regarding re-insurance treaties, the power of the Authority to enter and search any building or place where books, accounts, or other documents relating to any claim, rebate, or commission are kept, the prohibition of cessation of payments of commission, the prohibition of offering of rebates as an inducement to take out or renew an insurance policy, the process for issuing a registration to act as an intermediary or insurance intermediary, the process for repudiating a life insurance policy on the ground of fraud, the prohibition of insurance agents, intermediaries, or insurance intermediaries to be or remain a director in an insurance company, the requirement to give notice to the policy-holder informing them of the options available to them on the lapsing of a policy, and the power of the National Company Law Tribunal to order the winding up of an insurance company. Penalties for non-compliance range from fines to imprisonment. The Act also outlines the formation of the Life Insurance Council and General Insurance Council, and the Executive Committees of each, the Tariff Advisory Committee, and the obligations of insurers in respect of rural or social or unorganized sector and backward classes."]
    descriptions.append("The Consumer Protection Act, 1986 is an Act that provides better protection for the interests of consumers in India, except for the State of Jammu and Kashmir. It defines a consumer as any person who buys goods or services for a consideration. The Act outlines the composition and jurisdiction of District Forums, State Commissions, and the National Commission, which are responsible for resolving consumer disputes. It also provides protection of action taken in good faith, and lays out rules for the Central and State Governments.")
    descriptions.append("The Employees' State Insurance Act, 1948 is an Act that provides benefits to employees in case of sickness, maternity, and employment injury. It outlines the duties of the Central Government, the Medical Benefit Council, the Director-General and the Financial Commissioner, and the Corporation. It also outlines the benefits that insured persons, their dependants, or other persons are entitled to, such as sickness benefit, maternity benefit, disablement benefit, dependants' benefit, and medical benefit. Additionally, the Act outlines the provisions for medical benefits for insured persons and their families, as well as other beneficiaries. It also provides for the exemption of a factory or establishment or class of factories or establishments in any specified area from the operation of this Act, and outlines the punishments for failure to pay contributions.")
    descriptions.append("The Insurance Regulatory and Development Authority of India (Health Insurance) Regulations, 2016 outlines the regulations for health insurance policies, including provisions for portability, AYUSH coverage, wellness and preventive aspects, standard definitions of terms, optional coverage for certain items, and special provisions for senior citizens. The regulations also provide details on the product filing procedure, underwriting, proposal forms, premiums, free look period, and pre-insurance health check-ups. Additionally, the regulations outline the process for the offering of 'Combi Products' which are a combination of life insurance and health insurance, as well as the responsibilities of insurers and TPAs in providing services to policyholders.")
    descriptions.append("The Transplantation of Human Organs and Tissues Act, 1994 outlines the regulations and procedures for the removal, preservation, and distribution of human organs and tissues for therapeutic purposes. It defines a donor as any person over 18 who voluntarily authorises the removal of any of their organs, and outlines the duties of the registered medical practitioner and the Appropriate Authority. It also prohibits the removal or transplantation of organs or tissues for any purpose other than therapeutic, and outlines the punishment for illegal dealings in human tissues.")
    
    def query_engine(self):
        query_engine_tools = []
        temp = ['insurance', 'cpa', 'tesia', 'iradaoi', 'ttohoata']
        for n, x in enumerate(temp):
            path = os.path.join("./MultiDocQA/",x)
            vector_store = FaissVectorStore.from_persist_dir(persist_dir=path)
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=path
            )
            index = load_index_from_storage(storage_context=storage_context)
            engine = index.as_query_engine(similarity_top_k=3)
            query_engine_tools.append(QueryEngineTool(
                query_engine = engine,
                metadata = ToolMetadata(name = RAG.names[n], description = RAG.descriptions[n])
            ))
        # query_engine = RouterQueryEngine.from_defaults(query_engine_tools = query_engine_tools)
        query_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=query_engine_tools, use_async=True)

        return query_engine
    
    @staticmethod
    def processing_agent(query:str):

        rag_instance = RAG()
        query_engine = rag_instance.query_engine()
    
        tools = [Tool(
            name="Llama-Index",
            func=query_engine.query,
            description=f"Useful for when you want to answer questions. The input to this tool should be a complete English sentence. Works best if you redirect the entire query back into this. This is an AI Assistant, ask complete questions, articulate well.",
            return_direct=True
            )
        ]

        template1 = """
                        You are a Smart Insurance Agent Assistant. The Agent will ask you domain specific questions. The tools provided to you have smart interpretibility if you specify keywords in your query to the tool [Example a query for two wheeler insurance rules should mention two wheelers]. You have access to the following tools:

                        {tools}

                        Use the following format:

                        Question: the input question you must answer
                        Thought: you should always think about what to do
                        Action: the action to take, should be one of [{tool_names}]
                        Action Input: the input to the action, a complete English sentence
                        Observation: the result of the action
                        ... (this Thought/Action/Action Input/Observation can repeat N times)
                        Thought: I now know the final answer
                        Final Answer: the final answer to the original input question

                        Begin! Remember to be ethical and articulate when giving your final answer. Use lots of "Arg"s

                        Question: {input}
                        {agent_scratchpad}"""

        prompt = PromptTemplate(
            template=template1,
            input_variables=["input", "intermediate_steps", "agent_scratchpad", "tools", "tool_names"]
        )

        llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=prompt,
        )

        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

        return agent_executor.invoke({"input":query})

