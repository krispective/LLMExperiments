from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_community.llms import Ollama
import json

model_name = 'llama3.2:3b'
llm=Ollama(model=model_name)

class UserQuery(BaseModel):
    question: str = Field(..., title="User Question", description="The question the user asks the agent.")

class Context(BaseModel):
    retrieved_context: list[str] = Field(..., title="Retrieved Context", description="Any context retieved from custom knowledge base to help agent make informed decision.")

class Tool(BaseModel):
    name: str = Field(..., title="Tool Name", description="The name of the tool or agent.")
    description: str = Field(..., title="Tool Description", description="A brief description of what the tool does.")

class SelectedTools(BaseModel):
    tools: list[str] = Field(..., title="Selected Tools", description="List of selected tools needed to answer the user query.")

class PlannerAgent:
    def __init__(self, model_name: str = "llama3.2:3b"):
        self.llm = Ollama(model=model_name)
        self.system_prompt = ("""You are an AI agent responsible for selecting the most appropriate tools based on given descriptions. Analyze the descriptions carefully and determine which tools best match the requirements. Provide your response in strightly in the following format only.
        <JSON_output_format>
        {{"tools"}}: ["tool_name1", "tool_name2"],
        {{"reasoning"}}: "Explain why the selected tools are the best fit for the given descriptions. How do you use the given context obtained from each selected tool to complete task from user? Ensure that your reasoning is clear, concise, and directly related to the descriptions provided.'
        </JSON_output_format>
        Here should be the end of your output. Do not include additonal information.                     
        """)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "User Query: {question}. Available Tools: {tools}. Select the appropriate tools.")
        ])

    def parse_output(self, response):
        json_response = json.loads(f"""{response}""")
        
        return json_response["tools"], json_response["reasoning"]

    
    def select_tools(self, user_query: UserQuery, available_tools: list[Tool]) -> SelectedTools:
        try:
            tool_descriptions = {tool.name: tool.description for tool in available_tools}
            chain = self.prompt | self.llm
            response = chain.invoke({"question": user_query.question, "tools": tool_descriptions})
            return self.parse_output(response)
        except Exception as e:
            return SelectedTools(tools=[f"Error: {str(e)}"])

class FinalResponseAgent:
    def __init__(self, model_name: str = "llama3.2:3b"):
        self.llm = Ollama(model=model_name)
        self.system_prompt = ("""
        You are an customer service AI assistant specialised in problem solving and logical reasoning and an expert at domain straighttalk.com. Your task is to carefully the information enclosed in the context and provide response that solves the user's query.
        Only utilize the information provided in the context and do not fabricate any new filler information.                      
        """)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "Question: {question} and for context you have this <context>{context}</context>")
        ])

    def finalResponse(self, user_query: UserQuery, all_context: list[str]) -> str:
        try:
            chain = self.prompt | self.llm
            context = "####".join(all_context)
            response = chain.invoke({"question": user_query.question, "context": context})
            return response
        except Exception as e:
            return f"Error processing request: {str(e)}"

        
class ContextResponseAgent:
    def __init__(self, model_name: str = "llama3.2:3b"):
        self.llm = Ollama(model=model_name)
        self.system_prompt = ("""
        You are an data selection expert at domain straighttalk.com. Your task is to carefully select and respond with necessary information with respect to user query. Strictly retrieve information not more than 50 tokens.
        Do not create or fabricate any new filler information. Summarize your response in normal text format and avoid duplicate information. Only respond with the information provided in the context.                      
        """)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "Question: {question} and for context you have this {context}")
        ])

    def toolSelector(self, user_query: UserQuery) -> str:
        try:
            available_tools = [
                    Tool(name="OneMonthPlanChecker", description="Checks details of all one-month wireless plans and its included features. If the user has mentioned a plan name its best to retrieve that details for more context."),
                    Tool(name="InternationalPlanChecker", description="Checks details of international plans. Plans that have international capabilities areonly included in this tool.")
                ]
                
            planner = PlannerAgent()
            print("\nPlanner agent trying to select tool")
            tools, reasoning = planner.select_tools(user_query, available_tools)
            
            print("\nTools Selected : ", tools)
            print("\nReasoning : ", reasoning)

            return tools, reasoning
        except Exception as e:
            return f"Error processing request: {str(e)}"

    def OneMonthPlanChecker(self, user_query: UserQuery) -> str:
        try:
            with open("wireless_mobile_phone_plans.txt", "r") as file:
                context = file.read()
            
            chain = self.prompt | self.llm
            response = chain.invoke({"question": user_query.question, "context": context})
            return response
        except Exception as e:
            return f"Error processing request: {str(e)}"
        
    def InternationalPlanChecker(self, user_query: UserQuery) -> str:
        try:
            with open("international_wireless_plans.txt", "r") as file:
                context = file.read()
            
            chain = self.prompt | self.llm
            response = chain.invoke({"question": user_query.question, "context": context})
            return response
        except Exception as e:
            return f"Error processing request: {str(e)}"

def planner_agent(user_question):
    user_input = UserQuery(question=user_question)
    agent = ContextResponseAgent()
    tools, reasoning = agent.toolSelector(user_input)

    context = []
    for tool in tools:
        exec(f"context.append(agent.{tool}(user_input))")

    for i, cur_context in enumerate(context):
        print(f"\nRetrieved context {i}: ", cur_context)

    context_input = Context(retrieved_context=context)
    final_response_agent = FinalResponseAgent()
    final_response = final_response_agent.finalResponse(user_input, context_input.retrieved_context)
    print(final_response)
    return final_response