from src.agents.llm import LLMFactory, get_sql_query_from_llm
from typing import Optional


class SQLAgent:
    def __init__(self, agent_provider, model_name: Optional[str] = None, base_url: Optional[str] = None, api_key: Optional[str] = None):
        if agent_provider == "ollama":
            self.llm_agent = LLMFactory.create_agent(agent_provider, base_url=base_url, model=model_name)
        else:
            self.llm_agent = LLMFactory.create_agent(agent_provider, api_key=api_key, model=model_name)

    def generate_sql(self, user_question, table_info):
        return get_sql_query_from_llm(self.llm_agent, user_question, table_info)
    
    def query(self, prompt):
        return self.llm_agent.generate_response(prompt)