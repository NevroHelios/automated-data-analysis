import json
import openai
from google import genai
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]


class LLMAgent(ABC):
    """Abstract base class for LLM agents"""
    
    @abstractmethod
    def generate_response(self, prompt: str, tools: Optional[List[Dict]] = None) -> str:
        pass
    
    @abstractmethod
    def parse_tool_calls(self, response: str) -> List[ToolCall]:
        pass


class OpenAIAgent(LLMAgent):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def generate_response(self, prompt: str, tools: Optional[List[Dict]] = None) -> str:
        try:
            if tools:
                # Format prompt with tools
                tools_str = json.dumps(tools)
                formatted_prompt = f"[AVAILABLE_TOOLS] {tools_str}[/AVAILABLE_TOOLS][INST] {prompt} [/INST]"
            else:
                formatted_prompt = prompt
                
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=0.1
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"Error with OpenAI: {str(e)}"
    
    def parse_tool_calls(self, response: str) -> List[ToolCall]:
        tool_calls = []
        try:
            # Method 1: Look for [TOOL_CALLS] pattern
            if "[TOOL_CALLS]" in response:
                start = response.find("[TOOL_CALLS]") + len("[TOOL_CALLS]")
                end = response.find("]", start) + 1
                tool_calls_str = response[start:end].strip()
                
                # Parse the JSON
                calls_data = json.loads(tool_calls_str)
                for call in calls_data:
                    tool_calls.append(ToolCall(
                        name=call["name"],
                        arguments=call["arguments"]
                    ))
            
            # Method 2: Look for direct JSON array in response
            elif "[{" in response and "}]" in response:
                import re
                # Extract JSON array
                json_match = re.search(r'\[\{.*?\}\]', response, re.DOTALL)
                if json_match:
                    calls_data = json.loads(json_match.group(0))
                    for call in calls_data:
                        tool_calls.append(ToolCall(
                            name=call["name"],
                            arguments=call["arguments"]
                        ))
                        
        except Exception as e:
            print(f"Error parsing tool calls: {e}")
        
        return tool_calls


class GeminiAgent(LLMAgent):
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        if genai is None:
            raise ImportError("google-generativeai package not installed")
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY", "")
        self.model = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate_response(self, prompt: str, tools: Optional[List[Dict]] = None) -> str:
        try:
            if tools:
                # Format prompt with tools
                tools_str = json.dumps(tools)
                formatted_prompt = f"[AVAILABLE_TOOLS] {tools_str}[/AVAILABLE_TOOLS][INST] {prompt} [/INST]"
            else:
                formatted_prompt = prompt

            response = self.model.models.generate_content(model=self.model_name, contents=formatted_prompt)
            return response.text or ""
        except Exception as e:
            return f"Error with Gemini: {str(e)}"
    
    def parse_gemini_tool_calls(self, response: str) -> List[ToolCall]:
        """Parse Gemini-specific tool calls that use tool_code format"""
        tool_calls = []
        try:
            # Look for ```tool_code blocks
            if "```tool_code" in response:
                import re
                # Extract content between ```tool_code and ```
                pattern = r'```tool_code\s*\n(.*?)\n```'
                matches = re.findall(pattern, response, re.DOTALL)
                
                for match in matches:
                    try:
                        # Parse the JSON content
                        tool_data = json.loads(match.strip())
                        
                        # Extract function information
                        if tool_data.get("type") == "function" and "function" in tool_data:
                            function_info = tool_data["function"]
                            
                            # For Gemini, we need to extract the actual query from the description or parameters
                            # This is a simplified approach - you might need to adjust based on actual Gemini responses
                            if "parameters" in function_info and "properties" in function_info["parameters"]:
                                properties = function_info["parameters"]["properties"]
                                if "query" in properties:
                                    # This would need actual argument values, not just the schema
                                    # For now, we'll try to extract from the response text
                                    continue
                            
                    except json.JSONDecodeError:
                        continue
            
            # Also look for actual function calls with arguments
            if "execute_sql_query" in response:
                import re
                # Look for function call patterns with actual arguments
                patterns = [
                    r'execute_sql_query\s*\(\s*["\']([^"\']+)["\']',
                    r'"query":\s*"([^"]+)"',
                    r"'query':\s*'([^']+)'"
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, response)
                    for match in matches:
                        tool_calls.append(ToolCall(
                            name="execute_sql_query",
                            arguments={"query": match}
                        ))
                        break
                    if tool_calls:
                        break
                        
        except Exception as e:
            print(f"Error parsing Gemini tool calls: {e}")
        
        return tool_calls

    def parse_tool_calls(self, response: str) -> List[ToolCall]:
        """Parse tool calls - uses Gemini-specific parser first, then falls back to generic"""
        # Try Gemini-specific parser first
        tool_calls = self.parse_gemini_tool_calls(response)
        
        if tool_calls:
            return tool_calls
        
        # Fall back to generic parser
        try:
            # Method 1: Look for [TOOL_CALLS] pattern
            if "[TOOL_CALLS]" in response:
                start = response.find("[TOOL_CALLS]") + len("[TOOL_CALLS]")
                end = response.find("]", start) + 1
                tool_calls_str = response[start:end].strip()
                
                # Parse the JSON
                calls_data = json.loads(tool_calls_str)
                for call in calls_data:
                    tool_calls.append(ToolCall(
                        name=call["name"],
                        arguments=call["arguments"]
                    ))
            
            # Method 2: Look for direct JSON array in response
            elif "[{" in response and "}]" in response:
                import re
                # Extract JSON array
                json_match = re.search(r'\[\{.*?\}\]', response, re.DOTALL)
                if json_match:
                    calls_data = json.loads(json_match.group(0))
                    for call in calls_data:
                        tool_calls.append(ToolCall(
                            name=call["name"],
                            arguments=call["arguments"]
                        ))
                        
        except Exception as e:
            print(f"Error parsing tool calls (Gemini fallback): {e}")
        
        return tool_calls


class OllamaAgent(LLMAgent):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "mistral:7b"):
        self.base_url = base_url
        self.model = model
    
    def generate_response(self, prompt: str, tools: Optional[List[Dict]] = None) -> str:
        try:
            if tools:
                # Format prompt with tools
                tools_str = json.dumps(tools)
                formatted_prompt = f"[AVAILABLE_TOOLS] {tools_str}[/AVAILABLE_TOOLS][INST] {prompt} [/INST]"
            else:
                formatted_prompt = prompt
            
            payload = {
                "model": self.model,
                "prompt": formatted_prompt,
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                return f"Error with Ollama: HTTP {response.status_code}"
        except Exception as e:
            return f"Error with Ollama: {str(e)}"
    
    def parse_tool_calls(self, response: str) -> List[ToolCall]:
        tool_calls = []
        try:
            # Method 1: Look for [TOOL_CALLS] pattern
            if "[TOOL_CALLS]" in response:
                start = response.find("[TOOL_CALLS]") + len("[TOOL_CALLS]")
                end = response.find("]", start) + 1
                tool_calls_str = response[start:end].strip()
                
                # Parse the JSON
                calls_data = json.loads(tool_calls_str)
                for call in calls_data:
                    tool_calls.append(ToolCall(
                        name=call["name"],
                        arguments=call["arguments"]
                    ))
            
            # Method 2: Look for direct JSON array in response
            elif "[{" in response and "}]" in response:
                import re
                # Extract JSON array
                json_match = re.search(r'\[\{.*?\}\]', response, re.DOTALL)
                if json_match:
                    calls_data = json.loads(json_match.group(0))
                    for call in calls_data:
                        tool_calls.append(ToolCall(
                            name=call["name"],
                            arguments=call["arguments"]
                        ))
                        
        except Exception as e:
            print(f"Error parsing tool calls (Ollama): {e}")
        
        return tool_calls


class LLMFactory:
    """Factory class to create LLM agents"""
    
    @staticmethod
    def create_agent(agent_type: str, **kwargs) -> LLMAgent:
        if agent_type.lower() == "openai":
            api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
            model = kwargs.get("model", "gpt-3.5-turbo")
            if not api_key:
                raise ValueError("OpenAI API key is required")
            return OpenAIAgent(api_key=api_key, model=model)
        
        elif agent_type.lower() == "gemini":
            api_key = kwargs.get("api_key") or os.getenv("GEMINI_API_KEY")
            model = kwargs.get("model", "gemini-2.0-flash")
            if not api_key:
                raise ValueError("Gemini API key is required")
            return GeminiAgent(api_key=api_key, model_name=model)
        
        elif agent_type.lower() == "ollama":
            base_url = kwargs.get("base_url", "http://localhost:11434")
            model = kwargs.get("model", "mistral:7b")
            return OllamaAgent(base_url=base_url, model=model)
        
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")


# SQL Query Tool Definition
SQL_QUERY_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_sql_query",
        "description": "Execute a SQL query on the uploaded dataset",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The SQL query to execute on the dataset"
                }
            },
            "required": ["query"]
        }
    }
}


def get_sql_query_from_llm(agent: LLMAgent, user_question: str, table_info: str) -> str:
    """Get SQL query from LLM based on user question and table info"""
    
    prompt = f"""
You are a SQL expert. Based on the table information below, generate ONLY a SQL query to answer the user's question.

IMPORTANT RULES:
1. The table name is ALWAYS "data_table" 
2. To get column names, use: PRAGMA table_info(data_table)
3. To get column names in a result format, use: SELECT name FROM PRAGMA_TABLE_INFO('data_table')
4. For data queries, use: SELECT ... FROM data_table
5. Do NOT query sqlite_master - use the data_table directly
6. Always use proper SQLite syntax
7. You query only one time so combine your queries if necessary.

{table_info}

User Question: {user_question}

Examples:
- "What are the columns?" → PRAGMA table_info(data_table) OR SELECT name FROM PRAGMA_TABLE_INFO('data_table')
- "Show me data" → SELECT * FROM data_table LIMIT 10
- "Count rows" → SELECT COUNT(*) FROM data_table

You must respond with a tool call using the execute_sql_query function. Generate ONLY the tool call, nothing else.
"""
    
    response = agent.generate_response(prompt, tools=[SQL_QUERY_TOOL])
    print(f"Raw LLM response: {response}")  # Debug print
    
    tool_calls = agent.parse_tool_calls(response)
    
    if tool_calls and len(tool_calls) > 0:
        query = tool_calls[0].arguments.get("query", "")
        print(f"Extracted query from tool call: {query}")  # Debug print
        return query
    else:
        # Enhanced fallback: try multiple extraction methods
        print("No tool calls found, trying fallback extraction")
        print(response) # Debug
        
        # Method 1: Look for JSON-like tool call in response
        if "[{" in response and "}]" in response:
            try:
                import re
                # Extract JSON from response
                json_match = re.search(r'\[\{"name".*?"query":\s*"([^"]+)".*?\}\]', response)
                if json_match:
                    query = json_match.group(1)
                    print(f"Extracted query via regex: {query}")
                    return query
            except Exception as e:
                print(f"Regex extraction failed: {e}")
        
        # Method 2: Look for direct SQL in response
        sql_keywords = ["SELECT", "PRAGMA", "WITH", "INSERT", "UPDATE", "DELETE"]
        for keyword in sql_keywords:
            if keyword in response.upper():
                lines = response.split('\n')
                for line in lines:
                    if keyword in line.upper():
                        query = line.strip()
                        print(f"Found SQL line: {query}")
                        return query
        
        # Method 3: Try to extract from quotes
        import re
        sql_patterns = [
            r'"query":\s*"([^"]+)"',
            r"'query':\s*'([^']+)'",
            r'(PRAGMA.*?);?',
            r'(SELECT.*?);?',
            r'(pragma.*?);?',
            r'(select.*?);?'
        ]
        
        for pattern in sql_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                query = match.group(1) if match.groups() else match.group(0)
                query = query.strip().rstrip(';')
                print(f"Pattern {pattern} found query: {query}")
                return query
    
    print("No query could be extracted")
    return ""


if __name__ == "__main__":
    # Example usage
    try:
        # Create Ollama agent (default for this project)
        # agent = LLMFactory.create_agent("ollama", model="mistral:7b")
        agent = LLMFactory.create_agent("gemini", model="gemini-2.0-flash")

        # Test the agent
        # response = agent.generate_response("Hello, how are you?")
        # print(f"Response: {response}")
        
        # Test with tools
        tools = [SQL_QUERY_TOOL]
        query_response = agent.generate_response(
            "show me the average price", 
            tools=tools
        )
        print(f"Tool response: {query_response}")
        
    except Exception as e:
        print(f"Error: {e}")
