# Auto Data Analysis with LLM

This project provides an automated data analysis system using multiple LLM agents (OpenAI, Gemini, and Ollama) with a Streamlit web interface.

## Features

- **Multiple LLM Agents**: Factory pattern implementation supporting OpenAI GPT, Google Gemini, and Ollama (using Mistral 7B)
- **Data Upload & Analysis**: Upload CSV files and get automatic data type analysis and summary statistics
- **Natural Language Queries**: Ask questions about your data in natural language
- **SQL Generation**: LLM generates SQL queries based on your questions
- **Results Display**: Execute queries and display results in an interactive interface

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Ollama (Recommended)

Install Ollama and pull the Mistral model:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Mistral 7B model
ollama pull mistral:7b

# Start Ollama server
ollama serve
```

### 3. Optional: Setup API Keys

For OpenAI:
- Get API key from https://platform.openai.com/
- Enter in the sidebar when using the app

For Gemini:
- Get API key from https://makersuite.google.com/
- Enter in the sidebar when using the app

## Usage

### 1. Start the Streamlit App

```bash
streamlit run app.py
```

### 2. Upload Your Data

- Click "Choose a CSV file" and upload your dataset
- The system will automatically:
  - Extract data types and save to `data/{filename}_dtypes.json`
  - Generate summary statistics and save to `data/{filename}_summary.txt`
  - Create a SQLite database for querying

### 3. Query Your Data

- Select your preferred LLM agent in the sidebar
- Enter API keys if using OpenAI or Gemini
- Type natural language questions about your data
- The system will:
  - Generate appropriate SQL queries using tool calls
  - Execute the queries on your database
  - Display results in an interactive table

## Tool Call Format

The system uses a specific tool call format for SQL generation:

**Prompt Template:**
```
[AVAILABLE_TOOLS] [{"type": "function", "function": {"name": "execute_sql_query", "description": "Execute a SQL query on the uploaded dataset", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The SQL query to execute on the dataset"}}, "required": ["query"]}}}][/AVAILABLE_TOOLS][INST] {user_question} [/INST]
```

**Expected Response:**
```
[TOOL_CALLS] [{"name": "execute_sql_query", "arguments": {"query": "SELECT * FROM data_table LIMIT 10"}}]
```

## File Structure

```
auto-analysis/
├── app.py              # Streamlit web application
├── llm.py              # LLM agent factory and implementations
├── requirements.txt    # Python dependencies
├── data/              # Data storage directory
│   ├── {filename}.csv         # Uploaded CSV files
│   ├── {filename}.db          # SQLite databases
│   ├── {filename}_dtypes.json # Data type information
│   └── {filename}_summary.txt # Summary statistics
├── main.py            # Original analysis script
├── Housing.csv        # Sample dataset
└── housing.db         # Sample database
```

## Example Queries

- "What is the average price by location?"
- "Show me the top 10 most expensive houses"
- "How many houses have more than 3 bedrooms?"
- "What is the correlation between size and price?"
- "Show me houses built after 2000"

## LLM Agent Details

### OpenAI Agent
- Models: GPT-3.5-turbo, GPT-4, GPT-4-turbo
- Requires: OpenAI API key

### Gemini Agent  
- Models: gemini-pro, gemini-pro-vision
- Requires: Google AI API key

### Ollama Agent (Recommended)
- Models: mistral:7b, llama2:7b, codellama:7b
- Requires: Local Ollama installation
- No API key needed

## Troubleshooting

1. **Ollama Connection Issues**
   - Ensure Ollama is running: `ollama serve`
   - Check if the model is available: `ollama list`
   - Verify the base URL in the sidebar

2. **API Key Issues**
   - Ensure valid API keys are entered
   - Check rate limits and quotas

3. **Import Errors**
   - Run: `pip install -r requirements.txt`
   - Ensure Python environment is activated

## Data Analysis Outputs

The system automatically generates:

1. **Data Types JSON** (`{filename}_dtypes.json`):
   ```json
   {
     "column1": "int64",
     "column2": "object",
     "column3": "float64"
   }
   ```

2. **Summary Statistics** (`{filename}_summary.txt`):
   - Data types for each column
   - Statistical summary (count, mean, std, min, max, quartiles)
   - Generated timestamp

3. **SQLite Database** (`{filename}.db`):
   - Table name: `data_table`
   - Contains all uploaded data
   - Used for SQL query execution
