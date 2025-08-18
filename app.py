import streamlit as st
import pandas as pd
import os
from datetime import datetime
import traceback
import subprocess
import logging
try:
    from sql_agent import SQLAgent
    from utils import get_table_info, execute_sql_query, save_data_info, create_sqlite_db
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

def create_plot_figure(plot_config, data):
    """Create a plot figure based on configuration"""
    try:
        import plotly.express as px
        
        plot_type = plot_config.get('plot_type')
        columns = plot_config.get('columns', [])
        title = plot_config.get('title', 'Plot')
        config = plot_config.get('config', {})
        
        # Validate columns exist in data
        available_cols = [col for col in columns if col in data.columns]
        if not available_cols:
            return None
        
        # Extract color/hue information from config
        color_col = None
        size_col = None
        
        # Check for various color column specifications
        if config.get('color_column') and config.get('color_column') in data.columns:
            color_col = config.get('color_column')
        elif config.get('hue') and config.get('hue') in data.columns:
            color_col = config.get('hue')
        elif config.get('color') and config.get('color') in data.columns:
            color_col = config.get('color')
        
        # Check for size column
        if config.get('size_column') and config.get('size_column') in data.columns:
            size_col = config.get('size_column')
        elif config.get('size') and config.get('size') in data.columns:
            size_col = config.get('size')
        
        fig = None
        
        if plot_type == 'histogram' and len(available_cols) >= 1:
            fig = px.histogram(data, x=available_cols[0], title=title, 
                             nbins=config.get('bins', 30),
                             color=color_col)
        elif plot_type == 'scatter' and len(available_cols) >= 2:
            fig = px.scatter(data, x=available_cols[0], y=available_cols[1], 
                           color=color_col, size=size_col, title=title,
                           hover_data=data.columns.tolist()[:5])  # Add hover data
        elif plot_type == 'bar' and len(available_cols) >= 1:
            if len(available_cols) == 1:
                # Count plot
                if color_col:
                    # Grouped bar chart
                    grouped_data = data.groupby([available_cols[0], color_col]).size().reset_index(name='count')
                    fig = px.bar(grouped_data, x=available_cols[0], y='count', 
                               color=color_col, title=title,
                               labels={'count': 'Count'})
                else:
                    # Simple count plot
                    value_counts = data[available_cols[0]].value_counts().head(20)
                    fig = px.bar(x=value_counts.index, y=value_counts.values, 
                               title=title, labels={'x': available_cols[0], 'y': 'Count'})
            else:
                # Bar plot with x and y
                fig = px.bar(data, x=available_cols[0], y=available_cols[1], 
                           color=color_col, title=title)
        elif plot_type == 'box' and len(available_cols) >= 1:
            if len(available_cols) == 1:
                fig = px.box(data, y=available_cols[0], color=color_col, title=title)
            else:
                fig = px.box(data, x=available_cols[0], y=available_cols[1], 
                           color=color_col, title=title)
        elif plot_type == 'violin' and len(available_cols) >= 1:
            if len(available_cols) == 1:
                fig = px.violin(data, y=available_cols[0], color=color_col, title=title)
            else:
                fig = px.violin(data, x=available_cols[0], y=available_cols[1], 
                              color=color_col, title=title)
        elif plot_type == 'line' and len(available_cols) >= 2:
            fig = px.line(data, x=available_cols[0], y=available_cols[1], 
                        color=color_col, title=title)
        elif plot_type == 'area' and len(available_cols) >= 2:
            fig = px.area(data, x=available_cols[0], y=available_cols[1], 
                        color=color_col, title=title)
        elif plot_type == 'pie' and len(available_cols) >= 1:
            if color_col:
                # Use color column for grouping
                grouped_data = data.groupby(available_cols[0])[color_col].count().head(10)
                fig = px.pie(values=grouped_data.values, names=grouped_data.index, title=title)
            else:
                value_counts = data[available_cols[0]].value_counts().head(10)
                fig = px.pie(values=value_counts.values, names=value_counts.index, title=title)
        elif plot_type == 'sunburst' and len(available_cols) >= 2:
            # Hierarchical pie chart
            fig = px.sunburst(data, path=available_cols[:3], title=title)
        elif plot_type == 'treemap' and len(available_cols) >= 1:
            # Tree map visualization
            if len(available_cols) == 1:
                value_counts = data[available_cols[0]].value_counts().head(15)
                fig = px.treemap(names=value_counts.index, values=value_counts.values, title=title)
            else:
                fig = px.treemap(data, path=available_cols[:3], title=title)
        elif plot_type == 'density_heatmap' and len(available_cols) >= 2:
            fig = px.density_heatmap(data, x=available_cols[0], y=available_cols[1], title=title)
        elif plot_type == 'density_contour' and len(available_cols) >= 2:
            fig = px.density_contour(data, x=available_cols[0], y=available_cols[1], 
                                   color=color_col, title=title)
        
        if fig:
            # Update layout for better appearance
            fig.update_layout(
                height=400, 
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=True if color_col else False
            )
            
            # If we have a color column, make sure the legend is visible
            if color_col:
                fig.update_layout(
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.01
                    )
                )
            
            return fig
        
    except Exception as e:
        logger.error(f"Error creating plot: {e}")
        return None
    
    return None

def build_metrics_text(df: pd.DataFrame) -> str:
    """Create a compact textual summary of a dataframe suitable for LLM input."""
    try:
        rows, cols = df.shape
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()

        parts = [
            f"Shape: {rows} rows x {cols} cols",
            f"Numeric columns: {len(numeric_cols)}",
            f"Categorical columns: {len(categorical_cols)}"
        ]

        # Basic stats for up to 3 numeric columns
        for col in numeric_cols[:3]:
            s = df[col].dropna()
            if not s.empty:
                parts.append(
                    f"{col}: mean={s.mean():.3f}, min={s.min():.3f}, max={s.max():.3f}"
                )

        # Top categories for up to 2 categorical cols
        for col in categorical_cols[:2]:
            vc = df[col].astype(str).value_counts().head(3)
            top = ", ".join([f"{k}({v})" for k, v in vc.items()])
            parts.append(f"Top {col}: {top}")

        return " | ".join(parts)
    except Exception as e:
        logger.debug(f"metrics build failed: {e}")
        return f"Shape: {df.shape}"

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@st.dialog("📋 Dataset Preview")
def show_data_preview(data, filename):
    st.subheader(f"Data Preview: {filename}")
    st.write(f"**Shape:** {data.shape[0]} rows × {data.shape[1]} columns")
    
    tab1, tab2 = st.tabs(["Preview", "Summary"])
    
    with tab1:
        st.dataframe(data.head(20), use_container_width=True)
    
    with tab2:
        st.write("**Data Types:**")
        dtypes_df = pd.DataFrame({
            'Column': data.dtypes.index,
            'Type': data.dtypes.values,
            'Non-Null Count': data.count().values,
            'Null Count': data.isnull().sum().values
        })
        st.dataframe(dtypes_df, use_container_width=True)
        
        st.write("**Summary Statistics:**")
        st.dataframe(data.describe(), use_container_width=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'plots_history' not in st.session_state:
        st.session_state.plots_history = []
    if 'show_query_panel' not in st.session_state:
        st.session_state.show_query_panel = True
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    if 'smart_summary_enabled' not in st.session_state:
        st.session_state.smart_summary_enabled = True
    if 'summaries' not in st.session_state:
        st.session_state.summaries = []  # list of {timestamp, question, summary, suggestions}
    if 'analysis_context' not in st.session_state:
        st.session_state.analysis_context = []  # recent compact metrics texts

def add_to_chat_history(question, sql_query, result, error=None, timestamp=None):
    """Add a query and its result to chat history"""
    if timestamp is None:
        timestamp = datetime.now()
    
    chat_entry = {
        'timestamp': timestamp,
        'question': question,
        'sql_query': sql_query,
        'result': result,
        'error': error,
        'result_shape': result.shape if result is not None and not result.empty else None
    }
    st.session_state.chat_history.append(chat_entry)

def add_plot_to_history(plot_data, question, timestamp=None):
    """Add a plot to the plots history"""
    if timestamp is None:
        timestamp = datetime.now()
    
    plot_entry = {
        'timestamp': timestamp,
        'question': question,
        'plot_data': plot_data
    }
    # Add to beginning for latest-first display
    st.session_state.plots_history.insert(0, plot_entry)

def render_sidebar():
    """Render the sidebar with configuration and file upload"""
    st.sidebar.header("🤖 LLM Configuration")
    
    # Debug mode toggle
    st.session_state.debug_mode = st.sidebar.toggle(
        "Debug Mode",
        value=st.session_state.debug_mode,
        help="Show debug information and logs"
    )
    
    agent_type = st.sidebar.selectbox(
        "Choose LLM Agent:",
        ["ollama", "openai", "gemini"],
        index=0
    )
    
    # Configuration based on agent type
    base_url = None  # Initialize base_url
    if agent_type == "openai":
        api_key = st.sidebar.text_input("OpenAI API Key:", type="password")
        model = st.sidebar.selectbox("Model:", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"])
    elif agent_type == "gemini":
        api_key = st.sidebar.text_input("Gemini API Key:", type="password")
        model = st.sidebar.selectbox("Model:", ["gemma-3-27b-it", "gemma-3-12b-it", "gemini-2.0-flash"])
    else:  # ollama
        base_url = st.sidebar.text_input("Ollama Base URL:", value="http://localhost:11434")
        try:
            models = subprocess.run(['ollama', 'ls'], capture_output=True, text=True).stdout.strip().split('\n')
            models = [model.split()[0] for model in models if not model.split()[0].isalpha()]
            if not models:
                models = []  # fallback
        except Exception:
            models = []  # fallback
        model = st.sidebar.selectbox("Model:", models)
        api_key = None
    
    # Visualization settings
    st.sidebar.header("📊 Visualization Settings")
    enable_plots = st.sidebar.toggle(
        "Generate Plots",
        value=True,
        help="Enable automatic plot generation based on your queries"
    )

    # Smart summary & suggestions toggle
    st.sidebar.header("🧠 Insights")
    smart_summary_enabled = st.sidebar.toggle(
        "Auto Summary & Suggestions",
        value=st.session_state.get('smart_summary_enabled', True),
        help="Generate a short insight and suggest follow-up queries after each run"
    )
    st.session_state.smart_summary_enabled = smart_summary_enabled
    
    # File upload in sidebar
    st.sidebar.header("📁 Upload Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file to analyze"
    )
    
    # Show available files
    if os.path.exists('data') and os.listdir('data'):
        st.sidebar.header("📁 Available Files")
        files = os.listdir('data')
        for file in files:
            st.sidebar.text(f"• {file}")
    
    return agent_type, model, api_key, base_url, enable_plots, smart_summary_enabled, uploaded_file

def handle_file_upload(uploaded_file):
    """Handle file upload and show preview dialog"""
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            data = pd.read_csv(uploaded_file)
            
            # Save the file
            file_path = f"data/{uploaded_file.name}"
            data.to_csv(file_path, index=False)
            
            st.sidebar.success(f"✅ File uploaded! Shape: {data.shape}")
            
            # Show preview button
            if st.sidebar.button("📋 View Dataset Preview"):
                show_data_preview(data, uploaded_file.name)
            
            # Save data info
            dtypes_dict, summary_stats = save_data_info(data, uploaded_file.name)
            
            # Create SQLite database
            db_path = create_sqlite_db(data, uploaded_file.name)
            
            if st.session_state.debug_mode:
                st.sidebar.success(f"✅ Database created: {db_path}")
            
            # Store in session state
            st.session_state['data'] = data
            st.session_state['db_path'] = db_path
            st.session_state['table_info'] = get_table_info(db_path)
            st.session_state['current_file'] = uploaded_file.name
            
            return True
            
        except Exception as e:
            st.sidebar.error(f"❌ Error processing file: {str(e)}")
            if st.session_state.debug_mode:
                st.sidebar.error(traceback.format_exc())
            return False
    return False

def render_plots_section():
    """Render the plots section on the left"""
    if st.session_state.plots_history:
        st.header("📊 Generated Plots")
        
        for i, plot_entry in enumerate(st.session_state.plots_history):
            with st.expander(f"Plot {i+1}: {plot_entry['question'][:50]}...", expanded=(i==0)):
                st.write(f"**Question:** {plot_entry['question']}")
                st.write(f"**Generated:** {plot_entry['timestamp'].strftime('%H:%M:%S')}")
                
                # Display plots
                if 'plot_data' in plot_entry and plot_entry['plot_data']:
                    plots = plot_entry['plot_data']
                    for j, plot_info in enumerate(plots):
                        if 'figure' in plot_info and plot_info['figure']:
                            st.plotly_chart(plot_info['figure'], use_container_width=True, key=f"plot_{i}_{j}")
                        elif 'config' in plot_info:
                            st.write(f"Plot {j+1}: {plot_info['config'].get('title', 'Untitled')}")
    else:
        st.header("📊 Plots")
        st.info("No plots generated yet. Enable plot generation and ask questions about your data!")

def render_query_panel(agent_type, model, api_key, base_url, enable_plots, smart_summary_enabled):
    """Render the query panel on the right"""
    # Toggle button for hiding/showing query panel
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("🔍 Query Your Data")
    with col2:
        if st.button("👁️" if st.session_state.show_query_panel else "👁️‍🗨️"):
            st.session_state.show_query_panel = not st.session_state.show_query_panel
    
    if not st.session_state.show_query_panel:
        return
    
    if 'db_path' not in st.session_state:
        st.info("👆 Please upload a CSV file first to start querying your data.")
        return
    
    # Initialize LLM agent
    try:
        agent = SQLAgent(agent_type, model_name=model, base_url=base_url, api_key=api_key)
        
        # Summary area (above the query input)
        st.subheader("🧠 Summary & Suggestions")
        if smart_summary_enabled and st.session_state.summaries:
            last_summary = st.session_state.summaries[-1]
            st.write(last_summary.get('summary', ''))
            suggestions = last_summary.get('suggestions', [])
            if suggestions:
                st.caption("Suggested next queries:")
                for idx, s in enumerate(suggestions[:3], 1):
                    st.write(f"{idx}. {s}")
        elif smart_summary_enabled:
            st.info("Insights will appear here after you run a query.")
        else:
            st.caption("Auto Summary is off. Turn it on in the sidebar to see insights.")

        # Query input
        user_question = st.text_area(
            "Ask a question about your data:",
            placeholder="e.g., What is the average price by location? Show me the top 10 most expensive houses.",
            height=100,
            key="query_input"
        )
        
        col1, col2 = st.columns([2, 1])
        with col1:
            execute_query = st.button("🚀 Generate SQL & Execute", type="primary")
        with col2:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("💬 Query History")
            
            for i, chat_entry in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"Q{len(st.session_state.chat_history)-i}: {chat_entry['question'][:50]}...", expanded=(i==0)):
                    st.write(f"**Question:** {chat_entry['question']}")
                    st.write(f"**Time:** {chat_entry['timestamp'].strftime('%H:%M:%S')}")
                    
                    if chat_entry['error']:
                        st.error(f"❌ Error: {chat_entry['error']}")
                    else:
                        with st.expander("🔧 View SQL Query"):
                            st.code(chat_entry['sql_query'], language="sql")
                        
                        if chat_entry['result'] is not None and not chat_entry['result'].empty:
                            st.write(f"**Results:** {chat_entry['result_shape'][0]} rows × {chat_entry['result_shape'][1]} columns")
                            st.dataframe(chat_entry['result'], use_container_width=True)
                            
                            # Download button for individual query results
                            csv = chat_entry['result'].to_csv(index=False)
                            st.download_button(
                                label="💾 Download",
                                data=csv,
                                file_name=f"query_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                key=f"download_{i}"
                            )
                        else:
                            st.info("No results found for this query.")
        
    # Execute new query
        if execute_query and user_question:
            with st.spinner("🤖 Generating SQL query..."):
                try:
                    # Get SQL query from LLM (only send latest query, not all history)
                    sql_query = agent.generate_sql(
                        user_question=user_question,
                        table_info=st.session_state['table_info']
                    )

                    if st.session_state.debug_mode:
                        st.write("**Debug Info:**")
                        st.write(f"Raw query returned: `{sql_query}`")
                    
                    if sql_query:
                        # Validate SQL query before execution
                        if sql_query.strip().upper().startswith(('SELECT', 'WITH', 'PRAGMA')):
                            # Execute the query
                            result, error = execute_sql_query(st.session_state['db_path'], sql_query)
                            
                            if error:
                                add_to_chat_history(user_question, sql_query, None, error)
                                st.error(f"❌ SQL Error: {error}")
                            elif result is not None:
                                # Add to chat history
                                add_to_chat_history(user_question, sql_query, result)
                                
                                # Generate plots if enabled
                                if enable_plots and not result.empty:
                                    try:
                                        # Generate plot suggestions and create plots
                                        from plotting import generate_plot_suggestions
                                        
                                        # Get column information for plot generation
                                        columns_info = {col: result[col].dtype for col in result.columns}
                                        plot_suggestions = generate_plot_suggestions(agent, user_question, columns_info, result)
                                        
                                        # Store plot data for later display
                                        plots = []
                                        for plot_config in plot_suggestions:
                                            # Create plot and capture it
                                            plot_fig = create_plot_figure(plot_config, result)
                                            if plot_fig:
                                                plots.append({
                                                    'config': plot_config,
                                                    'figure': plot_fig
                                                })
                                        
                                        if plots:
                                            add_plot_to_history(plots, user_question)
                                    except Exception as plot_error:
                                        if st.session_state.debug_mode:
                                            st.error(f"Plot generation error: {plot_error}")
                                            st.error(traceback.format_exc())

                                # Smart summary and suggestions
                                if smart_summary_enabled:
                                    try:
                                        # Build compact metrics/context from result
                                        metrics_text = build_metrics_text(result)
                                        st.session_state.analysis_context.append(metrics_text)
                                        context_tail = "\n\n".join(st.session_state.analysis_context[-5:])  # last 5
                                        
                                        summary_prompt = (
                                            "You are a data analyst. Given the latest query results summary and prior context, "
                                            "write a concise 1-2 sentence insight. Then suggest up to 3 follow-up analysis questions.\n\n"
                                            f"[PRIOR CONTEXT]\n{context_tail}\n\n"
                                            f"[LATEST QUESTION]\n{user_question}\n\n"
                                            f"[LATEST RESULT SUMMARY]\n{metrics_text}\n\n"
                                            "Respond in JSON with keys: summary (string), suggestions (array of strings)."
                                        )
                                        llm_resp = agent.query(summary_prompt).strip()
                                        if llm_resp.startswith('```json'):
                                            llm_resp = llm_resp[7:-3]
                                        elif llm_resp.startswith('```'):
                                            llm_resp = llm_resp[3:-3]
                                        import json as _json
                                        payload = {}
                                        try:
                                            payload = _json.loads(llm_resp)
                                        except Exception:
                                            # fallback: wrap raw text
                                            payload = {"summary": llm_resp[:300], "suggestions": []}
                                        st.session_state.summaries.append({
                                            'timestamp': datetime.now(),
                                            'question': user_question,
                                            'summary': payload.get('summary', ''),
                                            'suggestions': payload.get('suggestions', [])
                                        })
                                    except Exception as se:
                                        if st.session_state.debug_mode:
                                            st.warning(f"Summary generation failed: {se}")
                                
                                st.rerun()  # Refresh to show new chat entry
                            else:
                                add_to_chat_history(user_question, sql_query, None, "Unknown error occurred while executing query")
                                st.error("❌ Unknown error occurred while executing query.")
                        else:
                            error_msg = "Generated query is not a valid SELECT/PRAGMA statement"
                            add_to_chat_history(user_question, sql_query, None, error_msg)
                            st.error(f"❌ {error_msg}. Please try rephrasing your question.")
                            if st.session_state.debug_mode:
                                st.write(f"Generated query: `{sql_query}`")
                    else:
                        error_msg = "Could not generate SQL query"
                        add_to_chat_history(user_question, None, None, error_msg)
                        st.error("❌ Could not generate SQL query. Please try rephrasing your question.")
                        
                        # Show some example queries
                        st.info("💡 Try questions like:")
                        st.write("- What are the column names?")
                        st.write("- Show me the first 10 rows")
                        st.write("- What is the average of [column_name]?")
                        st.write("- How many records are there?")
                        st.write("- What are the unique values in [column_name]?")
                        
                except Exception as e:
                    error_msg = str(e)
                    add_to_chat_history(user_question, None, None, error_msg)
                    st.error(f"❌ Error: {error_msg}")
                    if st.session_state.debug_mode:
                        st.error(traceback.format_exc())
        
        elif execute_query and not user_question:
            st.warning("⚠️ Please enter a question about your data.")
    
    except Exception as e:
        st.error(f"❌ Error initializing LLM agent: {str(e)}")
        if st.session_state.debug_mode:
            st.error(traceback.format_exc())

def main():
    st.set_page_config(
        page_title="Auto Data Analysis with LLM",
        page_icon="📊",
        layout="wide"
    )
    os.makedirs("data", exist_ok=True)
    
    # Initialize session state
    initialize_session_state()

    st.title("📊 Auto Data Analysis with LLM")
    
    # Render sidebar
    agent_type, model, api_key, base_url, enable_plots, smart_summary_enabled, uploaded_file = render_sidebar()
    
    # Handle file upload
    handle_file_upload(uploaded_file)
    
    # Main layout: plots on left, queries on right
    col1, col2 = st.columns([1, 1])
    
    with col1:
        render_plots_section()
    
    with col2:
        render_query_panel(agent_type, model, api_key, base_url, enable_plots, smart_summary_enabled)


if __name__ == "__main__":
    main()
