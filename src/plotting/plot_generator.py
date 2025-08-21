import pandas as pd
import streamlit as st
from typing import List, Dict, Optional
import json
import numpy as np

import plotly.express as px

class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer, )):
            return int(o)
        elif isinstance(o, (np.floating, )):
            return float(o)
        elif isinstance(o, (np.ndarray, )):
            return o.tolist()
        return super(NpEncoder, self).default(o)

def generate_plot_suggestions(llm_agent, user_query: str, columns_info: Dict, data_sample: pd.DataFrame) -> List[Dict]:
    """
    Generate plot suggestions using LLM based on user query and data structure
    
    Args:
        llm_agent: The LLM agent instance
        user_query: User's natural language query
        columns_info: Dictionary containing column names and their data types
        data_sample: Sample of the dataframe for context
        
    Returns:
        List of plot dictionaries with format: [{"plot_type": "hist", "columns": ["col1", "col2"], "title": "Title", "config": {...}}]
    """
    
    # Prepare context for LLM
    columns_summary = []
    for col, dtype in columns_info.items():
        sample_values = data_sample[col].dropna().head(3).tolist()
        columns_summary.append({
            "name": col,
            "type": str(dtype),
            "sample_values": sample_values,
            "null_count": data_sample[col].isnull().sum(),
            "unique_count": data_sample[col].nunique()
        })
    
    prompt = f"""
Based on the user's query and data structure, suggest appropriate plots to visualize the data.

USER QUERY: "{user_query}"

DATA STRUCTURE:
{json.dumps(columns_summary, indent=2, cls=NpEncoder)}

Available plot types:
- histogram: For distribution of numerical data (can include color/hue for grouping)
- scatter: For relationship between two numerical variables (supports color, size, and hover data)
- box: For distribution and outliers of numerical data by categories (supports color grouping)
- violin: Similar to box plots but shows full distribution shape (supports color grouping)
- bar: For categorical data counts or aggregated values (supports color grouping)
- line: For time series or ordered data (supports color for multiple series)
- area: For filled line plots (supports color for stacking)
- pie: For categorical data proportions
- sunburst: For hierarchical categorical data
- treemap: For hierarchical data with size encoding
- density_heatmap: For 2D density plots
- density_contour: For contour plots of 2D data
- correlation: For correlation matrix of numerical columns

Return a JSON array of plot suggestions in this exact format:
[
    {{
        "plot_type": "histogram",
        "columns": ["column_name"],
        "title": "Distribution of Column Name",
        "config": {{
            "bins": 30,
            "hue": "category_column_for_grouping",
            "color": "specific_color_if_no_hue"
        }}
    }},
    {{
        "plot_type": "scatter",
        "columns": ["x_column", "y_column"],
        "title": "X Column vs Y Column",
        "config": {{
            "color_column": "category_column_for_color_coding",
            "size_column": "numerical_column_for_size",
            "hue": "another_way_to_specify_color_column"
        }}
    }},
    {{
        "plot_type": "box",
        "columns": ["categorical_x", "numerical_y"],
        "title": "Y Column by X Column",
        "config": {{
            "color_column": "additional_grouping_variable"
        }}
    }}
]

Guidelines:
1. Choose plot types that make sense for the data types
2. For numerical columns, suggest histograms, box plots, violin plots, or scatter plots
3. For categorical columns, suggest bar charts, pie charts, sunburst, or treemap
4. If user asks about relationships, suggest scatter plots or correlation matrices
5. If user asks about distributions, suggest histograms, box plots, or violin plots
6. If user asks about groups/categories, use hue/color_column for grouping
7. For complex hierarchical data, suggest sunburst or treemap
8. Maximum 4 plots to avoid overwhelming the user
9. Include meaningful titles and appropriate configurations
10. Only use columns that exist in the data
11. When suggesting color/hue, ensure the column has reasonable number of unique values (< 20)
12. Prefer color coding when comparing groups or categories

RESPOND ONLY WITH THE JSON ARRAY, NO OTHER TEXT:
"""
    
    try:
        response = llm_agent.query(prompt)
        # Extract JSON from response
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:-3]
        elif response.startswith('```'):
            response = response[3:-3]
        
        plot_suggestions = json.loads(response)
        
        # Filter out inappropriate plot types based on data types
        filtered_suggestions = filter_plot_suggestions(plot_suggestions, columns_info, data_sample)
        return filtered_suggestions
    except Exception as e:
        st.error(f"Error generating plot suggestions: {e}")
        # Return default suggestions based on data types
        return generate_default_plots(columns_info, data_sample)


def filter_plot_suggestions(plot_suggestions: List[Dict], columns_info: Dict, data_sample: pd.DataFrame) -> List[Dict]:
    """
    Filter out inappropriate plot suggestions based on data types and column compatibility.
    
    Args:
        plot_suggestions: List of plot configurations from LLM
        columns_info: Dictionary containing column names and their data types
        data_sample: Sample of the dataframe for context
        
    Returns:
        Filtered list of plot configurations
    """
    filtered_plots = []
    
    for plot_config in plot_suggestions:
        plot_type = plot_config.get("plot_type", "")
        columns = plot_config.get("columns", [])
        
        # Skip plots with missing columns
        if not all(col in data_sample.columns for col in columns):
            continue
            
        # Check data type compatibility
        if plot_type == "correlation":
            # Correlation plots need at least 2 numerical columns
            numerical_cols = [col for col in columns if pd.api.types.is_numeric_dtype(data_sample[col])]
            if len(numerical_cols) < 2:
                continue  # Skip correlation plots with categorical data
                
        elif plot_type == "scatter":
            # Scatter plots need 2 numerical columns
            if len(columns) >= 2:
                x_col, y_col = columns[0], columns[1]
                if not (pd.api.types.is_numeric_dtype(data_sample[x_col]) and 
                       pd.api.types.is_numeric_dtype(data_sample[y_col])):
                    # Convert scatter to box plot if one column is categorical
                    if (not pd.api.types.is_numeric_dtype(data_sample[x_col]) and 
                        pd.api.types.is_numeric_dtype(data_sample[y_col])):
                        plot_config["plot_type"] = "box"
                        plot_config["title"] = plot_config["title"].replace("vs", "by")
                    else:
                        continue  # Skip if both are categorical or other issues
                        
        elif plot_type in ["histogram", "box", "violin"]:
            # These need at least one numerical column
            has_numerical = any(pd.api.types.is_numeric_dtype(data_sample[col]) for col in columns)
            if not has_numerical:
                continue
                
        elif plot_type in ["bar", "pie"]:
            # These work with categorical data, so they're usually fine
            pass
            
        filtered_plots.append(plot_config)
    
    return filtered_plots


def generate_default_plots(columns_info: Dict, data_sample: pd.DataFrame) -> List[Dict]:
    """Generate default plot suggestions based on data types"""
    plots = []
    numerical_cols = [col for col, dtype in columns_info.items() if pd.api.types.is_numeric_dtype(dtype)]
    categorical_cols = [col for col, dtype in columns_info.items() if not pd.api.types.is_numeric_dtype(dtype)]
    
    # Find good categorical columns for hue (those with reasonable number of unique values)
    good_hue_cols = [col for col in categorical_cols 
                     if data_sample[col].nunique() <= 10 and data_sample[col].nunique() > 1]
    
    # Add histogram for first numerical column with hue if available
    if numerical_cols:
        config = {"bins": 30}
        if good_hue_cols:
            config["hue"] = good_hue_cols[0]
        plots.append({
            "plot_type": "histogram",
            "columns": [numerical_cols[0]],
            "title": f"Distribution of {numerical_cols[0]}" + (f" by {good_hue_cols[0]}" if good_hue_cols else ""),
            "config": config
        })
    
    # Add scatter plot if we have 2+ numerical columns with color coding
    if len(numerical_cols) >= 2:
        config = {}
        if good_hue_cols:
            config["color_column"] = good_hue_cols[0]
        if len(numerical_cols) >= 3:
            config["size_column"] = numerical_cols[2]
        plots.append({
            "plot_type": "scatter",
            "columns": numerical_cols[:2],
            "title": f"{numerical_cols[0]} vs {numerical_cols[1]}" + (f" by {good_hue_cols[0]}" if good_hue_cols else ""),
            "config": config
        })
    
    # Add box plot if we have both numerical and categorical columns
    if numerical_cols and categorical_cols:
        config = {}
        if len(good_hue_cols) >= 2:
            config["color_column"] = good_hue_cols[1]
        plots.append({
            "plot_type": "box",
            "columns": [categorical_cols[0], numerical_cols[0]],
            "title": f"{numerical_cols[0]} by {categorical_cols[0]}" + (f" and {good_hue_cols[1]}" if len(good_hue_cols) >= 2 else ""),
            "config": config
        })
    
    # Add bar chart for first categorical column with grouping if available
    if categorical_cols:
        config = {}
        if len(good_hue_cols) >= 2:
            config["color_column"] = good_hue_cols[1]
        plots.append({
            "plot_type": "bar",
            "columns": [categorical_cols[0]],
            "title": f"Count of {categorical_cols[0]}" + (f" by {good_hue_cols[1]}" if len(good_hue_cols) >= 2 else ""),
            "config": config
        })
    
    return plots


def create_plot(plot_config: Dict, data: pd.DataFrame) -> None:
    """
    Create and display a plot based on the configuration
    
    Args:
        plot_config: Dictionary containing plot type, columns, title, and config
        data: The dataframe to plot from
    """
    
    plot_type = plot_config.get("plot_type")
    columns = plot_config.get("columns", [])
    title = plot_config.get("title", "Plot")
    config = plot_config.get("config", {})
    
    st.write(f"**Debug:** Creating {plot_type} plot with columns: {columns}")
    st.write(f"**Debug:** Available data columns: {list(data.columns)}")
    
    try:
        if plot_type == "histogram":
            if len(columns) >= 1 and columns[0] in data.columns:
                fig = px.histogram(
                    data, 
                    x=columns[0], 
                    title=title,
                    nbins=config.get("bins", 30)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Column '{columns[0]}' not found in data for histogram")
            
        elif plot_type == "scatter":
            if len(columns) >= 2 and all(col in data.columns for col in columns[:2]):
                color_col = config.get("color_column") if config.get("color_column") in data.columns else None
                size_col = config.get("size_column") if config.get("size_column") in data.columns else None
                
                fig = px.scatter(
                    data,
                    x=columns[0],
                    y=columns[1],
                    color=color_col,
                    size=size_col,
                    title=title
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                missing_cols = [col for col in columns[:2] if col not in data.columns]
                st.error(f"Columns {missing_cols} not found in data for scatter plot")
                
        elif plot_type == "box":
            if len(columns) >= 1 and columns[0] in data.columns:
                y_col = columns[0]
                x_col = None
                
                # Check if we have a second column for grouping
                if len(columns) > 1:
                    if columns[1] in data.columns:
                        x_col = columns[1]
                    else:
                        # If the second column doesn't exist, use the first for grouping if it's categorical
                        # and look for numerical columns for y-axis
                        numerical_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
                        if numerical_cols and not pd.api.types.is_numeric_dtype(data[columns[0]]):
                            x_col = columns[0]  # Use first column as grouping variable
                            y_col = numerical_cols[0]  # Use first numerical column as y
                        elif len(numerical_cols) > 1:
                            # If we have multiple numerical columns, use them
                            y_col = numerical_cols[0]
                            x_col = columns[0] if not pd.api.types.is_numeric_dtype(data[columns[0]]) else None
                
                st.write(f"**Debug:** Box plot using x={x_col}, y={y_col}")
                fig = px.box(
                    data,
                    x=x_col,
                    y=y_col,
                    title=title,
                    color_discrete_sequence=[config.get("color", "blue")]
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Column '{columns[0]}' not found in data for box plot")
                
        elif plot_type == "bar":
            if len(columns) >= 1 and columns[0] in data.columns:
                # Check if we have aggregated data or if this is a simple count
                if len(columns) >= 2 and columns[1] in data.columns:
                    # We have x and y columns for bar chart
                    fig = px.bar(
                        data, 
                        x=columns[0], 
                        y=columns[1], 
                        title=title,
                        labels={columns[0]: config.get("x_axis_label", columns[0]), 
                               columns[1]: config.get("y_axis_label", columns[1])},
                        color_discrete_sequence=[config.get("color", "blue")]
                    )
                elif pd.api.types.is_numeric_dtype(data[columns[0]]):
                    # For numerical data, create value counts or use values directly if already aggregated
                    if len(data[columns[0]].unique()) <= 20:  # If few unique values, treat as categorical
                        value_counts = data[columns[0]].value_counts().sort_index()
                        fig = px.bar(
                            x=value_counts.index,
                            y=value_counts.values,
                            title=title,
                            labels={'x': columns[0], 'y': 'Count'},
                            color_discrete_sequence=[config.get("color", "blue")]
                        )
                    else:
                        # For continuous numerical data, use histogram instead
                        fig = px.histogram(
                            data, 
                            x=columns[0], 
                            title=title,
                            nbins=config.get("bins", 20)
                        )
                else:
                    # For categorical data, count occurrences
                    value_counts = data[columns[0]].value_counts()
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=title,
                        labels={'x': columns[0], 'y': 'Count'},
                        color_discrete_sequence=[config.get("color", "blue")]
                    )
                st.plotly_chart(fig, use_container_width=True)
                
        elif plot_type == "line":
            if len(columns) >= 1 and columns[0] in data.columns:
                y_col = columns[0]
                x_col = columns[1] if len(columns) > 1 and columns[1] in data.columns else data.index
                
                fig = px.line(
                    data,
                    x=x_col,
                    y=y_col,
                    title=title
                )
                st.plotly_chart(fig, use_container_width=True)
                
        elif plot_type == "correlation":
            # Select only numerical columns for correlation
            numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
            if len(numerical_cols) >= 2:
                corr_matrix = data[numerical_cols].corr()
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title=title or "Correlation Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 numerical columns for correlation plot")
                
        elif plot_type == "pie":
            if len(columns) >= 1 and columns[0] in data.columns:
                value_counts = data[columns[0]].value_counts()
                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=title
                )
                st.plotly_chart(fig, use_container_width=True)
                
        else:
            st.warning(f"Plot type '{plot_type}' not supported yet")
            
    except Exception as e:
        st.error(f"Error creating {plot_type} plot: {str(e)}")


def display_plots_section(data: pd.DataFrame, user_query: str, llm_agent, query_result: Optional[pd.DataFrame] = None):
    """
    Main function to display the plots section
    
    Args:
        data: The original dataframe
        user_query: User's natural language query
        llm_agent: The LLM agent instance
        query_result: Result from SQL query (if any)
    """
    
    # Use query result if available, otherwise use original data
    plot_data = query_result if query_result is not None and not query_result.empty else data
    
    if plot_data.empty:
        st.warning("No data available for plotting")
        return
    
    st.write(f"**Debug:** Using data with shape: {plot_data.shape}")
    st.write(f"**Debug:** Available columns: {list(plot_data.columns)}")
    
    # Get column information
    columns_info = {col: plot_data[col].dtype for col in plot_data.columns}
    
    # For now, let's test with hardcoded plot suggestions based on your example
    plot_suggestions = [
        {
            "plot_type": "bar",
            "columns": ["GarageCars"],
            "title": "Average SalePrice by GarageCars",
            "config": {
                "x_axis_label": "GarageCars",
                "y_axis_label": "Average SalePrice",
                "color": "green"
            }
        },
        {
            "plot_type": "box",
            "columns": ["GarageCars", "AVG(SalePrice)"],
            "title": "SalePrice Distribution by GarageCars",
            "config": {
                "color": "purple"
            }
        }
    ]
    
    # If hardcoded plots don't work with the data, generate new ones
    if not all(any(col in plot_data.columns for col in plot["columns"]) for plot in plot_suggestions):
        st.write("**Debug:** Hardcoded plots don't match data, generating new ones...")
        # Generate plot suggestions
        with st.spinner("🤖 Generating plot suggestions..."):
            plot_suggestions = generate_plot_suggestions(llm_agent, user_query, columns_info, plot_data)
    
    if not plot_suggestions:
        st.warning("No plot suggestions generated")
        return
    
    st.subheader("📊 Suggested Visualizations")
    
    # Show plot configurations for debugging
    st.write("**Debug:** Plot configurations:")
    st.json(plot_suggestions)
    
    # Create columns for plots
    num_plots = len(plot_suggestions)
    if num_plots == 1:
        create_plot(plot_suggestions[0], plot_data)
    elif num_plots == 2:
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Creating plot 1:** {plot_suggestions[0]['plot_type']}")
            create_plot(plot_suggestions[0], plot_data)
        with col2:
            st.write(f"**Creating plot 2:** {plot_suggestions[1]['plot_type']}")
            create_plot(plot_suggestions[1], plot_data)
    else:
        # For more than 2 plots, arrange in rows
        for i in range(0, num_plots, 2):
            cols = st.columns(2)
            with cols[0]:
                st.write(f"**Creating plot {i+1}:** {plot_suggestions[i]['plot_type']}")
                create_plot(plot_suggestions[i], plot_data)
            if i + 1 < num_plots:
                with cols[1]:
                    st.write(f"**Creating plot {i+2}:** {plot_suggestions[i+1]['plot_type']}")
                    create_plot(plot_suggestions[i + 1], plot_data)
    
    # Show plot configurations for reference
    with st.expander("🔧 Plot Configurations (for developers)"):
        st.json(plot_suggestions)
