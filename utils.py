import pandas as pd
import sqlite3
import os
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




def save_data_info(data: pd.DataFrame, filename: str):
    """Save data types and summary statistics to files"""
    base_name = os.path.splitext(filename)[0]
    os.makedirs(f"data/{base_name}", exist_ok=True)
    # Save data types
    dtypes_dict = data.dtypes.to_dict()
    # Convert numpy dtypes to strings for JSON serialization
    dtypes_str = {col: str(dtype) for col, dtype in dtypes_dict.items()}
    
    # Save to JSON
    with open(f'data/{base_name}/{base_name}_dtypes.json', 'w') as f:
        json.dump(dtypes_str, f, indent=2)
    
    # Save summary statistics
    summary_stats = data.describe()
    
    # Save to text file
    with open(f'data/{base_name}/{base_name}_summary.txt', 'w') as f:
        f.write(f"Data Analysis Summary for {filename}\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("DATA TYPES:\n")
        f.write("=" * 50 + "\n")
        for col, dtype in dtypes_dict.items():
            f.write(f"{col}: {dtype}\n")
        f.write("\n")
        f.write("SUMMARY STATISTICS:\n")
        f.write("=" * 50 + "\n")
        f.write(summary_stats.to_string())
    
    return dtypes_str, summary_stats


def create_sqlite_db(data: pd.DataFrame, filename: str):
    """Create SQLite database from uploaded data"""
    base_name = os.path.splitext(filename)[0]
    db_path = f'data/{base_name}/{base_name}.db'

    conn = sqlite3.connect(db_path)
    data.to_sql('data_table', conn, if_exists='replace', index=False)
    conn.close()
    
    return db_path


def execute_sql_query(db_path: str, query: str):
    """Execute SQL query on the database"""
    try:
        conn = sqlite3.connect(db_path)
        
        # Handle PRAGMA queries specially
        if query.strip().upper().startswith('PRAGMA'):
            cursor = conn.cursor()
            cursor.execute(query)
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            # Convert to DataFrame for consistent handling
            result = pd.DataFrame(rows, columns=columns)
            conn.close()
            return result, None
        else:
            # Regular pandas SQL execution
            result = pd.read_sql_query(query, conn)
            conn.close()
            return result, None
            
    except Exception as e:
        return None, str(e)


def get_table_info(db_path: str):
    """Get table schema information"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table schema
        cursor.execute("PRAGMA table_info(data_table)")
        columns = cursor.fetchall()
        
        # Get sample data
        cursor.execute("SELECT * FROM data_table LIMIT 3")
        sample_data = cursor.fetchall()
        
        # Get row count
        cursor.execute("SELECT COUNT(*) FROM data_table")
        row_count = cursor.fetchone()[0]
        
        conn.close()
        
        # Format table info with more detail
        schema_info = f"""DATABASE INFORMATION:
Table Name: data_table
Total Rows: {row_count}

COLUMNS:
"""
        for col in columns:
            schema_info += f"  - {col[1]} ({col[2]}) - Column ID: {col[0]}\n"
        
        schema_info += "\nSAMPLE DATA (first 3 rows):\n"
        column_names = [col[1] for col in columns]
        schema_info += f"Columns: {', '.join(column_names)}\n"
        for i, row in enumerate(sample_data, 1):
            schema_info += f"Row {i}: {row}\n"
        
        schema_info += """
QUERY EXAMPLES:
- To see column names: PRAGMA table_info(data_table)
- To see all data: SELECT * FROM data_table
- To count rows: SELECT COUNT(*) FROM data_table
- To see first 10 rows: SELECT * FROM data_table LIMIT 10
"""
        
        return schema_info
    except Exception as e:
        return f"Error getting table info: {str(e)}"


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