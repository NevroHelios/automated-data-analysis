import pandas as pd
import sqlite3
import os
import json
from datetime import datetime



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


