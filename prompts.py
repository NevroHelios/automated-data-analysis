import json
import os

from openai import Client
import pandas as pd
import streamlit as st
import plotly.express as px

client = Client(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


def get_seperator(df) -> str:
    """After anallyzing the data the model decides what the data separator is"""
    prompt = f"""
    From the data below, tell me what the literal separator character is:
    {df.to_csv(index=False)}

    Respond with only the literal character, e.g. ',', '\t', ';'
    """

    message = [
        {
            "role": "system",
            "content": "You are master at finding separators in csv files. You tell only the answer and nothing extra",
        },
        {"role": "user", "content": prompt},
    ]

    res = client.chat.completions.create(
        messages=message, model="meta-llama/llama-4-scout-17b-16e-instruct"
    )
    return res.choices[0].message.content.strip().strip("<>").strip()


def generate_plot(df, suggestion):
    """Generates a Plotly plot based on the LLM suggestion."""
    plot_type = suggestion.get("plot_type")
    x_col = suggestion.get("x_col")
    y_col = suggestion.get("y_col")
    color_col = suggestion.get("color_col")

    fig = None
    plot_title = f"{plot_type.capitalize()} plot" if plot_type else "Plot"

    try:
        # validating the columns
        required_cols = [c for c in [x_col, y_col, color_col] if c]
        if not required_cols:
            st.warning(f"Suggestion has no columns: {suggestion}")
            return None

        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            st.warning(
                f"Suggested columns not found in DataFrame: {missing_cols} for suggestion {suggestion}. Cannot generate plot."
            )
            return None

        if plot_type == "histogram" and x_col:
            fig = px.histogram(
                df, x=x_col, title=f"Distribution of {x_col}", color=color_col
            )
        elif plot_type == "bar" and x_col and y_col:
            # simple agg
            try:
                if pd.api.types.is_numeric_dtype(df[y_col]):  # say y is numeric
                    grouped_df = (
                        df.groupby(x_col, observed=True)[y_col].mean().reset_index()
                    )
                    fig = px.bar(
                        grouped_df,
                        x=x_col,
                        y=y_col,
                        title=f"Average {y_col} by {x_col}",
                        color=color_col,
                    )
                else:  # y is not
                    grouped_df = (
                        df.groupby(x_col, observed=True)
                        .size()
                        .reset_index(name="count")
                    )
                    fig = px.bar(
                        grouped_df,
                        x=x_col,
                        y="count",
                        title=f"Count by {x_col}",
                        color=color_col,
                    )
            except (TypeError, ValueError, KeyError) as group_err:
                st.warning(
                    f"Could not automatically group/aggregate for bar chart ({x_col}, {y_col}): {group_err}. Trying direct bar plot."
                )
                # alreaady agg??
                try:
                    fig = px.bar(
                        df,
                        x=x_col,
                        y=y_col,
                        title=f"{y_col} by {x_col}",
                        color=color_col,
                    )
                except Exception as direct_bar_err:
                    st.warning(f"Direct bar plot failed too: {direct_bar_err}")
                    fig = None  # idk but needed

        elif plot_type == "scatter" and x_col and y_col:
            fig = px.scatter(
                df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}", color=color_col
            )
        elif plot_type == "box" and x_col:
            if y_col and pd.api.types.is_numeric_dtype(
                df[y_col]
            ):  # box plot per category
                fig = px.box(
                    df,
                    x=x_col,
                    y=y_col,
                    title=f"Distribution of {y_col} by {x_col}",
                    color=color_col if color_col else x_col,
                )
            elif pd.api.types.is_numeric_dtype(
                df[x_col]
            ):  # single box plot for the numerical column
                fig = px.box(
                    df, y=x_col, title=f"Distribution of {x_col}"
                )  # use y axis for single box
            else:
                st.warning(
                    f"Box plot requires a numerical column for '{x_col}' or specified in 'y_col'."
                )

        elif plot_type == "line" and x_col and y_col:
            try:
                df_sorted = df.sort_values(by=x_col)
                fig = px.line(
                    df_sorted,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} over {x_col}",
                    color=color_col,
                )
            except Exception as line_err:
                st.warning(
                    f"Could not generate line plot (ensure '{x_col}' can be sorted): {line_err}"
                )

        else:
            st.warning(
                f"Plot type '{plot_type}' with columns X='{x_col}', Y='{y_col}', Color='{color_col}' is not supported or columns missing/invalid."
            )

        if fig and color_col and color_col in df.columns:
            try:  # A color title
                fig.update_layout(
                    title=f"{fig.layout.title.text} (Colored by {color_col})"
                )
            except Exception:
                pass

        return fig

    except Exception as e:
        st.error(
            f"Error generating plot '{plot_type}' for columns X='{x_col}', Y='{y_col}': {e}"
        )
        import traceback

        st.text(traceback.format_exc())  # More detailed error for debugging
        return None


def get_visualization_suggestions(
    columns, df_info, df_head, df_description, num_plots=3
):
    """Asks the LLM to suggest multiple plot types and columns in JSON list format."""

    prompt = f"""
    Analyze the following DataFrame context and suggest {num_plots} distinct and insightful plots for initial data exploration.
    Prioritize plots that reveal distributions, relationships between variables, or comparisons across categories.

    Available plot types: 'scatter', 'histogram', 'bar', 'box', 'line'

    DataFrame Context:
    Columns: {columns.tolist()}
    Data Types and Non-Null Counts:
    {df_info}
    First 5 Rows:
    {df_head.to_string()}
    Statistical Description:
    {df_description.to_string()}

    Instructions:
    1. Analyze the columns, their data types (object/category are categorical, int/float are numerical), distributions (from describe), and potential relationships.
    2. Choose up to {num_plots} DIFFERENT plot types and appropriate columns from the available list ('scatter', 'histogram', 'bar', 'box', 'line'). Select columns that make sense for the chosen plot type (e.g., numerical for histogram x, numerical x/y for scatter, categorical x / numerical y for bar/box).
    3. For each suggested plot, identify the necessary column(s): 'x_col', and optionally 'y_col', 'color_col'.
       - 'histogram', 'box' usually need 'x_col' (numerical). Box can optionally use categorical 'x_col' and numerical 'y_col'.
       - 'bar' typically needs 'x_col' (categorical) and 'y_col' (numerical, will be aggregated).
       - 'scatter', 'line' typically need 'x_col' and 'y_col' (numerical). Line often implies order (like time) in x_col.
       - 'color_col' (usually categorical) is optional for adding a third dimension.
    4. Return ONLY a JSON list, where each element is an object representing one plot suggestion. Each object must have keys: 'plot_type', 'x_col', and optionally 'y_col', 'color_col'.
    5. Ensure the suggestions are diverse if possible (e.g., don't suggest 3 histograms if other plot types are suitable). If fewer than {num_plots} distinct insightful plots seem appropriate, return fewer.

    Example JSON Output:
    [
      {{"plot_type": "scatter", "x_col": "Age", "y_col": "Salary", "color_col": "Department"}},
      {{"plot_type": "histogram", "x_col": "Salary"}},
      {{"plot_type": "bar", "x_col": "Department", "y_col": "Salary"}}
    ]

    Output only the JSON list. Do not include any other text, explanation, or markdown formatting.
    """

    message = [
        {
            "role": "system",
            "content": "You are an expert data analyst specializing in visualization recommendations. Return ONLY the requested JSON list of plot suggestions.",
        },
        {"role": "user", "content": prompt},
    ]

    try:
        res = client.chat.completions.create(
            messages=message, model="meta-llama/llama-4-scout-17b-16e-instruct"
        )
        response_content = res.choices[0].message.content.strip()
        ## for debugging
        # st.text("LLM Raw Response:")
        # st.text_area("Raw Output", response_content, height=150)

        # clean if llm generates markdown
        if response_content.startswith("```json"):
            response_content = response_content[7:]
            if response_content.endswith("```"):
                response_content = response_content[:-3]
        elif response_content.startswith("```"):
            response_content = response_content[3:]
            if response_content.endswith("```"):
                response_content = response_content[:-3]
        response_content = response_content.strip()

        if not response_content.startswith("[") or not response_content.endswith("]"):
            st.error(
                f"Error: LLM response doesn't appear to be a JSON list. Response:\n{response_content}"
            )
            return None

        suggestions = json.loads(response_content)
        if isinstance(suggestions, list):
            valid_suggestions = []
            for item in suggestions:
                if (
                    isinstance(item, dict)
                    and item.get("plot_type")
                    and item.get("x_col")
                ):
                    valid_suggestions.append(item)
                else:
                    st.warning(f"Skipping invalid suggestion structure: {item}")
            return valid_suggestions
        else:
            st.error(
                f"Error: LLM response was valid JSON but not a list. Response:\n{response_content}"
            )
            return None

    except json.JSONDecodeError as json_err:
        st.error(
            f"Error: LLM did not return valid JSON. Error: {json_err}. Response was:\n{response_content}"
        )
        return None
    except Exception as e:
        st.error(f"Error getting suggestions from LLM: {e}")
        import traceback

        st.text(traceback.format_exc())
        return None
