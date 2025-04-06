import streamlit as st
import pandas as pd
import copy

import dotenv
import io

dotenv.load_dotenv()

from prompts import get_seperator, get_visualization_suggestions, generate_plot

st.set_page_config(
    page_title="CSV Reader",
    page_icon="📊",
    layout="wide",
)


with st.sidebar:
    st.title("Data Analysis")


uploaded_file = st.file_uploader(
    "Choose a file", type="csv", accept_multiple_files=False
)

if uploaded_file is not None:
    preview = copy.copy(uploaded_file)
    df = pd.read_csv(preview, nrows=3)
    sep = get_seperator(df.head(2))
    st.write("return", repr(sep))
    df = pd.read_csv(uploaded_file, sep=sep)  # replace it with `sep`

    st.write(f"DataFrame: ({uploaded_file.name})")
    st.dataframe(df)

    columns = df.columns
    buffer = io.StringIO()
    df.info(buf=buffer)
    df_info_str = buffer.getvalue()
    df_head_df = df.head()
    df_desc_df = df.describe(include="all").fillna("N/A")

    st.subheader("DataFrame Info & Description")
    col1, col2 = st.columns(2)
    with col1:
        st.text("DataFrame Info:")
        st.text(df_info_str)
    with col2:
        st.text("DataFrame Description:")
        st.dataframe(df_desc_df)

    st.subheader("Suggested Visualizations")
    num_suggestions = st.slider("Number of plots to suggest:", 1, 5, 3)

    with st.spinner("Asking AI for visualization ideas..."):
        suggestions = get_visualization_suggestions(
            columns, df_info_str, df_head_df, df_desc_df, num_plots=num_suggestions
        )

    if suggestions:
        st.success(f"Received {len(suggestions)} plot suggestions from the LLM.")
        if not suggestions:
            st.info("LLM returned an empty list of suggestions.")

        for i, suggestion in enumerate(suggestions):
            st.markdown("---")  # Divider
            st.write(f"**Suggestion {i + 1}:**", suggestion)  # params for plot
            with st.spinner(f"Generating plot {i + 1}..."):
                fig = generate_plot(df.copy(), suggestion)

            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Could not generate Plot {i + 1} based on the suggestion.")
    elif suggestions == []:
        st.info("No valid suggestions were generated.")
    else:
        st.error("Failed to get visualization suggestions.")
