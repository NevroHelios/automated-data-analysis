from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.tools import BaseTool
from typing import List, Type, Optional, Literal
import os
import streamlit as st
import plotly.express as px

## model init
model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct", api_key=os.getenv("GROQ_API_KEY")
)


class ShowCharts(BaseModel):
    columns: List[str] = Field(description="List of columns to plot the graph. first one is `x` and second one is `y`")
    plot_type: Literal['bar', 'line', 'hist'] = Field(description="the type of plot to draw")


class ChartTool(BaseTool):
    name: str = "Shows relevent chart"
    description: str = "useful when given the param you need to show relevent charts"
    args_schema: Type[BaseModel] = ShowCharts

    def _run(self, columns: List[str],
             plot_type: str):
        if plot_type == 'bar':
            fig = px.bar()