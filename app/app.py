import time
import streamlit as st
import os
import importlib
import sys
import pandas as pd
from io import BytesIO
from util import load_lottie, stream_data, welcome_message, introduction_message
from prediction_model import prediction_model_pipeline
from cluster_model import cluster_model_pipeline
from regression_model import regression_model_pipeline
from visualization import data_visualization
from src.util import read_file_from_streamlit
import base64

# Modules for Chat Analyser
import os
import streamlit as st
from io import StringIO
import re
import sys
from modules.robby_sheet.table_tool import PandasAgent
from modules.history import ChatHistory
from modules.layout import Layout
from modules.utils import Utilities
from modules.sidebar import Sidebar
from dotenv import load_dotenv

load_dotenv()


# To be able to update the changes made to modules in localhost (press r)
def reload_module(module_name):
    import importlib
    import sys

    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    return sys.modules[module_name]


history_module = reload_module("modules.history")
layout_module = reload_module("modules.layout")
utils_module = reload_module("modules.utils")
sidebar_module = reload_module("modules.sidebar")

ChatHistory = history_module.ChatHistory
Layout = layout_module.Layout
Utilities = utils_module.Utilities
Sidebar = sidebar_module.Sidebar

st.set_page_config(page_title="Streamline Analyst", page_icon=":rocket:", layout="wide")

# Instantiate the main components
layout, sidebar, utils = Layout(), Sidebar(), Utilities()

# Sidebar
st.sidebar.title("DataNexus")

# Create a dropdown menu for selecting the page
page = st.sidebar.selectbox(
    "Go to",
    ["Home", "GenAI Data-to-Modeling", "AI Data Agents"],
)

# TITLE SECTION
with st.container():
    if page == "Home":
        layout.show_header("Chat ")
        st.title("Welcome to DataNexus Analysis")
        if "initialized" not in st.session_state:
            st.session_state.initialized = True
        if st.session_state.initialized:
            st.session_state.welcome_message = welcome_message()
            st.write(stream_data(st.session_state.welcome_message))
            time.sleep(0.5)
            st.session_state.initialized = False
        else:
            st.write(st.session_state.welcome_message)

        def get_image_string(path):
            with open(path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode()

        file_path = "assets/data-analysis-4179002-3479081-ezgif.com-optimize.gif"
        image_string = get_image_string(file_path)
        st.markdown(
            f'<p align="center"><img src="data:image/gif;base64,{image_string}" alt="DataNexus in action"></p>',
            unsafe_allow_html=True,
        )
        st.write(
            """  
            ### Key Features  
            - **Intuitive Chat Interface**: Interact with your data through natural language commands. No coding or technical expertise required.  
            - **AI-powered Data Assistant**: Let DataNexus handle the heavy lifting. It automates data cleaning, preprocessing, and feature engineering.  
            - **Intelligent Model Selection**: DataNexus analyzes your data and recommends the most suitable machine learning model for your specific needs.  
            - **Visualized Insights**: Understand your data with interactive visualizations and clear explanations presented through the chat interface.  
            - **Streamlined Workflow**: DataNexus guides you through each step, ensuring a smooth and efficient data analysis experience.  
            """
        )

    # DATA ANALYSIS SECTION
    elif page == "GenAI Data-to-Modeling":
        with st.container():
            st.divider()
            st.header("Let's Get Started")
            left_column, right_column = st.columns([6, 4])
            with left_column:
                uploaded_file = st.file_uploader(
                    "Choose a data file. Your data won't be stored as well!",
                    accept_multiple_files=False,
                    type=["csv", "json", "xls", "xlsx"],
                )
                if uploaded_file:
                    if uploaded_file.getvalue():
                        uploaded_file.seek(0)
                        st.session_state.DF_uploaded = read_file_from_streamlit(
                            uploaded_file
                        )
                        st.session_state.is_file_empty = False
                    else:
                        st.session_state.is_file_empty = True
            with right_column:
                MODE = st.selectbox(
                    "Select proper data analysis mode",
                    (
                        "Predictive Classification",
                        "Clustering Model",
                        "Regression Model",
                        "Data Visualization",
                    ),
                    key="data_analysis_dropdown",  # Unique key for the dropdown widget
                )
                st.write(f"Data analysis mode: :green[{MODE}]")
            # Proceed Button
            is_proceed_enabled = (
                uploaded_file is not None
                or uploaded_file is not None
                and MODE == "Data Visualization"
            )
            # Initialize the 'button_clicked' state
            if "button_clicked" not in st.session_state:
                st.session_state.button_clicked = False
            if st.button(
                "Start Analysis", type="primary", disabled=not is_proceed_enabled
            ):
                st.session_state.button_clicked = True
            if "is_file_empty" in st.session_state and st.session_state.is_file_empty:
                st.caption("Your data file is empty!")
            # Start Analysis
            if st.session_state.button_clicked:
                with st.container():
                    if "DF_uploaded" not in st.session_state:
                        st.error("File is empty!")
                    else:
                        if MODE == "Predictive Classification":
                            prediction_model_pipeline(
                                st.session_state.DF_uploaded,  # API_KEY, GPT_MODEL
                            )
                        elif MODE == "Clustering Model":
                            cluster_model_pipeline(
                                st.session_state.DF_uploaded  # API_KEY, GPT_MODEL
                            )
                        elif MODE == "Regression Model":
                            regression_model_pipeline(
                                st.session_state.DF_uploaded,  # API_KEY, GPT_MODEL
                            )
                        elif MODE == "Data Visualization":
                            data_visualization(st.session_state.DF_uploaded)

    elif page == "AI Data Agents":
        layout, sidebar, utils = Layout(), Sidebar(), Utilities()
        layout.show_header("Data")
        st.session_state.setdefault("reset_chat", False)
        uploaded_file = utils.handle_upload(["csv", "xlsx"])
        if uploaded_file:
            sidebar.about()
            uploaded_file_content = BytesIO(uploaded_file.getvalue())
            if (
                uploaded_file.type
                == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                or uploaded_file.type == "application/vnd.ms-excel"
            ):
                df = pd.read_excel(uploaded_file_content)
            else:
                df = pd.read_csv(uploaded_file_content)
            st.session_state.df = df
            if "chat_history" not in st.session_state:
                st.session_state["chat_history"] = []
            csv_agent = PandasAgent()
            with st.form(key="query"):
                query = st.text_input(
                    "Ask DataNexus",
                    value="",
                    type="default",
                    placeholder="e.g., How many rows?",
                )
                submitted_query = st.form_submit_button("Submit")
            if submitted_query:
                result, captured_out = csv_agent.get_agent_response(df, query)
                st.markdown(
                    f"<div style='background-color: #ADD8E6; padding: 10px; border-radius: 5px; text-align: left; color: black;'>"
                    f"{result}</div>",
                    unsafe_allow_html=True,
                )
                # cleaned_thoughts = csv_agent.process_agent_thoughts(captured_output)
                # csv_agent.display_agent_thoughts(cleaned_thoughts)
                # csv_agent.update_chat_history(query, result)
            # csv_agent.display_chat_history()
            if st.session_state.df is not None:
                st.subheader("Current dataframe:")
                st.write(st.session_state.df)
