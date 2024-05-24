import re
import sys
import os
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
import streamlit as st
from langchain_community.llms import Replicate
from langchain.callbacks import get_openai_callback
from streamlit_chat import message
import replicate
from pandasai import SmartDataframe
from pandasai.callbacks import BaseCallback
from dotenv import load_dotenv
from pandasai.responses.response_parser import ResponseParser
import pandas as pd
from langchain.chains import LLMChain
from langchain_community.llms import Replicate
from langchain_core.prompts import PromptTemplate


load_dotenv()

# The REPLICATE_API_TOKEN will now be available in the environment variables
os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")


class PandasAgent:
    @staticmethod
    def count_tokens_agent(agent, query):
        """Count the tokens used by the CSV Agent"""
        with get_openai_callback() as cb:
            result = agent(query)
            st.write(f"Spent a total of {cb.total_tokens} tokens")
        return result

    def __init__(self):
        pass

    class StreamlitResponse(ResponseParser):
        def __init__(self, context) -> None:
            super().__init__(context)

        def format_dataframe(self, result):
            st.dataframe(result["value"])
            return

        def format_plot(self, result):
            st.image(result["value"])
            return

        def format_number(self, result):
            st.write(result["value"])
            return

        def format_string(self, result):
            st.write(result["value"])
            return

        def format_other(self, result):
            st.write(result["value"])
            return

        def handle_response(self, result):
            if isinstance(result, dict) and "value" in result:
                value = result["value"]
                if isinstance(value, pd.DataFrame):
                    self.format_dataframe(result)
                elif isinstance(value, BytesIO):
                    self.format_plot(result)
                elif isinstance(value, (int, float)):
                    self.format_number(result)
                elif isinstance(value, str):
                    self.format_string(result)
                else:
                    self.format_other(result)
            else:
                st.write("Unsupported result format")
                st.write(result)
            return

    class StreamlitCallback(BaseCallback):
        def __init__(self, container) -> None:
            """Initialize callback handler."""
            self.container = container

        def on_code(self, response: str):
            self.container.code(response)

    def get_agent_response(self, uploaded_file_content, query):
        # llm = Replicate(
        #     model="snowflake/snowflake-arctic-instruct",
        #     model_kwargs={"temperature": 0.75, "max_length": 500, "top_p": 1},
        # )

        llm = Replicate(
            model="meta/meta-llama-3-8b-instruct",
            model_kwargs={"temperature": 0.75, "max_length": 500, "top_p": 1},
        )
        container = st.container()
        pandas_ai = SmartDataframe(
            uploaded_file_content,
            config={
                "llm": llm,
                "response_parser": self.StreamlitResponse,  # Correctly pass the class, not an instance
                "callback": self.StreamlitCallback(container),
            },
        )

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        try:
            print("Running pandas_ai with query:", query)
            response = pandas_ai.chat(query)  # Execute the query with pandas_ai
            print("Response from pandas_ai:", response)
            print("Response type:", type(response))
            self.StreamlitResponse(self).handle_response({"value": response})
            fig = plt.gcf()
            if fig.get_axes():
                fig.set_size_inches(12, 6)
                plt.tight_layout()
                buf = BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                st.image(buf, caption="Generated Plot")
        except Exception as e:
            print("Error during pandas_ai.run execution:", e)
            response = None
        sys.stdout = old_stdout
        return response, captured_output

    def process_agent_thoughts(self, captured_output):
        thoughts = captured_output.getvalue()
        cleaned_thoughts = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", thoughts)
        cleaned_thoughts = re.sub(r"\[1m>", "", cleaned_thoughts)
        return cleaned_thoughts

    def display_agent_thoughts(self, cleaned_thoughts):
        with st.expander("Display the agent's thoughts"):
            st.write(cleaned_thoughts)

    def update_chat_history(self, query, result):
        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("agent", result))

    def display_chat_history(self):
        for i, (sender, message_text) in enumerate(st.session_state.chat_history):
            if sender == "user":
                message(message_text, is_user=True, key=f"{i}_user")
            else:
                message(message_text, key=f"{i}")


# # Initialize chat history if not already done
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # File upload section
# uploaded_file = st.file_uploader("Choose a file")
# if uploaded_file is not None:
#     uploaded_file_content = pd.read_csv(uploaded_file)

#     # Query input section
#     query = st.text_input("Ask DataNexus")
#     if st.button("Submit"):
#         agent = PandasAgent()
#         response, captured_output = agent.get_agent_response(
#             uploaded_file_content, query
#         )
#         cleaned_thoughts = agent.process_agent_thoughts(captured_output)
#         agent.display_agent_thoughts(cleaned_thoughts)
#         agent.update_chat_history(query, response)
#         agent.display_chat_history()
