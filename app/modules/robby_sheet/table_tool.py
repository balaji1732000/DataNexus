import re
import sys
import os
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
import streamlit as st
from langchain.chains import LLMChain
from langchain_community.llms import Replicate
from langchain_core.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback

# from pandasai.callbacks import BaseCallback
from streamlit_chat import message
import replicate

from pandasai import SmartDataframe
from dotenv import load_dotenv
from pandasai.responses.response_parser import ResponseParser

load_dotenv()
# The REPLICATE_API_TOKEN will now be available in the environment variables
os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")


class PandasAgent:

    @staticmethod
    def count_tokens_agent(agent, query):
        """
        Count the tokens used by the CSV Agent
        """
        with get_openai_callback() as cb:
            result = agent(query)
            st.write(f"Spent a total of {cb.total_tokens} tokens")
        return result

    def __init__(self):
        pass

    # class StreamlitCallback(BaseCallback):
    #     def __init__(self, container) -> None:
    #         """Initialize callback handler."""
    #         self.container = container

    #     def on_code(self, response: str):
    #         self.container.code(response)

    class StreamlitResponse(ResponseParser):
        def __init__(self, context) -> None:
            super().__init__(context)

        def format_dataframe(self, result):
            st.dataframe(result["value"])
            return

        def format_plot(self, result):
            st.image(result["value"])
            return

        def format_other(self, result):
            st.write(result["value"])
            return

    def get_agent_response(self, uploaded_file_content, query):
        llm = Replicate(
            model="snowflake/snowflake-arctic-instruct",
            model_kwargs={"temperature": 0.75, "max_length": 500, "top_p": 1},
        )
        # llm = replicate(
        #     model="snowflake/snowflake-arctic-instruct",
        # )
        container = st.container()
        # Using a dictionary for config instead of a set
        pandas_ai = SmartDataframe(
            uploaded_file_content,
            config={
                "llm": llm,
                "response_parser": self.StreamlitResponse,
                # "callback": self.StreamlitCallback(container),
            },
        )

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        response = pandas_ai.chat(query)
        fig = plt.gcf()
        if fig.get_axes():
            # Adjust the figure size
            fig.set_size_inches(12, 6)

            # Adjust the layout tightness
            plt.tight_layout()
            buf = BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            st.image(buf, caption="Generated Plot")

        sys.stdout = old_stdout
        print(response)
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
