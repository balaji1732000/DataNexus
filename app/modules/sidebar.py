import streamlit as st


class Sidebar:

    MODEL_OPTIONS = ["snowflake/snowflake-arctic-instruct", "gpt-4"]
    TEMPERATURE_MIN_VALUE = 0.0
    TEMPERATURE_MAX_VALUE = 1.0
    TEMPERATURE_DEFAULT_VALUE = 0.0
    TEMPERATURE_STEP = 0.01

    @staticmethod
    def about():
        about = st.sidebar.expander("🤖 About DataNexus Chat")
        sections = [
            "#### DataNexus Chat is an AI-powered chatbot designed to simplify data analysis and empower users of all skill levels to harness the full potential of their data effortlessly. 📊",
            "#### It leverages the power of conversational AI and machine learning automation to guide users through the entire data analysis process, from data upload to generating insights. 🔬",
            "#### With its intuitive chat interface, DataNexus allows users to interact with their data through natural language commands, eliminating the need for coding or technical expertise. 💬",
            "#### The AI-powered data assistant handles complex tasks such as data cleaning, preprocessing, feature engineering, and intelligent model selection, ensuring accurate and reliable results. 🧠",
            "#### DataNexus aims to democratize data-driven decision-making by making data analysis accessible and efficient for everyone, regardless of their technical background. 🌐",
        ]
        for section in sections:
            about.write(section)

    @staticmethod
    def reset_chat_button():
        if st.button("Reset chat"):
            st.session_state["reset_chat"] = True
        st.session_state.setdefault("reset_chat", False)

    def model_selector(self):
        model = st.selectbox(label="Model", options=self.MODEL_OPTIONS)
        st.session_state["model"] = model

    def temperature_slider(self):
        temperature = st.slider(
            label="Temperature",
            min_value=self.TEMPERATURE_MIN_VALUE,
            max_value=self.TEMPERATURE_MAX_VALUE,
            value=self.TEMPERATURE_DEFAULT_VALUE,
            step=self.TEMPERATURE_STEP,
        )
        st.session_state["temperature"] = temperature

    def show_options(self):
        with st.sidebar.expander("🛠️ DataNexus Tools", expanded=False):

            self.reset_chat_button()
            self.model_selector()
            self.temperature_slider()
            st.session_state.setdefault("model", self.MODEL_OPTIONS[0])
            st.session_state.setdefault("temperature", self.TEMPERATURE_DEFAULT_VALUE)
