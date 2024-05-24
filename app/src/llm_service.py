import os
import yaml
import json
import re
import streamlit as st
# from langchain.prompts import PromptTemplate
# from langchain.schema import HumanMessage
# from langchain.chains import LLMChain
from langchain_community.llms import Replicate
import replicate
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()
# The REPLICATE_API_TOKEN will now be available in the environment variables
os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")

config_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)


def decide_target_attribute(attributes, types_info, head_info):
    """
    Determines the target attribute for modeling based on dataset attributes and characteristics.

    Parameters:
    - attributes: A list of dataset attributes.
    - types_info: Information about the data types of the attributes.
    - head_info: A snapshot of the dataset's first few rows.
    - model_type (int, optional): The model type to use for decision making (default 4).
    - user_api_key (str, optional): The user's API key for OpenAI.

    Returns:
    - The name of the recommended target attribute. Please refer to prompt templates in config.py for details.

    Raises:
    - Exception: If unable to access the OpenAI API or another error occurs.
    """
    try:
        # model_name = model4_name if model_type == 4 else model3_name
        # user_api_key = api_key if user_api_key is None else user_api_key

        # llm = Replicate(model="snowflake/snowflake-arctic-instruct")
        os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")
        template = config["decide_target_attribute_template"]
        prompt_template = PromptTemplate(
            input_variables=["attributes", "types_info", "head_info"], template=template
        )
        summary_prompt = prompt_template.format(
            attributes=attributes, types_info=types_info, head_info=head_info
        )
        print(summary_prompt)
        input = {"prompt": summary_prompt}
        llm_answer = replicate.run("snowflake/snowflake-arctic-instruct", input=input)
        print("LLM Answer:", llm_answer)
        # if "```json" in llm_answer:
        #     match = re.search(r"```json\n(.*?)```", llm_answer, re.DOTALL)
        #     if match:
        #         json_str = match.group(1)

        llm_answer = "".join(llm_answer)
        json_str = llm_answer[llm_answer.find("{") : llm_answer.rfind("}") + 1]

        # else:
        #     json_str = llm_answer

        output = json.loads(json_str)["target"]
        print(output)
        return output
    except Exception as e:
        # st.error(
        #     "Cannot access the OpenAI API. Please check your API key or network connection."
        # )
        st.error(f"Error  decide_target_attribute: {e}")
        st.stop()


def decide_encode_type(attributes, data_frame_head):
    """
    Decides the encoding type for given attributes using a language model via the Replicate API.

    Parameters:
    - attributes (list): A list of attributes for which to decide the encoding type.
    - data_frame_head (DataFrame): The head of the DataFrame containing the attributes.

    Returns:
    - A JSON object containing the recommended encoding types for the given attributes.

    Raises:
    - Exception: If there is an issue accessing the Replicate API.
    """
    try:
        os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")
        template = config["numeric_attribute_template"]
        prompt_template = PromptTemplate(
            input_variables=["attributes", "data_frame_head"], template=template
        )
        summary_prompt = prompt_template.format(
            attributes=attributes, data_frame_head=data_frame_head
        )
        print(summary_prompt)

        input = {"prompt": summary_prompt}
        llm_answer = replicate.run("snowflake/snowflake-arctic-instruct", input=input)
        print("LLM Answer:", llm_answer)

        llm_answer = "".join(llm_answer)
        json_str = llm_answer[llm_answer.find("{") : llm_answer.rfind("}") + 1]

        return json.loads(json_str)
    except Exception as e:
        st.error(f"Error encode_type: {e}")
        st.stop()


def decide_fill_null(attributes, types_info, description_info):
    """
    Decides the best way to fill null values for given attributes using a language model via the Replicate API.

    Parameters:
    - attributes (list): List of attribute names to consider for filling null values.
    - types_info (dict): Data types of the attributes.
    - description_info (str): Descriptive information about the attributes.

    Returns:
    - A JSON object with recommendations for filling null values.

    Raises:
    - Exception: If there is an issue accessing the Replicate API.
    """
    try:
        os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")
        template = config["null_attribute_template"]
        prompt_template = PromptTemplate(
            input_variables=["attributes", "types_info", "description_info"],
            template=template,
        )
        summary_prompt = prompt_template.format(
            attributes=attributes,
            types_info=types_info,
            description_info=description_info,
        )
        print(summary_prompt)

        input = {"prompt": summary_prompt}
        llm_answer = replicate.run("snowflake/snowflake-arctic-instruct", input=input)
        print("LLM Answer:", llm_answer)

        llm_answer = "".join(llm_answer)
        json_str = llm_answer[llm_answer.find("{") : llm_answer.rfind("}") + 1]

        return json.loads(json_str)
    except Exception as e:
        st.error(f"Error decide_fill_null: {e}")
        st.stop()


def decide_model(shape_info, head_info, nunique_info, description_info):
    """
    Decides the most suitable machine learning model based on dataset characteristics.

    Parameters:
    - shape_info (dict): Information about the shape of the dataset.
    - head_info (str or DataFrame): The head of the dataset or its string representation.
    - nunique_info (dict): Information about the uniqueness of dataset attributes.
    - description_info (str): Descriptive information about the dataset.

    Returns:
    - A JSON object containing the recommended model and configuration.

    Raises:
    - Exception: If there is an issue accessing the Replicate API.
    """
    try:
        os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")
        template = config["decide_model_template"]
        prompt_template = PromptTemplate(
            input_variables=[
                "shape_info",
                "head_info",
                "nunique_info",
                "description_info",
            ],
            template=template,
        )
        summary_prompt = prompt_template.format(
            shape_info=shape_info,
            head_info=head_info,
            nunique_info=nunique_info,
            description_info=description_info,
        )
        print(summary_prompt)

        input = {"prompt": summary_prompt}
        llm_answer = replicate.run("snowflake/snowflake-arctic-instruct", input=input)
        print("LLM Answer:", llm_answer)

        llm_answer = "".join(llm_answer)
        json_str = llm_answer[llm_answer.find("{") : llm_answer.rfind("}") + 1]

        return json.loads(json_str)
    except Exception as e:
        st.error(f"Error decide_model: {e}")
        st.stop()


def decide_cluster_model(shape_info, description_info, cluster_info):
    """
    Determines the appropriate clustering model based on dataset characteristics.

    Parameters:
    - shape_info: Information about the dataset shape.
    - description_info: Descriptive statistics or information about the dataset.
    - cluster_info: Additional information relevant to clustering.

    Returns:
    - A JSON object with the recommended clustering model and parameters.

    Raises:
    - Exception: If there is an issue accessing the Replicate API.
    """
    try:
        os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")
        template = config["decide_clustering_model_template"]
        prompt_template = PromptTemplate(
            input_variables=["shape_info", "description_info", "cluster_info"],
            template=template,
        )
        summary_prompt = prompt_template.format(
            shape_info=shape_info,
            description_info=description_info,
            cluster_info=cluster_info,
        )
        print(summary_prompt)

        input = {"prompt": summary_prompt}
        llm_answer = replicate.run("snowflake/snowflake-arctic-instruct", input=input)
        print("LLM Answer:", llm_answer)

        llm_answer = "".join(llm_answer)
        json_str = llm_answer[llm_answer.find("{") : llm_answer.rfind("}") + 1]

        return json.loads(json_str)
    except Exception as e:
        st.error(f"Error cluster_model: {e}")
        st.stop()


def decide_regression_model(shape_info, description_info, Y_name):
    """
    Determines the appropriate regression model based on dataset characteristics and the target variable.

    Parameters:
    - shape_info: Information about the dataset shape.
    - description_info: Descriptive statistics or information about the dataset.
    - Y_name: The name of the target variable.

    Returns:
    - A JSON object with the recommended regression model and parameters.

    Raises:
    - Exception: If there is an issue accessing the Replicate API.
    """
    try:

        template = config["decide_regression_model_template"]
        prompt_template = PromptTemplate(
            input_variables=["shape_info", "description_info", "Y_name"],
            template=template,
        )
        summary_prompt = prompt_template.format(
            shape_info=shape_info, description_info=description_info, Y_name=Y_name
        )
        print(summary_prompt)

        input = {"prompt": summary_prompt}
        llm_answer = replicate.run("snowflake/snowflake-arctic-instruct", input=input)
        print("LLM Answer:", llm_answer)

        llm_answer = "".join(llm_answer)
        json_str = llm_answer[llm_answer.find("{") : llm_answer.rfind("}") + 1]

        return json.loads(json_str)
    except Exception as e:
        st.error(f"Error decide_regression_model: {e}")
        st.stop()


def decide_test_ratio(shape_info):
    """
    Determines the appropriate train-test split ratio based on dataset characteristics.

    Parameters:
    - shape_info: Information about the dataset shape.

    Returns:
    - The recommended train-test split ratio as a float.

    Raises:
    - Exception: If there is an issue accessing the Replicate API.
    """
    try:
        os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")
        template = config["decide_test_ratio_template"]
        prompt_template = PromptTemplate(
            input_variables=["shape_info"], template=template
        )
        summary_prompt = prompt_template.format(shape_info=shape_info)
        print(summary_prompt)

        input = {"prompt": summary_prompt}
        llm_answer = replicate.run("snowflake/snowflake-arctic-instruct", input=input)
        print("LLM Answer:", llm_answer)

        llm_answer = "".join(llm_answer)
        json_str = llm_answer[llm_answer.find("{") : llm_answer.rfind("}") + 1]

        return json.loads(json_str).get("test_ratio")
    except Exception as e:
        st.error(f"Error test_ratio: {e}")
        st.stop()


def decide_balance(shape_info, description_info, balance_info):
    """
    Determines the appropriate method to balance the dataset based on its characteristics.

    Parameters:
    - shape_info: Information about the dataset shape.
    - description_info: Descriptive statistics or information about the dataset.
    - balance_info: Additional information relevant to dataset balancing.

    Returns:
    - The recommended method to balance the dataset. Please refer to prompt templates in config.py for details.

    Raises:
    - Exception: If unable to access the Replicate API or another error occurs.
    """
    try:
        os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")
        template = config["decide_balance_template"]
        prompt_template = PromptTemplate(
            input_variables=["shape_info", "description_info", "balance_info"],
            template=template,
        )
        summary_prompt = prompt_template.format(
            shape_info=shape_info,
            description_info=description_info,
            balance_info=balance_info,
        )
        print(summary_prompt)

        input = {"prompt": summary_prompt}
        llm_answer = replicate.run("snowflake/snowflake-arctic-instruct", input=input)
        print("LLM Answer:", llm_answer)

        llm_answer = "".join(llm_answer)
        json_str = llm_answer[llm_answer.find("{") : llm_answer.rfind("}") + 1]

        return json.loads(json_str)["method"]
    except Exception as e:
        st.error(f"Error in decide_balance: {e}")
        st.stop()


# model4_name = config["model4_name"]
# model3_name = config["model3_name"]
# api_key = config["openai_api_key"]


# def decide_encode_type(attributes, data_frame_head):
#     """
#     Decides the encoding type for given attributes using a language model via the OpenAI API.

#     Parameters:
#     - attributes (list): A list of attributes for which to decide the encoding type.
#     - data_frame_head (DataFrame): The head of the DataFrame containing the attributes. This parameter is expected to be a representation of the DataFrame (e.g., a string or a small subset of the actual DataFrame) that gives an overview of the data.
#     - model_type (int, optional): Specifies the model to use. The default model_type=4 corresponds to a predefined model named `model4_name`. Another option is model_type=3, which corresponds to `model3_name`.
#     - user_api_key (str, optional): The user's OpenAI API key. If not provided, a default API key `api_key` is used.

#     Returns:
#     - A JSON object containing the recommended encoding types for the given attributes. Please refer to prompt templates in config.py for details.

#     Raises:
#     - Exception: If there is an issue accessing the OpenAI API, such as an invalid API key or a network connection error, the function will raise an exception with a message indicating the problem.
#     """
#     try:
#         # model_name = model4_name if model_type == 4 else model3_name
#         # user_api_key = api_key if user_api_key is None else user_api_key

#
#         llm = Replicate("snowflake/snowflake-arctic-instruct")
#         print(llm)
#         print("Integrated Successfully")
#         template = config["numeric_attribute_template"]
#         prompt_template = PromptTemplate(
#             input_variables=["attributes", "data_frame_head"], template=template
#         )
#         summary_prompt = prompt_template.format(
#             attributes=attributes, data_frame_head=data_frame_head
#         )
#         print(summary_prompt)

#         llm_answer = llm.invoke(summary_prompt)
#         print("LLM Answer:", llm_answer)
#         if "```json" in llm_answer.content:
#             match = re.search(r"```json\n(.*?)```", llm_answer.content, re.DOTALL)
#             if match:
#                 json_str = match.group(1)
#         else:
#             json_str = llm_answer.content
#         return json.loads(json_str)
#     except Exception as e:
#         # st.error(
#         #     "Cannot access the OpenAI API. Please check your API key or network connection."
#         # )
#         st.error(f"Error accessing the Replicate API: {e}")
#         st.stop()


# def decide_fill_null(attributes, types_info, description_info):
#     """
#     Decides the best encoding type for given attributes using an AI model via OpenAI API.

#     Parameters:
#     - attributes (list): List of attribute names to consider for encoding.
#     - data_frame_head (DataFrame or str): The head of the DataFrame or a string representation, providing context for the encoding decision.
#     - model_type (int, optional): The model to use, where 4 is the default. Can be customized to use a different model.
#     - user_api_key (str, optional): The user's OpenAI API key. If None, a default key is used.

#     Returns:
#     - dict: A JSON object with recommended encoding types for the attributes. Please refer to prompt templates in config.py for details.

#     Raises:
#     - Exception: If there is an issue accessing the OpenAI API, such as an invalid API key or a network connection error, the function will raise an exception with a message indicating the problem.
#     """
#     try:
#         # model_name = model4_name if model_type == 4 else model3_name
#         # user_api_key = api_key if user_api_key is None else user_api_key

#
#         llm = Replicate(model="snowflake/snowflake-arctic-instruct")

#         template = config["null_attribute_template"]
#         prompt_template = PromptTemplate(
#             input_variables=["attributes", "types_info", "description_info"],
#             template=template,
#         )
#         summary_prompt = prompt_template.format(
#             attributes=attributes,
#             types_info=types_info,
#             description_info=description_info,
#         )
#         print(summary_prompt)
#         llm_answer = llm.invoke(summary_prompt)
#         print("LLM Answer:", llm_answer)
#         if "```json" in llm_answer.content:
#             match = re.search(r"```json\n(.*?)```", llm_answer.content, re.DOTALL)
#             if match:
#                 json_str = match.group(1)
#         else:
#             json_str = llm_answer.content
#         return json.loads(json_str)
#     except Exception as e:
#         # st.error(
#         #     "Cannot access the OpenAI API. Please check your API key or network connection."
#         # )
#         st.error(f"Error accessing the Replicate API: {e}")
#         st.stop()


# def decide_model(
#     shape_info,
#     head_info,
#     nunique_info,
#     description_info,
# ):
#     """
#     Decides the most suitable machine learning model based on dataset characteristics.

#     Parameters:
#     - shape_info (dict): Information about the shape of the dataset.
#     - head_info (str or DataFrame): The head of the dataset or its string representation.
#     - nunique_info (dict): Information about the uniqueness of dataset attributes.
#     - description_info (str): Descriptive information about the dataset.
#     - model_type (int, optional): Specifies which model to consult for decision-making.
#     - user_api_key (str, optional): OpenAI API key for making requests.

#     Returns:
#     - dict: A JSON object containing the recommended model and configuration. Please refer to prompt templates in config.py for details.

#     Raises:
#     - Exception: If there is an issue accessing the OpenAI API, such as an invalid API key or a network connection error, the function will raise an exception with a message indicating the problem.
#     """
#     try:
#         # model_name = model4_name if model_type == 4 else model3_name
#         # user_api_key = api_key if user_api_key is None else user_api_key
#         llm = Replicate(model="snowflake/snowflake-arctic-instruct")

#
#         template = config["decide_model_template"]
#         prompt_template = PromptTemplate(
#             input_variables=[
#                 "shape_info",
#                 "head_info",
#                 "nunique_info",
#                 "description_info",
#             ],
#             template=template,
#         )
#         summary_prompt = prompt_template.format(
#             shape_info=shape_info,
#             head_info=head_info,
#             nunique_info=nunique_info,
#             description_info=description_info,
#         )
#         print(summary_prompt)
#         # llm_answer = llm.generate([HumanMessage(content=summary_prompt)])
#         llm_answer = llm.invoke(summary_prompt)
#         print("LLM Answer:", llm_answer)
#         if "```json" in llm_answer.content:
#             match = re.search(r"```json\n(.*?)```", llm_answer.content, re.DOTALL)
#             if match:
#                 json_str = match.group(1)
#         else:
#             json_str = llm_answer.content
#         return json.loads(json_str)
#     except Exception as e:
#         # st.error(
#         #     "Cannot access the OpenAI API. Please check your API key or network connection."
#         # )
#         st.error(f"Error accessing the Replicate API: {e}")
#         st.stop()


# def decide_cluster_model(shape_info, description_info, cluster_info):
#     """
#     Determines the appropriate clustering model based on dataset characteristics.

#     Parameters:
#     - shape_info: Information about the dataset shape.
#     - description_info: Descriptive statistics or information about the dataset.
#     - cluster_info: Additional information relevant to clustering.
#     - model_type (int, optional): The model type to use for decision making (default 4).
#     - user_api_key (str, optional): The user's API key for OpenAI.

#     Returns:
#     - A JSON object with the recommended clustering model and parameters. Please refer to prompt templates in config.py for details.

#     Raises:
#     - Exception: If unable to access the OpenAI API or another error occurs.
#     """
#     try:
#         # model_name = model4_name if model_type == 4 else model3_name
#         # user_api_key = api_key if user_api_key is None else user_api_key
#         llm = Replicate(model="snowflake/snowflake-arctic-instruct")

#
#         template = config["decide_clustering_model_template"]
#         prompt_template = PromptTemplate(
#             input_variables=["shape_info", "description_info", "cluster_info"],
#             template=template,
#         )
#         summary_prompt = prompt_template.format(
#             shape_info=shape_info,
#             description_info=description_info,
#             cluster_info=cluster_info,
#         )
#         print(summary_prompt)
#         # llm_answer = llm.generate([HumanMessage(content=summary_prompt)])
#         llm_answer = llm.invoke(summary_prompt)
#         print("LLM Answer:", llm_answer)
#         if "```json" in llm_answer.content:
#             match = re.search(r"```json\n(.*?)```", llm_answer.content, re.DOTALL)
#             if match:
#                 json_str = match.group(1)
#         else:
#             json_str = llm_answer.content
#         return json.loads(json_str)
#     except Exception as e:
#         st.error(f"Error accessing the Replicate API: {e}")
#         st.stop()


# def decide_regression_model(shape_info, description_info, Y_name):
#     """
#     Determines the appropriate regression model based on dataset characteristics and the target variable.

#     Parameters:
#     - shape_info: Information about the dataset shape.
#     - description_info: Descriptive statistics or information about the dataset.
#     - Y_name: The name of the target variable.
#     - model_type (int, optional): The model type to use for decision making (default 4).
#     - user_api_key (str, optional): The user's API key for OpenAI.

#     Returns:
#     - A JSON object with the recommended regression model and parameters. Please refer to prompt templates in config.py for details.

#     Raises:
#     - Exception: If unable to access the OpenAI API or another error occurs.
#     """
#     try:
#         # model_name = model4_name if model_type == 4 else model3_name
#         # user_api_key = api_key if user_api_key is None else user_api_key

#
#         llm = Replicate(model="snowflake/snowflake-arctic-instruct")
#         template = config["decide_regression_model_template"]
#         prompt_template = PromptTemplate(
#             input_variables=["shape_info", "description_info", "Y_name"],
#             template=template,
#         )
#         summary_prompt = prompt_template.format(
#             shape_info=shape_info, description_info=description_info, Y_name=Y_name
#         )
#         print(summary_prompt)

#         # llm_answer = llm.generate([HumanMessage(content=summary_prompt)])

#         llm_answer = llm.invoke(summary_prompt)
#         print("LLM Answer:", llm_answer)
#         if "```json" in llm_answer.content:
#             match = re.search(r"```json\n(.*?)```", llm_answer.content, re.DOTALL)
#             if match:
#                 json_str = match.group(1)
#         else:
#             json_str = llm_answer.content
#         return json.loads(json_str)
#     except Exception as e:
#         st.error(f"Error accessing the Replicate API: {e}")
#         st.stop()


# def decide_target_attribute(attributes, types_info, head_info):
#     """
#     Determines the target attribute for modeling based on dataset attributes and characteristics.

#     Parameters:
#     - attributes: A list of dataset attributes.
#     - types_info: Information about the data types of the attributes.
#     - head_info: A snapshot of the dataset's first few rows.
#     - model_type (int, optional): The model type to use for decision making (default 4).
#     - user_api_key (str, optional): The user's API key for OpenAI.

#     Returns:
#     - The name of the recommended target attribute. Please refer to prompt templates in config.py for details.

#     Raises:
#     - Exception: If unable to access the OpenAI API or another error occurs.
#     """
#     try:
#         # model_name = model4_name if model_type == 4 else model3_name
#         # user_api_key = api_key if user_api_key is None else user_api_key

#
#         # llm = Replicate(model="snowflake/snowflake-arctic-instruct")

#         template = config["decide_target_attribute_template"]
#         prompt_template = PromptTemplate(
#             input_variables=["attributes", "types_info", "head_info"], template=template
#         )
#         summary_prompt = prompt_template.format(
#             attributes=attributes, types_info=types_info, head_info=head_info
#         )
#         print(summary_prompt)
#         input = {"prompt": summary_prompt}
#         llm_answer = replicate.run("snowflake/snowflake-arctic-instruct", input=input)
#         print("LLM Answer:", llm_answer)
#         # if "```json" in llm_answer:
#         #     match = re.search(r"```json\n(.*?)```", llm_answer, re.DOTALL)
#         #     if match:
#         #         json_str = match.group(1)

#         llm_answer = "".join(llm_answer)
#         json_str = llm_answer[llm_answer.find("{") : llm_answer.rfind("}") + 1]

#         # else:
#         #     json_str = llm_answer

#         output = json.loads(json_str)["target"]
#         print(output)
#         return output
#     except Exception as e:
#         # st.error(
#         #     "Cannot access the OpenAI API. Please check your API key or network connection."
#         # )
#         st.error(f"Error accessing the Replicate API: {e}")
#         st.stop()


# def decide_test_ratio(shape_info):
#     """
#     Determines the appropriate train-test split ratio based on dataset characteristics.

#     Parameters:
#     - shape_info: Information about the dataset shape.
#     - model_type (int, optional): The model type to use for decision making (default 4).
#     - user_api_key (str, optional): The user's API key for OpenAI.

#     Returns:
#     - The recommended train-test split ratio as a float. Please refer to prompt templates in config.py for details.

#     Raises:
#     - Exception: If unable to access the OpenAI API or another error occurs.
#     """
#     try:
#         # model_name = model4_name if model_type == 4 else model3_name
#         # user_api_key = api_key if user_api_key is None else user_api_key

#
#         llm = Replicate(model="snowflake/snowflake-arctic-instruct")

#         template = config["decide_test_ratio_template"]
#         prompt_template = PromptTemplate(
#             input_variables=["shape_info"], template=template
#         )
#         summary_prompt = prompt_template.format(shape_info=shape_info)
#         print(summary_prompt)
#         # llm_answer = llm.generate([HumanMessage(content=summary_prompt)])
#         llm_answer = llm.invoke(summary_prompt)
#         print("LLM Answer:", llm_answer)
#         if "```json" in llm_answer.content:
#             match = re.search(r"```json\n(.*?)```", llm_answer.content, re.DOTALL)
#             if match:
#                 json_str = match.group(1)
#         else:
#             json_str = llm_answer.content
#         return json.loads(json_str)["test_ratio"]
#     except Exception as e:
#         st.error(f"Error accessing the Replicate API: {e}")
#         st.stop()


# def decide_balance(shape_info, description_info, balance_info):
#     """
#     Determines the appropriate method to balance the dataset based on its characteristics.

#     Parameters:
#     - shape_info: Information about the dataset shape.
#     - description_info: Descriptive statistics or information about the dataset.
#     - balance_info: Additional information relevant to dataset balancing.
#     - model_type (int, optional): The model type to use for decision making (default 4).
#     - user_api_key (str, optional): The user's API key for OpenAI.

#     Returns:
#     - The recommended method to balance the dataset. Please refer to prompt templates in config.py for details.

#     Raises:
#     - Exception: If unable to access the OpenAI API or another error occurs.
#     """
#     try:
#         # model_name = model4_name if model_type == 4 else model3_name
#         # user_api_key = api_key if user_api_key is None else user_api_key

#
#         llm = Replicate(model="snowflake/snowflake-arctic-instruct")

#         template = config["decide_balance_template"]
#         prompt_template = PromptTemplate(
#             input_variables=["shape_info", "description_info", "balance_info"],
#             template=template,
#         )
#         summary_prompt = prompt_template.format(
#             shape_info=shape_info,
#             description_info=description_info,
#             balance_info=balance_info,
#         )
#         print(summary_prompt)
#         llm_answer = llm.run(summary_prompt)
#         print("LLM Answer:", llm_answer)
#         if "```json" in llm_answer.content:
#             match = re.search(r"```json\n(.*?)```", llm_answer.content, re.DOTALL)
#             if match:
#                 json_str = match.group(1)
#         else:
#             json_str = llm_answer.content
#         return json.loads(json_str)["method"]
#     except Exception as e:
#         st.error(f"Error accessing the Replicate API: {e}")
#         st.stop()
