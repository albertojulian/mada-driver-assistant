from memory import get_memory
# from planner import Planner
from mlx_lm import load, generate
import ollama
from functions_schema import gather_functions_schema, FunctionCallParser
import os
from ast import literal_eval
import yaml
from time import time
import functions
from functions import *
import sys
sys.path.append("../common")
from text_to_speech import text_to_speech


# Executing a function inside a string: 'check_safety_distance_from_vehicle_v0(vehicle_type="car", position="in front")'
# - Use eval() to execute the function if it's a simple expression,
# - or exec() if it's a statement or requires more flexibility.

mada_file = "driver_agent.yaml"
with open(mada_file) as file:
    mada_config_dict = yaml.load(file, Loader=yaml.SafeLoader)


class DriverAgent:

    def __init__(self):

        print("\n<<<<<<<<<<<< Starting Driver Agent >>>>>>>>>>>>")

        self.memory = get_memory()
        self.planner = Planner()

    def evaluate_automatic_action(self, params_str, log=False):
        """

        :param params_str
        :param log
        :return:
        """

        params = literal_eval(params_str)
        # params = eval(params_str)

        # object, _ = self.memory.manage_space_event(params, log=log)
        self.memory.manage_space_event(params, log=log)

    def evaluate_action_from_request(self, text_input_message, log=True):

        text_input_message_event = self.memory.add_text_input_message(text_input_message, log=log)

        start_time = time()
        parsing_ok, response = self.planner.generate_response(text_input_message)
        processing_time = round(time() - start_time, 1)
        text_input_message_event.processing_time = processing_time
        print(f"LLM Response took {processing_time} seconds")

        if parsing_ok:
            eval(response)
        else:
            text_to_speech(response)


_driver_agent_instance = None


def get_driver_agent(log=False):
    global _driver_agent_instance
    if _driver_agent_instance is None:
        _driver_agent_instance = DriverAgent()
    else:
        if log:
            print("Driver Agent already exists")

    return _driver_agent_instance


class Planner:

    # TODO: think a two step process that needs the LLM to 1) call a function and
    #  2a) use the result to call another function
    #  or 2b) ask the driver to provide more info

    def __init__(self):

        print("\n<<<<<<<<<<<< Starting Planner >>>>>>>>>>>>")

        self.llm_type = mada_config_dict.get("llm_type", "mlx")

        if self.llm_type == "mlx":
            mlx_llm = mada_config_dict.get("mlx_llm", "mlx-community/gemma-2-2b-it-8bit")
            self.llm_model, self.llm_tokenizer = load(mlx_llm)
            # Avoid a HuggingFace tokenizer warning: "huggingface/tokenizers: The current process just got forked ...
            # ... Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # TODO: conf file
        # functions_list = ["check_safety_distance_from_vehicle", "get_current_speed"]
        functions_list = ["check_safety_distance_from_vehicle", "get_current_speed", "get_last_speed_limit_sign"]

        self.functions_schemas = gather_functions_schema(functions, functions_list)
        functions_schemas_str = str(self.functions_schemas)

        self.prompt_ini = """
            You are a helpful car driver assistant that takes a question and finds the most appropriate function to execute, along with the parameters required to run the function.
            Respond as JSON using the following schema: {"function_name": "function name", "parameters": [{"parameter_name": "name of parameter", "parameter_value": "value of parameter"}]}.
            The definition of functions and parameters is: """ + functions_schemas_str + ". "

        # If a parameter definition has the key "allowed_values", the value of the "parameter_value" key should be the allowed value most consistent with the question.


    def generate_response(self, input_message):

        prompt = self.prompt_ini + input_message
        message = [{'role': 'user', 'content': prompt}]

        if self.llm_type == "mlx":
            chat_prompt = self.llm_tokenizer.apply_chat_template(message, tokenize=False)
            llm_response = generate(self.llm_model, self.llm_tokenizer, prompt=chat_prompt, verbose=False)
        else:
            # ollama
            ollama_llm = mada_config_dict.get("ollama_llm", "gemma2:2b")
            llm_response = ollama.chat(model=ollama_llm, messages=message)
            llm_response = llm_response['message']['content']

        print("\n<<<<<<<<<<< Start of LLM Response: ")
        print(llm_response)
        print(">>>>>>>>>>> End of LLM Response\n")

        function_call_parser = FunctionCallParser(self.functions_schemas, llm_response)
        # TODO: si no hay función, contesta a lo loco
        parsing_ok, function_call_str = function_call_parser.parse_function_call()

        return parsing_ok, function_call_str


if __name__ == "__main__":
    planner = Planner()
    print(planner.functions_schemas)