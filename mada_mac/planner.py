from mlx_lm import load, generate
import ollama
import yaml
import driver_agent
from functions_schema import gather_functions_schema, FunctionCallParser
import os


# TODO: think a two step process that needs the LLM to 1) call a function and
#  2a) use the result to call another function
#  or 2b) ask the driver to provide more info


mada_file = "mada.yaml"
with open(mada_file) as file:
    mada_config_dict = yaml.load(file, Loader=yaml.SafeLoader)

class Planner:
    def __init__(self):

        print("\n<<<<<<<<<<<< Starting Planner >>>>>>>>>>>>")

        self.llm_type = mada_config_dict.get("llm_type", "mlx")

        if self.llm_type == "mlx":
            mlx_llm = mada_config_dict.get("mlx_llm", "mlx-community/gemma-2-2b-it-8bit")
            self.llm_model, self.llm_tokenizer = load(mlx_llm)
            # Avoid a HuggingFace tokenizer warning: "huggingface/tokenizers: The current process just got forked ...
            # ... Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        functions_list = ["check_safety_distance_from_vehicle", "get_current_speed"]

        self.functions_schemas = gather_functions_schema(driver_agent, functions_list)
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
