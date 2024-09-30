import json
import inspect
from typing import get_origin, get_args, Literal
import functions


##########################################################################
# Functions to generate schemas of functions to input them as a string inside the llm prompt
##########################################################################
def gather_functions_schema(module, functions_list):
    # functions_schema = {}
    functions_schema = []

    functions = inspect.getmembers(module, inspect.isfunction)

    # Para cada función, extraer su esquema
    for func_name, func in functions:
        if func_name in functions_list:
            schema = generate_function_schema(func)
            # functions_schema[f"{func_name}_schema"] = schema
            functions_schema.append(schema)

    return functions_schema


def generate_function_schema(func):

    signature = inspect.signature(func)
    parameters = []
    for param in signature.parameters.values():
        # filter parameters with default values
        if isinstance(param.default, type):
            param_schema = {"parameter_name": param.name}
            param_type = param.annotation
            if param_type is not inspect.Parameter.empty:
                if get_origin(param_type) is Literal:
                    # Obtener los valores permitidos del Literal
                    allowed_values = get_args(param_type)
                    param_schema["allowed_values"] = list(allowed_values)
                else:
                    # Si no es Literal, solo almacenar el tipo de parámetro
                    param_schema["type"] = str(param_type)

            parameters.append(param_schema)

    schema = {
        "function_name": func.__name__,
        "parameters": parameters
    }

    return schema


# The schema of the functions in functions_list is automatically generated
# functions_schemas = gather_functions_schema(functions, functions_list)
# functions_schemas_str = str(functions_schemas)
# print(f"functions_schemas_str {functions_schemas_str}")

#########################################################################
# Class to parse a function call from the llm response
##########################################################################

class FunctionCallParser:

    def __init__(self, functions_schemas, llm_response):
        self.llm_response = llm_response
        self.functions_schemas = functions_schemas
        self.llm_function_name = None

    def parse_function_call(self):

        start_index = self.llm_response.rfind('{"function_name"') # look for (last) {"function_name", which is the beginning of (last) json object
        end_index = self.llm_response.rfind('}')  # look for last '}', which is the end of (last) json object

        if start_index == -1 or end_index == -1:
            error_message = "There is no json string in llm response"
            print(error_message)
            return False, error_message

        json_str = self.llm_response[start_index:end_index + 1]
        print("\n<<<<<<<<<<< Start of JSON string: ")
        print(json_str)
        print(">>>>>>>>>>> End of JSON string\n")

        parsed_json = json.loads(json_str)

        self.llm_function_name = parsed_json.get('function_name', None)

        parsing_ok, status_message = self.check_function_parsing_ok(parsed_json)

        if not parsing_ok:
            return False, status_message
        else:
            llm_parameters = parsed_json['parameters']
            # Construct the function call string
            llm_param_str = ', '.join([f'{llm_param["parameter_name"]}="{llm_param["parameter_value"]}"' for llm_param in llm_parameters])

            function_call_str = f'{self.llm_function_name}({llm_param_str})'

            return True, function_call_str

    def check_function_parsing_ok(self, parsed_json):

        if self.llm_function_name is None:
            return False, "No function name has been parsed"

        function_schema = self.get_function_schema()
        if function_schema is None:
            return False, f"There is no function '{self.llm_function_name}'"

        # Extract parameters
        llm_parameters = parsed_json.get('parameters', None)
        if llm_parameters is None:
            return False, f"No parameters have been parsed in function '{self.llm_function_name}'"

        parameters = function_schema["parameters"]
        if len(llm_parameters) != len(parameters):
            return False, f"Function '{self.llm_function_name}' has {len(parameters)} parameters, but {len(llm_parameters)} have been parsed"

        for llm_param in llm_parameters:
            llm_param_name = llm_param.get("parameter_name", None)
            if llm_param_name is None:
                return False, f"At least one parameter name has no key 'parameter_name' in function '{self.llm_function_name}'"

            param_schema = self.get_parameter_schema(parameters, llm_param_name)
            if param_schema is None:
                return False, f"There is no parameter '{llm_param_name}' in function '{self.llm_function_name}'"

            llm_param_value = llm_param.get("parameter_value", None)
            if llm_param_value is None:
                return False, f"No value has been provided for parameter '{llm_param_name}' in function '{self.llm_function_name}'"

            param_allowed_values = param_schema.get("allowed_values", None)
            if param_allowed_values is not None:
                if llm_param_value not in param_allowed_values:
                    return False, f"Value '{llm_param_value}' is not allowed for parameter '{llm_param_name}' in function '{self.llm_function_name}'"

        return True, ""

    def get_function_schema(self):

        for function_schema in self.functions_schemas:
            if function_schema["function_name"] == self.llm_function_name:
                return function_schema

        return None


    def get_parameter_schema(self, parameters, llm_param_name):
        for param_schema in parameters:
            if param_schema["parameter_name"] == llm_param_name:
                return param_schema

        return None


def main1():
    functions_list = ["check_safety_distance_from_vehicle", "get_current_speed"]

    functions_schema = gather_functions_schema(functions, functions_list)
    print("\n<<<<<<<<<<< Functions schemas: ")
    print(functions_schema)
    print(">>>>>>>>>>>>>> End of Functions schemas \n")

    return functions_schema


def main2():
    from functions import check_safety_distance_from_vehicle  # can be used when executing eval(function_call_str)

    functions_schema = main1()

    llm_response = """
        * **Question:** Is distance safe?

    **Response:**
    ```json
    {"function_name": "check_safety_distance_from_vehicle", "parameters": []}
    ``` 
    <end_of_turn>
        """

    function_call_parser = FunctionCallParser(functions_schema, llm_response)
    parsing_ok, function_call_str = function_call_parser.parse_function_call()

    if parsing_ok:
        eval(function_call_str)
    else:
        print(function_call_str)


if __name__ == "__main__":
    # functions_schema = main1()
    main2()
