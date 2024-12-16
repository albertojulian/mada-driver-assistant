from time import time
from functions_langgraph import *
from PIL import Image
from io import BytesIO
from typing_extensions import TypedDict
from typing import Annotated, Literal
from pprint import pprint
from ast import literal_eval

import sys; sys.path.append("../common")
from text_to_speech import text_to_speech, text_to_speech_async

from langchain_ollama import ChatOllama

from langchain_core.messages.ai import AIMessage

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

mada_file = "../mada.yaml"
with open(mada_file) as file:
    mada_config_dict = yaml.load(file, Loader=yaml.SafeLoader)

class DriverAgent:

    def __init__(self):

        print("\n<<<<<<<<<<<< Starting Driver Agent >>>>>>>>>>>>")

        self.memory = get_memory()
        self.driver_agent_graph = None
        self.listen_mode = False

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

    def start_driver_agent_graph(self):
        self.driver_agent_graph = DriverAgentGraph()
        welcome_message = self.driver_agent_graph.LISTENING_ACK

        # text_to_speech(welcome_message + "\n")

    def evaluate_action_from_request(self, text_request, log=False):

        if self.driver_agent_graph is not None:
            text_request_event = self.memory.add_text_request(text_request, log=log)

            start_time = time()
            response = self.driver_agent_graph.generate_response(text_request)
            processing_time = round(time() - start_time, 1)
            text_request_event.processing_time = processing_time
            print(f"Driver Agent response took {processing_time} seconds")

            # no need to print the response, the driver agent graph already prints it with the rest of messages
            text_to_speech(response, print_message=False)

    def remove_driver_agent_graph(self):
        del self.driver_agent_graph


_driver_agent_instance = None


def get_driver_agent(log=False):
    global _driver_agent_instance
    if _driver_agent_instance is None:
        _driver_agent_instance = DriverAgent()
    else:
        if log:
            print("Driver Agent already exists")

    return _driver_agent_instance


class RequestState(TypedDict):
    """State representing the driver's requests."""

    # The chat conversation. This preserves the conversation history
    # between nodes. The `add_messages` annotation indicates to LangGraph
    # that state is updated by appending returned messages, not replacing
    # them.
    messages: Annotated[list, add_messages]

    # Flag indicating that the driver wants to finish the session.
    finished: bool


class DriverAgentGraph:
    """
    Based on LangGraph
    """
    def __init__(self):

        driver_agent_conf = mada_config_dict["driver_agent"]
        tool_names = driver_agent_conf["tools"]
        tools = [eval(tool_name) for tool_name in tool_names]

        self.DRIVER_AGENT_SYS_INST = (
            "system",  # 'system' indicates the message is a system instruction.
            f"""
            You are Driver Agent, an interactive driver assistant system. The driver will ask questions.
            You have the following functions to answer the question: {tool_names}
            \n\n
            The answer should be as brief as possible.
            """,
        )

        # This is the message which the system starts the conversation with.
        communications_config = mada_config_dict["communications"]
        self.LISTENING_ACK = communications_config.get("listening_ack", "Tell me")

        self.llm_type = driver_agent_conf["llm_type"]
        if self.llm_type == "ollama":
            self.ollama_llm = driver_agent_conf["ollama_llm"]

        self.tool_node = ToolNode(tools)

        self.llm = ChatOllama(model=self.ollama_llm).bind_tools(tools=tools)

        graph_builder = StateGraph(RequestState)

        # Add the nodes
        graph_builder.add_node("driver agent", self.driver_agent_with_tools)
        graph_builder.add_node("tools", self.tool_node)

        # Add the edges
        graph_builder.add_edge(START, "driver agent")
        # Tools always route back to chat.
        graph_builder.add_edge("tools", "driver agent")

        # Add the conditional edges
        graph_builder.add_conditional_edges("driver agent", self.maybe_route_to_tools)

        memory = MemorySaver()
        self.driver_agent_graph = graph_builder.compile(checkpointer=memory)

        self.driver_agent_outputs = []

    #####################
    ###### NODES ########
    #####################
    def driver_agent_with_tools(self, state : RequestState) -> RequestState:
        """The driver agent with tools. A simple wrapper around the model's own chat interface."""
        defaults = {"finished": False}

        if state["messages"]:
            new_output = self.llm.invoke([self.DRIVER_AGENT_SYS_INST] + state["messages"])

        else:
            # first time, only welcome answer to "listen" from driver
            new_output = AIMessage(content=self.LISTENING_ACK)

        # Set up some defaults if not already set, then pass through the provided state,
        # overriding only the "messages" field.
        state = defaults | state | {"messages": [new_output]}
        last_msg = state["messages"][-1]
        output_message = last_msg.content
        print(f"Driver Agent: {output_message}")

        if output_message:
            self.driver_agent_outputs.append(output_message)

        else:
            print("(empty llm response)")

        return state

    #####################
    ###### EDGES ########
    #####################
    def maybe_route_to_tools(self, state : RequestState) -> Literal["__end__", "tools"]:
        """Route between chat and tool nodes if a tool call is made."""
        if not (msgs := state.get("messages", [])):
            raise ValueError(f"No messages found when parsing state: {state}")

        msg = msgs[-1]

        if state.get("finished", False):
            # When the request is satisfied, exit the app.
            return END

        elif hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
            print(f"The selected tool is {msg.tool_calls[0]['name']}")
            # Route to `tools` node for any automated tool calls first.
            if any(
                    tool["name"] in self.tool_node.tools_by_name.keys() for tool in msg.tool_calls
            ):
                return "tools"
            else:
                print(f"No tool recognised from msg.tool_calls {msg.tool_calls}")
                return END

        else:
            return END

    def draw_langgraph(self):
        image_bytes = self.driver_agent_graph.get_graph().draw_mermaid_png()
        image = Image.open(BytesIO(image_bytes))
        image.show()


    def generate_response(self, text_request):

        print(f"<<<<<< Generate LangGraph Driver Agent response <<<<<<")

        text_request_dict = {"messages": [("user", text_request)]}

        # Thread
        thread = {"configurable": {"thread_id": "1"}}

        # Run the graph until the first interruption
        for event in self.driver_agent_graph.stream(text_request_dict, thread, stream_mode="values"):
            # somehow, Driver Agent node (driver_agent_with_tools function) is called and output is generated
            # print(event)
            pass

        print(f">>>>>>> End of LangGraph Driver Agent response >>>>>>")

        # like calling state["messages"][-1]
        response = self.driver_agent_outputs[-1]

        return response

if __name__ == "__main__":

    driver_agent_graph = DriverAgentGraph()
    driver_agent_graph.draw_langgraph()
