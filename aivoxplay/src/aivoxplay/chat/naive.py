import operator
import os
from collections import deque
from typing import Annotated
from typing_extensions import TypedDict
import asyncio
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph
from langchain_openai import OpenAI, ChatOpenAI
import time
import httpx
from dotenv import load_dotenv

# Load environment variables from .env file (for OpenAI API key, etc.)
load_dotenv()
env_openai_key = os.getenv("OPENAI_API_KEY")

# Initialize the Chat LLM (GPT-4 or similar) for structured outputs
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.5, openai_api_key=env_openai_key)

async def external_api_call(query: str) -> str:
    """
    Async call to an external chatbot API or knowledge base.
    This is a placeholder that returns a static response.
    In practice, it should query the external system and return its answer.
    """
    # Simulate a delay for demonstration (if needed, e.g., await asyncio.sleep(3))
    return "Hello, how can I help you today?"



# Define the schemas for structured output from the LLM
class InputClassification(BaseModel):
    casual: bool = Field(
        description="True if the user input is casual/small talk (no external lookup needed). "
                    "False if it requires external processing (IT/HR/Finance queries, etc.)."
    )
    query: str = Field(
        description="The exact user query to ask an external agent if required (may be same as input or rephrased)."
    )

class AudioOutput(BaseModel):
    response: str = Field(
        description="Agent's response in natural language (as if speaking to the user). "
                    "Should be conversational, polite, and concise (<=40 words). "
                    "Avoid symbols except punctuation. Use `..` sparingly for a human-like pause."
    )

# Define the overall state for the conversation, using a reducer for message history
class State(TypedDict):
    pending_inputs: deque[str]
    user_input: str
    route_external: bool
    external_agent_output: str
    # 'aggregate' serves as conversation history, appending new messages as they occur
    aggregate: Annotated[list[dict], operator.add]

class AIVoxChat:
    def __init__(self, external_chat_fn, agent_output=deque(), on_agent_response=None):
        """
        Initialize the chat orchestrator.
        :param external_chat_fn: function to call for external queries (async).
        :param on_agent_response: callback to execute whenever the agent has a new message to send.
        """
        self.external_chat_fn = external_chat_fn
        self.on_agent_response = on_agent_response
        self.agent_output = agent_output
        # Initialize conversation state with empty history
        self.state: State = {
            "pending_inputs": deque(),
            "user_input": "",
            "route_external": False,
            "external_agent_output": "",
            "aggregate": []  # Will hold dicts like {"user": "..."} or {"agent": "..."}
        }
        # Build the state graph for orchestrating the conversation
        self.compiled_graph = self.build_graph()
        # This list will collect agent outputs in order (for logging or testing)
        self.agent_outputs = agent_output

    def build_graph(self):
        gb = StateGraph(State)
        gb.add_node("input", self.input_node)
        gb.add_node("external", self.external_node)
        gb.add_node("output", self.output_node)
        gb.add_edge(START, "input")
        # Conditional transition: go to 'external' node if external route needed, otherwise go to 'output'
        gb.add_conditional_edges("input", lambda state: "external" if state.get("route_external") else "output",
                                 {"external": "external", "output": "output"})
        gb.add_edge("external", "output")
        gb.add_edge("output", END)
        return gb.compile()

    async def input_node(self, state: State):
        """
        Processes the latest user input:
        - Classify if it's casual or requires external processing.
        - Prepare the query for external call if needed.
        """
        if state["user_input"] == "":
            # No new input to process
            return {}
        query = state["user_input"]
        # Use LLM to classify the input and possibly rephrase for external agent
        structured_llm = llm.with_structured_output(InputClassification)
        classification = await structured_llm.ainvoke(query)
        print("InputNode Classification:", classification.dict())
        # Determine routing: if not casual (e.g., work-related query), route to external
        route_ext = not classification.casual
        # Update conversation history with the user message
        # Use the original user input for history (preserve the user's actual words)
        state_update = {
            "user_input": classification.query if route_ext else query,
            "route_external": route_ext,
            "aggregate": [ {"user": query} ]
        }
        return state_update

    async def external_node(self, state: State):
        """
        Handles queries that require external processing:
        - Immediately send a short wait message to the user.
        - Perform the external API call asynchronously.
        - Provide periodic "still working" updates if the call is slow.
        - Once the external result is obtained, update state with the result.
        """
        if not state.get("route_external"):
            # Not an external route scenario, skip this node
            return {}
        user_query = state["user_input"]
        print(f"ExternalNode: Routing query externally -> '{user_query}'")

        # 1. Immediately inform the user to wait (short polite message)
        structured_llm = llm.with_structured_output(AudioOutput)
        conv_history_text = self._format_history(state["aggregate"])
        # Ask LLM to produce a brief waiting acknowledgement in context
        wait_prompt = f"{conv_history_text}\nAgent: (acknowledge and ask the user to please wait briefly in less than 10 words)"
        wait_output = await structured_llm.ainvoke(wait_prompt)
        wait_message = wait_output.response.strip()
        # Ensure brevity (if the model returns a long text, truncate or shorten)
        if len(wait_message.split()) > 10:  # if more than ~6 words, shorten it
            wait_message = "One moment please.."  # fallback short message
        # Add the wait message to conversation history and output it
        state["aggregate"].append({"agent": wait_message})
        self.agent_outputs.append(wait_message)
        if self.on_agent_response:
            self.on_agent_response()  # trigger external side effect (e.g., TTS or UI update) for wait message
        print("ExternalNode Wait Message to user:", wait_message)

        # 2. Initiate external API call asynchronously
        external_task = asyncio.create_task(self.external_chat_fn(user_query))

        # 3. Periodically check if the external call is done, send updates if not
        update_messages = [
            "Just a moment..", 
            "Still checking on that..", 
            "Thanks for your patience.."
        ]
        update_index = 0
        total_wait = 0
        max_updates = len(update_messages)
        # Loop while waiting for the external task, sending up to max_updates interim messages
        while not external_task.done() and update_index < max_updates:
            await asyncio.sleep(5)  # wait 5 seconds before each check
            total_wait += 5
            if external_task.done():
                break
            # If still not done, send a periodic update
            update_msg = update_messages[update_index]
            state["aggregate"].append({"agent": update_msg})
            self.agent_outputs.append(update_msg)
            if self.on_agent_response:
                self.on_agent_response()
            print(f"ExternalNode Update ({total_wait}s):", update_msg)
            update_index += 1

        # 4. Wait for the external task to finish (if it hasn't already)
        external_reply: str = await external_task  # get the result of external call
        if external_reply is None:
            external_reply = ""
        print("ExternalNode: External agent reply received.")
        # Update the state with the external agent's raw output (for context)
        # Store it under a distinct key in history (not directly shown to user)
        state_update = {
            "external_agent_output": external_reply,
            "aggregate": [ {"external": external_reply} ]
        }
        return state_update

    async def output_node(self, state: State):
        """
        Generates the final agent response to the user:
        - If an external result was obtained, incorporate it into the answer.
        - If no external call was needed, answer directly using the conversation context.
        """
        user_query = state["user_input"]
        ext_answer = state.get("external_agent_output", "")  # external answer if present
        # Prepare conversation context for the LLM prompt
        conv_history_text = self._format_history(state["aggregate"])
        structured_llm = llm.with_structured_output(AudioOutput)

        # Construct the prompt for the final answer
        if state.get("route_external"):
            # If external data was used, include it in the prompt for context
            prompt = (f"{conv_history_text}\nExternal info: {ext_answer}\n"
                      f"Agent: (use the external info above to answer the user's query)")
        else:
            prompt = f"{conv_history_text}\nAgent:"  # Agent should continue the conversation based on context

        final_output = await structured_llm.ainvoke(prompt)
        agent_answer = final_output.response.strip()
        print("OutputNode Final Agent Answer:", agent_answer)

        # Update conversation history with the final agent answer
        state_update = {"aggregate": [ {"agent": agent_answer} ]}
        # Log and output the final agent answer
        self.agent_outputs.append(agent_answer)
        if self.on_agent_response:
            self.on_agent_response()
        # (The external agent's raw output is not directly shown to the user; it's used for context only)
        return state_update

    def _format_history(self, history: list[dict]) -> str:
        """
        Helper to format the conversation history for prompts.
        Each entry in history is a dict with keys "user" or "agent" (and possibly "external").
        We will format them as dialogue lines, skipping any internal-only entries.
        """
        lines = []
        for entry in history:
            if "user" in entry:
                lines.append(f"User: {entry['user']}")
            elif "agent" in entry:
                lines.append(f"Agent: {entry['agent']}")
            elif "external" in entry:
                # This is external data (not a user or agent utterance). 
                # We include it in context labeling it as external information.
                lines.append(f"(External system provided info: {entry['external']})")
        return "\n".join(lines)

    async def listen(self, msg: str):
        """
        Processes a new user message (async). 
        This triggers the state graph to handle the message and generate responses.
        """
        # Append the new user input to pending queue and state, preserving prior history
        self.state["pending_inputs"].append(msg)
        self.state["user_input"] = msg
        self.state["external_agent_output"] = ""  # reset external output for new message
        # Run one pass through the state graph for this input
        await self.compiled_graph.ainvoke(self.state)
        # After processing, clear pending_inputs for next message
        self.state["pending_inputs"].clear()

# Example usage and testing
async def main():
    chat = AIVoxChat(external_chat_fn=external_api_call)
    # Function to simulate sending agent outputs (on_agent_response callback)
    chat.on_agent_response = lambda: print("(Agent output sent)")  # simple print callback for demo

    # Simulate a conversation:
    print("User: Hello!")
    await chat.listen("Hello!")
    # Agent should respond without external call (casual greeting)
    await asyncio.sleep(0.5)  # small delay to allow processing

    print("User: Is there an update on the IT ticket I created recently?")
    await chat.listen("Is there an update on the IT ticket I created recently?")
    # Agent will likely route externally (non-casual), send wait message, then final answer after external dummy reply.
    await asyncio.sleep(6)  # wait to observe at least one periodic update in demo

    print("User: What's the ticket number?")
    await chat.listen("What's the ticket number?")
    # Depending on context, agent might ask for clarification or route externally again.
    await asyncio.sleep(1)

    # Print the collected agent outputs in order
    print("\nConversation log (Agent outputs):")
    for resp in chat.agent_outputs:
        print(f"Agent: {resp}")

async def consumer(chat: AIVoxChat, queue: asyncio.Queue[str], stop_word: str = "Bye"):
    """
    Pulls msgs off the queue, dispatching each to process_message.
    Stops when it pulls stop_word.
    """
    while True:
        msg = await queue.get()
        # dispatch concurrently
        asyncio.create_task(chat.listen(msg))
        if msg == stop_word:
            break

# Run the main function if executing this script directly
if __name__ == "__main__":
    asyncio.run(main())
