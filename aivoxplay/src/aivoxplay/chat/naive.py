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
# Load environment variables from .env file
load_dotenv()

# Load configuration from environment
env_openai_key = os.getenv("OPENAI_API_KEY")

speech_order = []
input_queue: asyncio.Queue[str] = asyncio.Queue()
llm = ChatOpenAI(
    model_name="gpt-4o", 
    temperature=0.5, 
    openai_api_key=env_openai_key
)

async def external_api_call(query: str) -> str:
    """
    Async call to external chatbot API.
    """
    return "Hello, how can I help you today?"
# ─── 1) Define your State schema ───────────────────────────────────────────────
class State(TypedDict):
    pending_inputs: deque[str]
    user_input: str
    # new flag: should this go through the external node?
    route_external: bool
    # the external node’s eventual output
    external_agent_output: str
    # for collecting what each node emits
    aggregate: Annotated[list, operator.add]

class InputClassification(BaseModel):
    casual: bool = Field(description="Only if the user input is casual, it should not be routed to an external agent. If the user input is casual, it should be processed by the internal agent. If the user input is not casual, such as IT, HR, Finance, it should be routed to an external agent for further processing. True only if it is small talk or casual conversation about life.")
    query: str = Field(description="exact user query to ask an external agent if requierd.")

class AudioOutput(BaseModel):
    response: str = Field(description="Imagine you are talking to a user over phone. Generate a response to the user's query in simple natural language for TTS synthesis. You are talking to the user. focus on clarity and simplicity. response should be based on latest response and recent conv history and question. You are a proxy to an enterprise support agent. Pretend to be the support agent and chat with the user. Keep your answers free flowing with less than 40 words unless asked to elaborate. Only use the following to sound more human like `..`. Do not use any symbols except `.,!?`. Other sysmbols are strictly restricted.. Do not use links or hyperlinks. Remeber you are speaking to a user.")

# ─── 2) Node implementations ───────────────────────────────────────────────────
    
class AIVoxChat:
    def __init__(self, external_chat_fn, agent_output= deque()):
        self.state = {
            "pending_inputs": deque(),
            "user_input": "",
            "route_external": False,
            "external_agent_output": "",
            "aggregate": [],
        }
        self.compiled = None
        self.external_chat_fn = external_chat_fn
        self.agent_output = agent_output
        self.build_graph()

    async def input_node(self, state: State):

        if not state["pending_inputs"] and  state["user_input"]:
            return {}
        # If input is provided, add it to the pending inputs
        if state['user_input']:
            query = state['user_input']
        #query = state["pending_inputs"].popleft()
        # decide whether to route to external
        structured_llm = llm.with_structured_output(InputClassification)
        output = structured_llm.invoke(query)
        print("From Input Node:", output)

        route = (output.casual == False)  # route to external if not casual
        return {
            "user_input":       query if output.casual else output.query,
            "route_external":   route,
            "aggregate":       [{"user_input": query, "route_external": route}],
        }

    async def external_node(self, state: State):
        """Simulate an external agent response based on the latest user_input."""

        if not state["route_external"]:
            return {}
        ui = state["user_input"]
        print("External Node Input: ", ui)

        structured_llm = llm.with_structured_output(AudioOutput)
        convHistroy = " ".join(str(s) for s in state["aggregate"]) + ". User Query " + ui
        output = structured_llm.invoke("Ask the user to wait in natural langugage with context." + convHistroy)
        self.agent_output.append(output.response)
        reply = await self.external_chat_fn(ui)  # call the external API

        
        return {
            "external_agent_output": reply,
            "aggregate": [{"agent": output.response},{"external_agent_output": reply}],
        }

    async def output_node(self, state: State):
        ui = state["user_input"]
        ea = state.get("external_agent_output", "")
        query = ea if ea else ui
        print("Query for output node: ", query)
        structured_llm = llm.with_structured_output(AudioOutput)
        convHistroy = " ".join(str(s) for s in state["aggregate"])
        # if we *should* go external but haven’t gotten a reply yet:
        if state["route_external"] and not ea:
            output = structured_llm.invoke(convHistroy + ". User Query " + query)
            result = f"▶ Final output → agent: “{output.response}” | external: “{ea}”"
        else:
            output = structured_llm.invoke(convHistroy + ". External Agent Response -  " + query)
            result = f"▶ Final output → agent: “{output.response}” | external: “{ea}”"
        
        # print("Aggregate: ", state["aggregate"])
        print("Agent Response: ", output.response)
        speech_order.append(output.response)
        self.agent_output.append(output.response)
        
        return { "aggregate": [result]}

# ─── 3) Build & compile the graph ──────────────────────────────────────────────
    
    def build_graph(self):
        gb = StateGraph(State)
        gb.add_node("input", self.input_node)
        gb.add_node("external", self.external_node)
        gb.add_node("output", self.output_node)
        gb.add_edge(START, "input")

        def route(state: State):
            return "external" if state.get("route_external") else "output"

        gb.add_conditional_edges(
            "input",
            route,
            {"external": "external", "output": "output"}
        )

        gb.add_edge("external", "output")
        gb.add_edge("output", END)

        self.compiled = gb.compile()

# ─── 4) Per‐message processing task ────────────────────────────────────────────
    async def listen(self, msg: str):
        # Each message gets its own fresh state,
        # so they can all run concurrently.
        state: State = {
            "pending_inputs": deque([msg]),
            "user_input": msg,
            "external_agent_output": "",
            "aggregate": [],
        }
        # run exactly one pass through the graph
        await self.compiled.ainvoke(state)


# ─── 6) Async Expeiments ─────────────────────────────────────────────────

# — helper: consume messages and dispatch —
async def consumer(chat: AIVoxChat, queue: asyncio.Queue[str], stop_word: str = "Bye"):
    """
    Pulls msgs off the queue, dispatching each to process_message.
    Stops when it pulls `stop_word`.
    """
    while True:
        msg = await queue.get()
        # dispatch concurrently
        asyncio.create_task(chat.listen(msg))
        if msg == stop_word:
            break

async def main():
    q: asyncio.Queue[str] = asyncio.Queue()
    chat = AIVoxChat(external_chat_fn=external_api_call)
      # build the state graph
    # start the consumer
    consumer_task = asyncio.create_task(consumer(chat, q, stop_word="Bye"))

    # --- anywhere in your code, you can now push messages: ---
    await q.put("Hello!")
    await asyncio.sleep(0.6)
    #await q.put("How are you today?")
    await asyncio.sleep(0.1)
   # print(speech_order)
    # ... e.g. user input, websocket callbacks, etc. ...
    #await q.put("Bye")
    await q.put("Is there an update on the ticket IT ticket I created recently?")
    await asyncio.sleep(1.5)
    await q.put("Whats the ticket number?")
    await asyncio.sleep(1.5)
    #await q.put("Bye")
    # wait for the consumer to see "Bye" and finish
    await consumer_task

if __name__ == "__main__":
    asyncio.run(main())
    print("Speech order after processing all messages:")
    print(speech_order)

