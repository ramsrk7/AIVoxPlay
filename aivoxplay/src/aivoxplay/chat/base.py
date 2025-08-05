# ─── base_graph_chat.py ───────────────────────────────────────────────────────
import asyncio
from collections import deque
from typing_extensions import TypedDict

class BaseGraphChat:
    """
    Ultra-light wrapper around langgraph.StateGraph.

    Sub-class contract
    ------------------
    • Declare an inner `State` TypedDict (or import one).
    • Implement:
        - build_graph(self)             → sets self.compiled
        - init_state(self, msg: str)    → returns a fresh State for that msg
    • Inside *any* node, append final user-visible text to  `self.agent_output`.

    Helpers provided
    ----------------
    • self.agent_output : deque[str]    rolling transcript
    • await self.listen("hi")           run one message through one graph pass
    """

    def __init__(self, agent_output: deque[str] = deque()):
        self.agent_output = agent_output
        self.compiled = None           # subclasses must set this
        self.build_graph()             # <- subclass implementation

    # ------------- convenience API ------------------------------------------
    async def listen(self, msg: str):
        """Invoke exactly one pass of the compiled graph for a single message."""
        if self.compiled is None:
            raise RuntimeError("build_graph() did not set `self.compiled`")
        state = self.init_state(msg)    # delegate to subclass
        await self.compiled.ainvoke(state)

    # ------------- hooks every subclass must supply -------------------------
    def build_graph(self):        
        raise NotImplementedError

    def init_state(self, msg):    
        raise NotImplementedError
