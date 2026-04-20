import os
import json
from src.agent_backend.crew import AgentBackend

inputs = {"user_input": "a simple express js as backend and react as frontend", "input_source": "text"}
try:
    res = AgentBackend().crew().kickoff(inputs=inputs)
    print("SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
