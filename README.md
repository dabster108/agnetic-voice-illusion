agentic-voice-illusion

    Agentic Voice Illusion is an AI-powered interactive tool that combines voice input with drawing capabilities to create and manipulate diagrams on a canvas. Users can speak commands to draw shapes, add text, and build visual ideas, making diagram creation faster and more intuitive.

    This project is built to demonstrate the integration of voice-based agents with a visual drawing interface, enabling a seamless experience for generating diagrams through natural language. It focuses on simplicity, responsiveness, and extensibility for future AI-driven features.

#cd /Users/dikshanta/Documents/agent-excalidraw/agent-backend

# one-time: install uv if you don't have it

pip install uv

# install project deps (first run)

uv sync

# set your env vars (required key at minimum)

# add GROQ_API_KEY=... in .env

# start backend API server

uv run agent_backend
