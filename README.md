Agentic Voice Illusion

Agentic Voice Illusion is an AI-assisted diagramming system that lets users describe ideas in natural language and see them rendered as editable diagrams on a canvas. The project connects a multi-agent backend with a React-based frontend and an Excalidraw MCP (Model Context Protocol) server to translate user intent into structured drawing operations. The goal is to keep the experience fast, visual, and extensible while maintaining clear separation between orchestration, UI, and drawing execution.

This repository is organized as a full-stack workspace with three primary components:

1. Agent backend (Python, crew-based orchestration)
2. Agent frontend (React + Vite UI)
3. Excalidraw MCP server (drawing execution layer)

Repository structure (high level)

- agent-backend: multi-agent orchestration and tool calling
- agent-frontend/crewai_excalidraw: user interface and canvas
- excalidraw-mcp: Excalidraw MCP server and drawing runtime

Project description

The system converts user prompts into a structured plan and then into a sequence of drawing operations. The backend focuses on intent interpretation, task planning, and tool selection. The frontend presents a prompt box and canvas, relaying structured commands and rendering results. The MCP server provides a consistent interface for writing Excalidraw elements, which keeps the drawing logic isolated and reusable.

Agent backend

Location: agent-backend/

Responsibilities

- Parse user input and determine intent
- Plan tasks and orchestrate agent steps
- Call tools that generate drawing instructions
- Return structured output to the frontend

Key files

- agent-backend/src/agent_backend/crew.py: agent workflow and orchestration
- agent-backend/src/agent_backend/main.py: entry point
- agent-backend/src/agent_backend/config/agents.yaml: agent definitions
- agent-backend/src/agent_backend/config/tasks.yaml: task definitions
- agent-backend/src/agent_backend/tools/custom_tool.py: custom tool bridge

Agent frontend

Location: agent-frontend/crewai_excalidraw/

Responsibilities

- Provide UI for prompting and canvas interaction
- Manage local canvas state and rendering
- Display progress and results from the backend
- Dispatch drawing operations to the MCP server

Key files

- agent-frontend/crewai_excalidraw/app/routes/home.tsx: main screen
- agent-frontend/crewai_excalidraw/app/canvas/CanvasEngine.tsx: canvas integration
- agent-frontend/crewai_excalidraw/app/utils/layoutEngine.ts: layout helpers
- agent-frontend/crewai_excalidraw/app/utils/nodeGenerator.ts: diagram node creation

Excalidraw MCP

Location: excalidraw-mcp/

Responsibilities

- Expose drawing capabilities through MCP
- Translate structured instructions into Excalidraw elements
- Provide a stable boundary for backend and frontend integration

Key files

- excalidraw-mcp/src/mcp-entry.tsx: MCP entry point
- excalidraw-mcp/src/server.ts: server runtime
- excalidraw-mcp/api/mcp.ts: MCP API surface

System flow (end to end)

1. User enters a prompt in the frontend.
2. Frontend sends the prompt to the agent backend.
3. Backend plans tasks and produces structured drawing instructions.
4. Frontend relays instructions to the Excalidraw MCP server.
5. MCP server creates Excalidraw elements and returns results.
6. Frontend renders the updated diagram on the canvas.

Development setup

Backend (Python)

1. Install uv (one time):
   pip install uv
2. Install dependencies:
   cd /Users/dikshanta/Documents/agent-excalidraw/agent-backend
   uv sync
3. Configure environment:
   Create .env and set GROQ_API_KEY=...
4. Run the backend:
   uv run agent_backend

Frontend (React)

1. Install dependencies:
   cd /Users/dikshanta/Documents/agent-excalidraw/agent-frontend/crewai_excalidraw
   pnpm install
2. Run the dev server:
   pnpm dev

Excalidraw MCP server

1. Install dependencies:
   cd /Users/dikshanta/Documents/agent-excalidraw/excalidraw-mcp
   pnpm install
2. Run the dev server:
   pnpm dev

Notes

- Keep all three services running for full end-to-end flow.
- If you change agent definitions or tasks, restart the backend.
