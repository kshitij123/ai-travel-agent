"""
Entry point. Run: python main.py

Starts the TravelAgent which handles the full LLM conversation loop.
See agent.py for the detailed call/response flow.
"""

from agent import TravelAgent

if __name__ == "__main__":
    TravelAgent().run()
