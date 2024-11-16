import json
import os
import sys
import time

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper
from cdp_langchain.tools import CdpTool
from pydantic import BaseModel, Field
from cdp import *

# File to persist the agent's CDP MPC Wallet Data
wallet_data_file = "wallet_data.txt"


class MusicAnalysisResult(BaseModel):
    """Schema for music content analysis results"""
    genre: str = Field(description="Music genre classification")
    style: str = Field(description="Musical style characteristics")
    similarity_score: float = Field(
        description="Similarity score to known content")
    safe_for_minting: bool = Field(
        description="Whether content is safe to mint")
    tags: list = Field(description="Relevant tags for the content")
    confidence: float = Field(description="Confidence score of the analysis")


def initialize_agent():
    """Initialize the agent with CDP Agentkit for music content analysis"""
    # Initialize LLM with music analysis capabilities
    llm = ChatOpenAI(model="gpt-4")

    wallet_data = None
    if os.path.exists(wallet_data_file):
        with open(wallet_data_file) as f:
            wallet_data = f.read()

    # Configure CDP Agentkit
    values = {}
    if wallet_data is not None:
        values = {"cdp_wallet_data": wallet_data}

    agentkit = CdpAgentkitWrapper(**values)

    # Persist wallet data
    wallet_data = agentkit.export_wallet()
    with open(wallet_data_file, "w") as f:
        f.write(wallet_data)

    # Initialize toolkit with custom music analysis tools
    cdp_toolkit = CdpToolkit.from_cdp_agentkit_wrapper(agentkit)
    tools = cdp_toolkit.get_tools()

    # Add custom music analysis tools
    tools.extend([{
        "name": "analyze_music_content",
        "description":
        "Analyze music content for genre, style, and similarity to existing content",
        "parameters": {
            "content_uri": "URI of the music content to analyze",
            "content_type": "Type of content (audio/video)"
        }
    }, {
        "name": "verify_content_rights",
        "description": "Verify content ownership and rights",
        "parameters": {
            "content_uri": "URI of the content to verify",
            "creator_address": "Creator's wallet address"
        }
    }])

    memory = MemorySaver()
    config = {"configurable": {"thread_id": "MVer Content Analysis Agent"}}

    return create_react_agent(
        llm,
        tools=tools,
        checkpointer=memory,
        state_modifier=
        """You are a specialized AI agent for MVer music platform that:
        1. Analyzes music content for genre, style, and originality
        2. Verifies content rights and ownership
        3. Suggests tags and metadata
        4. Determines if content is safe for minting

        Use the available tools to help creators verify and mint their content.
        Be thorough in analysis but concise in responses.
        Focus on musical elements and creator rights."""), config


def analyze_content(agent_executor, config, content_uri):
    """Analyze music content and provide detailed results"""
    analysis_prompt = f"""
    Please analyze the music content at {content_uri} and provide:
    1. Genre and style classification
    2. Similarity check with existing content
    3. Content safety verification
    4. Suggested tags
    5. Confidence score
    """

    result = None
    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content=analysis_prompt)]}, config):
        if "agent" in chunk:
            result = chunk["agent"]["messages"][0].content
        elif "tools" in chunk:
            print("Processing...", end="\r")

    return MusicAnalysisResult(**json.loads(result)) if result else None


def run_interactive_mode(agent_executor, config):
    """Run interactive content analysis mode"""
    print("MVer Content Analysis Mode - Type 'exit' to end")

    while True:
        try:
            command = input(
                "\nCommand (analyze/verify/exit): ").lower().strip()

            if command == "exit":
                break

            elif command == "analyze":
                content_uri = input("Enter content URI: ")
                result = analyze_content(agent_executor, config, content_uri)
                if result:
                    print("\nAnalysis Results:")
                    print(f"Genre: {result.genre}")
                    print(f"Style: {result.style}")
                    print(f"Similarity Score: {result.similarity_score}")
                    print(f"Safe for Minting: {result.safe_for_minting}")
                    print(f"Tags: {', '.join(result.tags)}")
                    print(f"Confidence: {result.confidence}")

            elif command == "verify":
                content_uri = input("Enter content URI: ")
                creator = input("Enter creator address: ")

                verification_prompt = f"""
                Verify rights and ownership for content at {content_uri}
                from creator {creator}
                """

                for chunk in agent_executor.stream(
                    {"messages": [HumanMessage(content=verification_prompt)]},
                        config):
                    if "agent" in chunk:
                        print(chunk["agent"]["messages"][0].content)
                    elif "tools" in chunk:
                        print("Verifying...", end="\r")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    """Start the MVer content analysis agent"""
    agent_executor, config = initialize_agent()
    run_interactive_mode(agent_executor, config)


if __name__ == "__main__":
    print("Starting MVer Content Analysis Agent...")
    main()
