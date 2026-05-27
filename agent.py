from dotenv import load_dotenv
import os
import base64
import io
import discord
import asyncio

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from tavily import TavilyClient

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def generateAndSendImage(prompt: str, runTime: ToolRuntime):
    """""use this tool to generate and send image"""

    llm = ChatOpenAI(model="gpt-4o-mini")

    message = runTime.config["configurable"]["message"]
    loop = runTime.config["configurable"]["loop"]

    tool = {"type": "image_generation", "quality": "low"}

    llm_with_tools = llm.bind_tools([tool])

    ai_message = llm_with_tools.invoke(prompt)

    image = ai_message.content_blocks[0]["base64"]

    base64_string = base64.b64decode(image)
    image_bytes = io.BytesIO(base64_string)

    file = discord.File(fp=image_bytes, filename="image.png")

    asyncio.run_coroutine_threadsafe(
        message.channel.send(file=file),
        loop
    )

    return "Image sent successfully!"


@tool
def surfInterNet(query: str):
    """"Use this tool to surf internet and get latest information"""
    
    result = tavily_client.search(query=query)
    
    return str(result)


model = ChatMistralAI(
    model="mistral-small-latest"
)

agent = create_agent(
    model=model,
    tools=[surfInterNet, generateAndSendImage],
    system_prompt=""" provide clean out to the user """
)