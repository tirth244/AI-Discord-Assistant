from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

import discord
from agent import agent
from langchain.messages import HumanMessage

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    
    async with message.channel.typing():
        content = message.content

        response = await agent.ainvoke(
            {"messages": [HumanMessage(content=content)]},
            config={
                "configurable": {
                    "message": message,
                    "loop": asyncio.get_event_loop()
                }
            }
        )

        agent_message = response["messages"][-1].content

    for i in range(0, len(agent_message), 2000):
         await message.channel.send(agent_message[i:i+2000])


client.run(os.getenv("DISCORD_API_KEY"))