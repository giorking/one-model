import pytest
from loguru import logger
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import JinaChat, ChatOllama
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
import os

os.environ[
    "SERPAPI_API_KEY"
] = "bf18861eba7c19d7fa0928aa2b212273a4b970b6b75bd1cbd17f1b85845e2ed7"
os.environ[
    "JINACHAT_API_KEY"
] = "i49s9e9Wjn40z1TgEMRb:fb75fa97e1a0db5cf8922db9b29d4c372395c353d3f7e71ec21efc779ea68a20"


def test_langchain_llm():
    llm: Ollama = Ollama(model="llama2")
    answer = llm("The first man on the moon was ...")
    logger.info("answer {}", answer)

    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    prompt_text = prompt.format(product="colorful socks")
    answer = llm(prompt_text)
    logger.info("answer {}", answer)


def test_langchain_agent():
    llm: ChatOllama = ChatOllama()
    # llm: JinaChat = JinaChat()
    # Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
    tools = load_tools(["serpapi", "llm-math"], llm=llm)
    # Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
    agent = initialize_agent(
        tools,
        llm,
        # agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )

    # Now let's test it out!
    # result = agent.run("What was the high temperature in SF yesterday in Fahrenheit?")
    result = agent(
        {
            "input": "What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?",
            "chat_history": [],
        }
    )
    logger.info("result {}", result)


if __name__ == "__main__":
    # test_langchain_llm()
    test_langchain_agent()
