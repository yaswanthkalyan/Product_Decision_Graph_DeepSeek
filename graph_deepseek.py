from __future__ import annotations as _annotations
import os
import asyncio
import requests
from colorama import Fore
from dotenv import load_dotenv
from enum import Enum
from bs4 import BeautifulSoup
from dataclasses import dataclass, field
from typing import List
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_graph import BaseNode, End, Graph, GraphRunContext
from pydantic_ai.models.groq import GroqModel
from tavily import TavilyClient
import streamlit as st

# Load environment variables
load_dotenv()

# ✅ Groq Model Initialization
model = GroqModel(
    model_name="deepseek-r1-distill-llama-70b",
    api_key=os.getenv("GROQ_API_KEY")
)

# ✅ Data Classes
@dataclass
class Product:
    name: str
    url: str
    keywords: List[str]
    num_rounds: int = 0

class Choice(Enum):
    buy = "Buy"
    skip = "Skip"

@dataclass
class Decision:
    sentiment: int
    decision: Choice
    explanation: str

@dataclass
class Argument:
    sentiment: int
    body: str

@dataclass
class State:
    product: Product
    messages: List[str] = field(default_factory=list)
    num_rounds: int = 0
    sentiment: int = 0  # Aggregate sentiment score
    pro_sentiment: int = 0  # Track positive argument strength
    con_sentiment: int = 0  # Track negative argument strength

# ✅ Agents (Using Groq Model)
pro_agent = Agent[str, Argument](
    model=model,
    result_type=Argument,
    deps_type=Product,
    system_prompt="You are a product manager. Write a strong, positive argument for why the product is worth buying. Assign a sentiment score from 0 to 100."
)

con_agent = Agent[str, Argument](
    model=model,
    result_type=Argument,
    deps_type=Product,
    system_prompt="You are a critic. Write a strong, negative argument for why the product is not worth buying. Assign a sentiment score from -100 to 0."
)

# ✅ **Balanced Sentiment Reasoning Agent**
reasoning_agent = Agent[str, Decision](
    model=model,
    result_type=Decision,
    system_prompt=(
        "You are a product reviewer. Analyze both arguments and make a decision. "
        "Consider the following sentiment scores: "
        "Pro Sentiment: {pro_sentiment}, Con Sentiment: {con_sentiment}. "
        "Return 'Buy' if the pro sentiment is at least 20 points stronger. "
        "Return 'Skip' if the con sentiment is at least 15 points stronger. "
        "If the difference is less than 15 points, return 'Skip' but indicate the decision is uncertain."
    ),
    deps_type=Product
)

decision_format_agent = Agent[str, Decision](
    model=model,
    result_type=Decision,
    system_prompt="Format the final decision in a structured way.",
    deps_type=Product
)

# ✅ **Graph Nodes**
@dataclass
class ModeratorNode(BaseNode[State]):
    async def run(self, ctx: GraphRunContext[State]) -> BaseNode:
        if ctx.state.num_rounds <= 1:
            return ProNode() if ctx.state.num_rounds % 2 == 0 else ConNode()
        print(Fore.YELLOW + "Calling reasoning_agent for final decision.")
        result = await reasoning_agent.run(
            user_prompt=f"Analyze arguments and make a decision. Pro Sentiment: {ctx.state.pro_sentiment}, Con Sentiment: {ctx.state.con_sentiment}.",
            deps=ctx.state.product
        )
        return DecisionNode(result.data)

@dataclass
class ProNode(BaseNode[State, None, Argument]):
    async def run(self, ctx: GraphRunContext[State]) -> BaseNode:
        result = await pro_agent.run(
            user_prompt="Make a strong argument for buying this product.",
            deps=ctx.state.product
        )
        if result.data and isinstance(result.data, Argument):
            ctx.state.messages.append(f"Pro: {result.data.body} (Sentiment: {result.data.sentiment})")
            ctx.state.pro_sentiment += result.data.sentiment  # Track only positive sentiment
        ctx.state.num_rounds += 1
        return ModeratorNode()

@dataclass
class ConNode(BaseNode[State, None, Argument]):
    async def run(self, ctx: GraphRunContext[State]) -> BaseNode:
        result = await con_agent.run(
            user_prompt="Make a strong argument against buying this product.",
            deps=ctx.state.product
        )
        if result.data and isinstance(result.data, Argument):
            ctx.state.messages.append(f"Con: {result.data.body} (Sentiment: {result.data.sentiment})")
            ctx.state.con_sentiment += abs(result.data.sentiment)  # Convert to positive for comparison
        ctx.state.num_rounds += 1
        return ModeratorNode()

@dataclass
class DecisionNode(BaseNode[State]):
    decision: Decision | None = None

    async def run(self, ctx: GraphRunContext[State]) -> End:
        prompt = f"Format the final decision for {ctx.state.product.name}."
        result = await decision_format_agent.run(user_prompt=prompt, deps=ctx.state.product)
        return End(result.data)

# ✅ Main Function with Streamlit UI
def main():
    st.title("Product Purchase Decision Maker 🛒")
    st.write("Enter the product details below to get a decision on whether to buy or skip the product.")

    # Input fields
    product_url = st.text_input("Enter the product URL:")
    keywords = st.text_input("Enter keywords (comma-separated):")
    keywords = [k.strip() for k in keywords.split(",")] if keywords else []

    if st.button("Analyze Product"):
        if not product_url or not keywords:
            st.error("Please provide both the product URL and keywords.")
        else:
            with st.spinner("Analyzing product..."):
                product = Product(
                    name="User Selected Product",
                    url=product_url,
                    keywords=keywords,
                )

                state = State(product)
                graph = Graph(nodes=(ModeratorNode, ProNode, ConNode, DecisionNode))

                decision, _ = asyncio.run(graph.run(ModeratorNode(), state=state, deps=product))

                # **Display Result in Streamlit**
                if decision.decision == Choice.buy:
                    st.success(f"✅ Decision: {decision.decision.value}\n💡 Reasoning: {decision.explanation}")
                else:
                    st.error(f"❌ Decision: {decision.decision.value}\n💡 Reasoning: {decision.explanation}")

# ✅ Run the script
if __name__ == "__main__":
    main()