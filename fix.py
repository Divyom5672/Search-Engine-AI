import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# -------------------- TOOLS --------------------
arxiv = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200),
    name="arxiv_research"
)

wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200),
    name="wikipedia_lookup"
)

search = DuckDuckGoSearchRun(name="web_search")

tools = [arxiv, wiki, search]

# -------------------- STREAMLIT UI --------------------
st.title("🔎 Groq LLM Search Engine App")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")

if not api_key:
    st.warning("Please enter your Groq API key.")
    st.stop()

# -------------------- SESSION STATE --------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I can search research papers, Wikipedia, and the web. Ask me anything!"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -------------------- USER INPUT --------------------
user_query = st.chat_input("Ask me anything about research papers or Wikipedia topics...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    # -------------------- LLM --------------------
    llm = ChatGroq(
        groq_api_key=api_key,
        model="openai/gpt-oss-120b",
        temperature=0
    ).bind_tools(tools)

    # -------------------- PROMPT --------------------
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a smart research assistant.

Tool usage rules:
- Research papers, studies, authors, citations → arxiv_research
- Definitions, concepts, people, history → wikipedia_lookup
- Current events, comparisons, general web info → web_search

Do NOT mention tools.
Return only the final answer.
"""
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    # -------------------- AGENT --------------------
    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt_template
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        verbose=True
    )

    # -------------------- RUN --------------------
    response = executor.invoke({"input": user_query})

    final_answer = response["output"]
    steps = response.get("intermediate_steps", [])

    st.session_state.messages.append(
        {"role": "assistant", "content": final_answer}
    )

    st.chat_message("assistant").write(final_answer)

    # -------------------- REASONING VIEW --------------------
    with st.expander("🧠 See how the agent reasoned"):
        
        for i, (action, observation) in enumerate(steps, 1):
            st.markdown(f"### Step {i}")
            st.markdown(f"**Tool used:** `{action.tool}`")
            st.markdown("**Input:**")
            st.code(action.tool_input)
            st.markdown("**Output:**")
            st.write(observation)