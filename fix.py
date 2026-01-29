import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from langchain_core.prompts import ChatPromptTemplate

# -------------------- TOOLS --------------------
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

tools = [arxiv, wiki, search]

# -------------------- STREAMLIT UI --------------------
st.title("🔎 Groq LLM Search Engine App")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")

if not api_key:
    st.warning("Please enter your Groq API key to continue.")
    st.stop()

# -------------------- SESSION STATE --------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I am a chatbot who can also search the web. How may I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -------------------- USER INPUT --------------------
if prompt := st.chat_input("Ask me anything about research papers or Wikipedia topics..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # -------------------- LLM --------------------
    llm = ChatGroq(
        groq_api_key=api_key,
        model="openai/gpt-oss-120b",
        temperature=0,
        streaming=True
    )

    # 🔥 REQUIRED FOR GROQ TOOL CALLING
    llm = llm.bind_tools(tools)

    # -------------------- PROMPT --------------------
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful research assistant. "
         "Use tools when needed. "
         "Do not explain tool usage. "
         "Return only the final answer."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    # -------------------- AGENT --------------------
    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt_template
    )

    search_agent = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True
    )

    # -------------------- RUN AGENT --------------------
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        response = search_agent.invoke(
            {"input": prompt},
            callbacks=[st_cb]
        )

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

        st.write(response)