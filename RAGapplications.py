import os
import re
import validators
import streamlit as st
import traceback

## Langchain Imports
from langchain_community.document_loaders import PyPDFLoader, YoutubeLoader, UnstructuredURLLoader
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# Load environment variables
load_dotenv()

# Langsmith Tracking 
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "RAG MEGA PROJECT"

## Set up Streamlit app
st.set_page_config(
    page_title="AI-Powered Knowledge Hub",
    page_icon="🤖",
    layout="wide",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/',
        'Report a bug': 'https://github.com/Chhotoo-11/RAG-Applications/issues',
        'About': "This app was created by Chhotoo Solanki."
    }
)
st.title("🤖 AI Knowledge Assistant")

## Sidebar for settings
with st.sidebar:
    st.title("🛠️ Configuration")
    st.write("Configure your app settings below.")
    st.markdown("---")

    # Section for app navigation
    st.subheader("🌐 App Mode")
    app_mode = st.selectbox(
        "Choose the app mode",
        ["Chat with PDF", "URL/YouTube Summarizer", "Web Search"]
    )

# Initialize LLM (always Groq)
def get_llm():
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    return ChatGroq(model_name="Gemma2-9b-It")

# Get LLM
llm = get_llm()

# -------------------- HELPER FUNCTION --------------------
def normalize_youtube_url(url: str) -> str:
    """Convert short youtu.be links to long youtube.com format"""
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    if match:
        return f"https://www.youtube.com/watch?v={match.group(1)}"
    return url

# ------------------- CHAT WITH PDF -------------------
if app_mode == "Chat with PDF":
    st.header("📄 Chat with PDF")
    st.write("Start interacting with your PDF documents in a chat format. Upload a PDF and ask questions or extract information effortlessly.")

    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf, 'wb') as file:
                file.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        def get_summary(splits):
            prompt_template = """
            Provide a summary of the following content in 300 words:
            Content:{text}
            """
            prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
            chain = load_summarize_chain(
                llm,
                chain_type="map_reduce",
                map_prompt=prompt,
                combine_prompt=prompt
            )
            result = chain.invoke(splits)
            return result["output_text"] if isinstance(result, dict) else str(result)

        def is_summary_request(query):
            summary_keywords = ['summary', 'summarize', 'summarization', 'description',
                                'describe', 'overview', 'brief', 'briefly',
                                'digest', 'recap', 'outline']
            return any(keyword in query.lower() for keyword in summary_keywords)

        user_input = st.text_input("Your question:")
        if user_input:
            if is_summary_request(user_input):
                summary = get_summary(splits)
                st.write("Assistant: Here's a summary of the document(s):")
                st.success(summary)
            else:
                response = rag_chain.invoke({"input": user_input})
                st.success(f"Assistant: {response['answer']}")

# ------------------- WEB SEARCH -------------------
elif app_mode == "Web Search":
    st.header("🔎 Web Search")
    st.write("Easily search the web right from this app. Simply enter your query below to begin.")

    arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

    search = DuckDuckGoSearchRun(name="Search")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg['content'])

    if prompt := st.chat_input(placeholder="Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        tools = [search, arxiv, wiki]

        search_agent = initialize_agent(
            tools, llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True
        )

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({'role': 'assistant', "content": response})
            st.write(response)

# ------------------- URL / YOUTUBE SUMMARIZER -------------------
elif app_mode == "URL/YouTube Summarizer":
    st.header("🌐 URL/YouTube Summarizer")
    st.write("Enter a URL or YouTube link to quickly generate a concise summary of the content.")

    generic_url = st.text_input("Enter a URL", label_visibility="collapsed").strip()

    prompt_template = """
    Provide a summary of the following content in 300 words:
    Content: {text}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    def process_and_summarize(docs):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_documents(docs)
        chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=prompt, combine_prompt=prompt)
        result = chain.invoke(texts)
        return result["output_text"] if isinstance(result, dict) else str(result)

    if st.button("Summarize the Content from YT or Website"):
        if not generic_url:
            st.error("Please enter a URL to get started.")
        elif not validators.url(generic_url):
            st.error("Please enter a valid URL. It can be a YouTube video URL or website URL.")
        else:
            try:
                with st.spinner("Processing..."):
                    if any(domain in generic_url for domain in ["youtube.com", "youtu.be"]):
                        generic_url = normalize_youtube_url(generic_url)
                        try:
                            loader = YoutubeLoader.from_youtube_url(
                                generic_url,
                                add_video_info=True,
                                language=["en"]
                            )
                            docs = loader.load()
                        except Exception:
                            st.warning("Transcript not available, falling back to video description.")
                            docs = [Document(page_content=f"Video description: {generic_url}")]

                    else:
                        loader = UnstructuredURLLoader(
                            urls=[generic_url],
                            ssl_verify=True,
                            headers={"User-Agent": "Mozilla/5.0"}
                        )
                        docs = loader.load()

                    output_summary = process_and_summarize(docs)
                    st.success(output_summary)

            except Exception as e:
                st.error(f"Error: {e}")
                st.code(traceback.format_exc())
