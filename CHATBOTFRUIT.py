
# ---- imports (place at top of file) ----
import os
import tempfile
import pandas as pd
import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --- Groq API key (replace with your actual key or set via env before running) ---
GROQ_API_KEY = ""
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

st.title("The Best Fruit Chatbot Expert!")
st.write("Upload a CSV file with your fruit plot data to explore its data and ask questions using the chatbot.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Load & preview
        df = pd.read_csv(uploaded_file)
        st.success("File loaded successfully!")
        st.write("Data Description:")
        st.write(df.describe(include="all"))
        st.write("Preview:")
        st.write(df.head())

        # Make temp CSV for CSVLoader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file_path = tmp_file.name
            df.to_csv(tmp_file_path, index=False)

        # RAG pipeline
        loader = CSVLoader(file_path=tmp_file_path)
        documents = loader.load()

        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(documents)

        # Embeddings & vector store (local embeddings, no API needed)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)

        # LLM (Groq) + memory
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question",   # what the chain expects as input
            output_key="answer"     # tell memory which output to save
        )

        # Chain (no need to pass chat_history manually now)
        conversational_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            memory=memory,
            return_source_documents=True
        )

        st.success("Chatbot initialized and connected to data.")
        st.write("---")
        st.write("Chat with the data:")

        # show history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_query = st.chat_input("Ask a question about the data:")

        if user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = conversational_chain({"question": user_query})
                    answer = result["answer"]
                    st.markdown(answer)

            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # cleanup
        os.remove(tmp_file_path)

    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")

else:
    st.info("Please upload a CSV file to get started.")



#pip install streamlit pandas langchain langchain-community langchain-text-splitters faiss-cpu langchain-groq sentence-transformers
#python -m streamlit run app.py

