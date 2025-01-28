import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from transformers import pipeline  # Hugging Face pipeline for using T5 model
from bs4 import BeautifulSoup
import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline


# Function to scrape text content from a website
def scrape_website(url):
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Failed to retrieve the website. Please check the URL.")
        return ""
    
    soup = BeautifulSoup(response.text, "html.parser")
    text = " ".join([p.get_text() for p in soup.find_all("p")])
    return text

# Function to split the text into manageable chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vectorstore from the text chunks
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-base")  # Using lightweight instructor model
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create the conversation chain using a smaller model
def get_conversation_chain(vectorstore):
    # Use HuggingFacePipeline with a smaller model like `t5-small`
    t5_model = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",  # Smaller model for low-end systems
        tokenizer="google/flan-t5-large",
        max_new_tokens=512,  # Increase the maximum token output
        temperature=1.0,  # Control creativity
        top_p=0.9,  # Nucleus sampling
        top_k=50
    )

    llm = HuggingFacePipeline(pipeline=t5_model)

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

    # Create a conversation chain using the T5 model
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain

# Function to handle the user input
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Display the conversation (alternating user and bot messages)
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(f"**You:** {message.content}", unsafe_allow_html=True)
        else:
            st.write(f"**Bot:** {message.content}", unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Websites", page_icon=":globe_with_meridians:")

    # Initialize session state for conversation
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Title of the app
    st.header("Chat with Websites :globe_with_meridians:")

    # User input for querying the website
    user_question = st.text_input("Ask a question about the website:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Website URL")

        # Input for website URL
        website_url = st.text_input("Enter the website URL here:")

        if st.button("Process"):
            with st.spinner("Processing..."):
                # Scrape text from the website
                raw_text = scrape_website(website_url)

                if raw_text:
                    # Split the text into chunks
                    text_chunks = get_text_chunks(raw_text)

                    # Create a vector store using the text chunks
                    vectorstore = get_vectorstore(text_chunks)

                    # Create the conversation chain using the T5 model
                    st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
