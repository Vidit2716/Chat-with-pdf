import streamlit as st 
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import requests
from urllib.parse import quote
load_dotenv()
CX = os.getenv("GOOGLE_ENGINE_ID")

API_KEY =  os.getenv("GOOGLE_API_KEY")

def search_links(query, site):
    """Searches Google Custom Search for the given site and returns top URLs."""
    q = quote(f"{query} site:{site}")

    resp = requests.get(f"https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={CX}&q={q}")
    resp.raise_for_status()
    items = resp.json().get("items", [])
    return [item["link"] for item in items]


def get_related_links(question):
    """Fetches related YouTube and Wikipedia links for a question."""
    yt_links = search_links(question, "youtube.com")
    wiki_links = search_links(question, "wikipedia.org")
    return yt_links, wiki_links



def generate_embeddings_and_save_faiss(text_chunks: list[str], save_path: str = "faiss_index"):
    """
    Generates embeddings for text chunks using Google Generative AI and
    saves them to a FAISS index locally. 
    """
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create FAISS index from texts and embeddings model
    # This handles embedding generation and FAISS index creation internally [2, 3]
    vector_store = FAISS.from_texts(text_chunks, embeddings_model)

    # Save the FAISS index locally
    vector_store.save_local(save_path)
    print(f"FAISS index saved to {save_path}/")






def get_pdf_text(pdf_docs) -> str:
    """
    Reads multiple PDF files and concatenates all extracted text into a single string.

    Args:
        pdf_docs: List of file-like objects from Streamlit uploader.

    Returns:
        A combined string containing text extracted from all pages of all PDFs.
    """
    text = ""
    for pdf_doc in pdf_docs:
        reader = PdfReader(pdf_doc)
        num_pages = len(reader.pages)
        for page_index in range(num_pages):
            page = reader.pages[page_index]
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def get_text_chunks (text ):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks

def get_conversational_chain():
    prompt_template = """ 
    Answer the question as detailed as possible from the provided context , make sure to provide all the details, if it asks simple definations of full forms ans them even if 
     it is not in the context  ,if the ans is not in 
    the provided context just say ,"ans is not available in the context ", don't provide the wrong answer \n\n 
    Context :\n{context}?\n
    question:\n{question}\n
    Answer: 
    """ 
    model = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    st.write("Reply: ", response["output_text"]) 
    yt_links, wiki_links = get_related_links(user_question)
    
    # Show up to 4 YouTube videos with styled title, thumbnail, and link
    if yt_links:
        st.markdown('<span style="font-size:28px; color:#FFFFFF;">Related YouTube videos:</span>', unsafe_allow_html=True)
        # Group links into pairs
        for i in range(0, min(4, len(yt_links)), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(yt_links[:4]):
                    link = yt_links[i + j]
                    if "watch?v=" in link:
                        video_id = link.split("watch?v=")[-1].split("&")[0]
                        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/0.jpg"
                        try:
                            oembed_url = f"https://www.youtube.com/oembed?url={link}&format=json"
                            video_info = requests.get(oembed_url).json()
                            title = video_info.get("title", "YouTube Video")
                        except Exception:
                            title = "YouTube Video"
                        col.image(thumbnail_url, width=200)
                        col.markdown(
                            f'<a href="{link}" target="_blank">'
                            f'<span style="font-family:Arial, sans-serif; font-size:20px; color:#FFFFFF;">{title}</span>'
                            '</a>',
                            unsafe_allow_html=True
                        )
                    else:
                        col.markdown(
                            f'<a href="{link}" target="_blank">'
                            f'<span style="font-family:Arial, sans-serif; font-size:20px; color:#FFFFFF;">YouTube Video</span>'
                            '</a>',
                            unsafe_allow_html=True
                        )

    # Show up to 4 Wikipedia articles with styled title and link
    if wiki_links:
        st.markdown('<span style="font-size:28px; color:#FFFFFF;">Related Wikipedia articles:</span>', unsafe_allow_html=True)
        for link in wiki_links[:4]:
            try:
                title = link.split("/wiki/")[-1].replace("_", " ")
            except Exception:
                title = "Wikipedia Article"
            st.markdown(
                f'<a href="{link}" target="_blank">'
                f'<span style="font-family:Arial, sans-serif; font-size:20px; color:#FFFFFF;">{title}</span>'
                '</a>',
                unsafe_allow_html=True
            )
    






def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                generate_embeddings_and_save_faiss(text_chunks)
                st.success("Done")
  
    
  

if __name__ == "__main__":
    main()