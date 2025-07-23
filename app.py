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
from keybert import KeyBERT
from urllib.parse import quote
load_dotenv()
CX = os.getenv("GOOGLE_ENGINE_ID")

API_KEY =  os.getenv("GOOGLE_API_KEY")

def extract_keywords(text, num_keywords=5):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=num_keywords)
    # Return only the keyword strings
    return [kw[0] for kw in keywords]

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
    the provided context just say ,"Ans is not available in the context ", don't provide the wrong answer \n\n 
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
    
    if response["output_text"] != "Ans is not available in the context":
        keywords = extract_keywords(response["output_text"])
        print(keywords)
        
        # Limit to top 3 keywords
        top_keywords = keywords[:3]
        keyword_links = []
        for kw in top_keywords:
            yt_links = search_links(kw, "youtube.com")[:2]
            wiki_links = search_links(kw, "wikipedia.org")[:2]
            keyword_links.append((kw, yt_links, wiki_links))

        # Display related links for each keyword
        for kw, yt_links, wiki_links in keyword_links:
            st.markdown(f'<span style="font-size:24px; color:#FFFFFF;">Related links for keyword: <b>{kw}</b></span>', unsafe_allow_html=True)
            
            # YouTube links
            if yt_links:
                st.markdown('<span style="font-size:20px; color:#FFFFFF;">YouTube videos:</span>', unsafe_allow_html=True)
                cols = st.columns(len(yt_links))
                for i, link in enumerate(yt_links):
                    if "watch?v=" in link:
                        video_id = link.split("watch?v=")[-1].split("&")[0]
                        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/0.jpg"
                        try:
                            oembed_url = f"https://www.youtube.com/oembed?url={link}&format=json"
                            video_info = requests.get(oembed_url).json()
                            title = video_info.get("title", "YouTube Video")
                        except Exception:
                            title = "YouTube Video"
                        cols[i].image(thumbnail_url, width=200)
                        cols[i].markdown(
                            f'<a href="{link}" target="_blank">'
                            f'<span style="font-family:Arial, sans-serif; font-size:18px; color:#FFFFFF;">{title}</span>'
                            '</a>',
                            unsafe_allow_html=True
                        )
                    else:
                        cols[i].markdown(
                            f'<a href="{link}" target="_blank">'
                            f'<span style="font-family:Arial, sans-serif; font-size:18px; color:#FFFFFF;">YouTube Video</span>'
                            '</a>',
                            unsafe_allow_html=True
                        )

        # Wikipedia links
        if wiki_links:
            st.markdown('<span style="font-size:20px; color:#FFFFFF;">Wikipedia articles:</span>', unsafe_allow_html=True)
            for link in wiki_links:
                try:
                    title = link.split("/wiki/")[-1].replace("_", " ")
                except Exception:
                    title = "Wikipedia Article"
                st.markdown(
                    f'<a href="{link}" target="_blank">'
                    f'<span style="font-family:Arial, sans-serif; font-size:18px; color:#FFFFFF;">{title}</span>'
                    '</a>',
                    unsafe_allow_html=True
                )






def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            type=["pdf"],  # ⬅️ only allows PDFs
            accept_multiple_files=True
        )

        if st.button("Submit & Process") and pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                generate_embeddings_and_save_faiss(text_chunks)
                st.success("Done")


    if pdf_docs:
        pdf_names = [pdf.name for pdf in pdf_docs]
        selected_pdf = st.selectbox("Select a PDF to summarize:", pdf_names)
        if selected_pdf:
            if st.button("Summarize"):
                with st.spinner(f"Summarizing {selected_pdf}..."):
                    quest = f"summarize this document {selected_pdf}"
                    user_input(quest)

    if user_question:
        user_input(user_question)
  

if __name__ == "__main__":
    main()