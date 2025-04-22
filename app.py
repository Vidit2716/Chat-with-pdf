import streamlit as st
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import requests
import speech_recognition as sr
from transformers import T5Tokenizer, T5ForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="Chat with PDF", layout="wide")
# Load environment variables from .env file
load_dotenv()

def get_pdf_text(pdf_docs):
    """Extract text from PDF documents."""
    texts = []
    for pdf in pdf_docs:
        text = ""
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() + " "
        texts.append((pdf.name, text))
    return texts

def get_text_chunks(text, chunk_size=2000):
    """Split text into manageable chunks."""
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def get_vector_store(text_chunks, pdf_name):
    """Create and save a vector store from text chunks."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(api_key=os.getenv("GOOGLE_API_KEY"), model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(f"faiss_index_{pdf_name}")
    except Exception as e:
        st.error(f"Failed to embed content. Error: {e}")

def get_conversational_chain():
    """Load the conversational chain for answering questions."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in the provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def get_resource_links(question):
    """Fetch resource links related to a question."""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        search_engine_id = os.getenv("SEARCH_ENGINE_ID")
        
        if not api_key or not search_engine_id:
            raise ValueError("Google API key or search engine ID is not set.")

        # Fetching YouTube links
        youtube_query = f"{question} site:youtube.com"
        youtube_url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={search_engine_id}&q={youtube_query}"
        youtube_response = requests.get(youtube_url)
        youtube_response.raise_for_status()
        youtube_data = youtube_response.json()
        youtube_links = [item['link'] for item in youtube_data.get('items', [])]

        # Fetching Wikipedia links
        wikipedia_query = f"{question} site:wikipedia.org"
        wikipedia_url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={search_engine_id}&q={wikipedia_query}"
        wikipedia_response = requests.get(wikipedia_url)
        wikipedia_response.raise_for_status()
        wikipedia_data = wikipedia_response.json()
        wikipedia_links = [item['link'] for item in wikipedia_data.get('items', [])]

        return youtube_links, wikipedia_links
    except requests.HTTPError as e:
        st.error(f"Failed to fetch resources. Error: {e.response.content}")
    except Exception as e:
        st.error(f"An error occurred while fetching resources: {e}")

    return [], []

def user_input(user_question):
    """Handle user input and fetch the answer along with resources."""
    pdf_docs = st.session_state.pdf_docs
    embeddings = GoogleGenerativeAIEmbeddings(api_key=os.getenv("GOOGLE_API_KEY"), model="models/embedding-001")

    # Initialize an empty list to store all the documents from all PDFs
    all_docs = []

    # Loop through each PDF and add its documents to the all_docs list
    for pdf in pdf_docs:
        new_db = FAISS.load_local(f"faiss_index_{pdf.name}", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        all_docs.extend(docs)

    chain = get_conversational_chain()

    response = chain.invoke({"input_documents": all_docs, "question": user_question}, return_only_outputs=True)

    # Define keywords that indicate an out-of-context question
    keywords = ['I cannot answer this question from the provided context', 'I am unable to provide an answer based on the given context','This question cannot be answered from the given context.','This question cannot be answered from the given context because it does not mention anything about.','Answer is not available in the context','answer is not available in the context','This question cannot be answered from the given context because it does not contain any information about','not available in the context', 'cannot answer', 'unable to provide']

    # Check if the response contains any of the keywords
    if any(keyword.lower() in response['output_text'].lower() for keyword in keywords):
        answer_with_resources = "The answer to your question is not available in the provided context."
        youtube_links = []
        wikipedia_links = []
    else:
        youtube_links, wikipedia_links = get_resource_links(user_question)
        
        # Format the links only if answer is available in the context
        youtube_links_formatted = "\n".join(f"- {link}" for link in youtube_links)
        wikipedia_links_formatted = "\n".join(f"- {link}" for link in wikipedia_links)
        
        # Combine the links with the answer
        answer_with_resources = f"{response['output_text']}"
        
        if youtube_links_formatted:
            answer_with_resources += f"\n\n**Related Resources:**\n\nYouTube:\n{youtube_links_formatted}"
        if wikipedia_links_formatted:
            answer_with_resources += f"\n\nWikipedia:\n{wikipedia_links_formatted}"

    return answer_with_resources, youtube_links, wikipedia_links

def summarize_text(text):
    """Summarize the provided text."""
    try:
        tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        
        summary = ""
        text_chunks = get_text_chunks(text)
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(summarize_chunk, chunk, tokenizer, model) for chunk in text_chunks]
            for future in as_completed(futures):
                summary += "- " + future.result().capitalize() + "\n"  # Capitalize the first letter of each point
        
        return summary
    except Exception as e:
        st.error(f"Error occurred during summarization: {e}")
        return None

def summarize_chunk(chunk, tokenizer, model):
    """Summarize a text chunk."""
    inputs = tokenizer.encode("summarize: " + chunk, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=100, num_beams=4, length_penalty=2.0, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def main():
    """Main function to run the Streamlit app."""
    

    # Replace "C:\Users\ADMIN\OneDrive\Desktop\ai_image-removebg-preview.png" with the actual path of the image you want to display
    st.image("images/ai_chatbot_image_with_background-removebg-preview.png", caption="HELLO!!, HOW CAN I HELP YOU TODAY?", use_column_width=False, width=500)

    
    st.header("Chat with PDF using Gemini💁")
    st.markdown("Welcome to the PDF chatbot. Upload your PDFs, ask questions, and get answers!")
    
    # Initialize session state variables
    if 'pdf_docs' not in st.session_state:
        st.session_state.pdf_docs = []
    if 'user_question' not in st.session_state:
        st.session_state.user_question = ""

    with st.form(key='question_form'):
        user_question = st.text_input("Ask a Question from the PDF Files", value=st.session_state.user_question)
        submit_button = st.form_submit_button(label='Enter')
    
    if submit_button:
        answer_with_resources, youtube_links, wikipedia_links = user_input(user_question)
        st.markdown("### Answer:")
        st.markdown(answer_with_resources)


    st.sidebar.title("Menu:")
    pdf_docs = st.sidebar.file_uploader("Upload PDF Files", accept_multiple_files=True, key="pdf_uploader")

    if pdf_docs and st.sidebar.button("Process PDFs"):
        with st.spinner("Processing..."):
            st.session_state.pdf_docs = pdf_docs
            pdf_texts = get_pdf_text(pdf_docs)
            for pdf_name, raw_text in pdf_texts:
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, pdf_name)
            st.success("Finished processing PDFs.")

    if pdf_docs:
        selected_pdf = st.sidebar.selectbox("Select PDF to Summarize:", [pdf.name for pdf in pdf_docs])
        if st.sidebar.button("Summarize Selected PDF"):
            with st.spinner("Summarizing..."):
                pdf_texts = get_pdf_text(pdf_docs)
                for pdf_name, raw_text in pdf_texts:
                    if pdf_name == selected_pdf:
                        summary = summarize_text(raw_text)
                        st.subheader(f"Summary for {selected_pdf}:")
                        st.write(summary)
                        break

    r = sr.Recognizer()
    with sr.Microphone() as source:
        if 'listening' not in st.session_state:
            st.session_state.listening = False

        button_label = "🎤 Start Listening" if not st.session_state.listening else "🎤 Stop Listening"
        if st.button(button_label):  # Use the placeholder for the listening button
            st.session_state.listening = not st.session_state.listening

        if st.session_state.listening:
            audio = r.record(source, duration=5)  # Listen for up to 5 seconds
            try:
                text = r.recognize_google(audio)
                st.session_state.user_question = text
                st.session_state.listening = False  # Stop listening after getting the input
                st.rerun()  # Rerun the app to update the text input field
            except sr.UnknownValueError:
                st.error("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                st.error(f"Could not request results from Google Speech Recognition service; {e}")

if __name__ == "__main__":
    main()
