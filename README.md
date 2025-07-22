# Chat-with-pdf

Chat-with-pdf is a Streamlit web application that lets you upload one or more PDF files and interact with their content using natural language. You can ask questions, request summaries, or explanations from your documents, and the app will provide detailed answers using Google Gemini AI. It also suggests related YouTube videos and Wikipedia articles for further learning based on your queries.

## Features
- Upload and process multiple PDF files at once
- Ask questions about the content of your PDFs
- Get detailed, context-aware answers powered by Google Gemini
- Summarize any uploaded PDF
- Receive related YouTube and Wikipedia links for further exploration
- Simple, interactive web interface (Streamlit)

## Demo
![screenshot](demo_screenshot.png) <!-- Add a screenshot if available -->

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/Chat-with-pdf.git
cd Chat-with-pdf
```

### 2. Install dependencies
It is recommended to use a virtual environment (e.g., `venv` or `conda`).

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file in the project root with the following content:

```
GOOGLE_API_KEY=your_google_api_key
GOOGLE_ENGINE_ID=your_google_custom_search_engine_id
```
- `GOOGLE_API_KEY`: Your Google Generative AI API key
- `GOOGLE_ENGINE_ID`: Your Google Custom Search Engine ID (for fetching YouTube and Wikipedia links)

### 4. Run the application
```bash
streamlit run app.py
```

The app will open in your browser. Upload your PDF files using the sidebar, process them, and start chatting!

## Usage
- **Upload PDFs:** Use the sidebar to upload one or more PDF files and click "Submit & Process".
- **Ask Questions:** Type your question in the main input box and press Enter.
- **Summarize:** Select a PDF from the dropdown and click "Summarize" to get a summary.
- **Related Links:** After each answer, the app suggests relevant YouTube videos and Wikipedia articles.

## Requirements
- Python 3.8+
- See `requirements.txt` for all Python dependencies

## Notes
- You need valid Google API credentials for both Generative AI and Custom Search.
- The app uses FAISS for fast document retrieval and KeyBERT for keyword extraction.

## License
[MIT](LICENSE)

---

*Feel free to contribute or open issues for suggestions and improvements!*
