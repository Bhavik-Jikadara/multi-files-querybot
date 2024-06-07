# Multiple PDFs QueryBot

The Multiple PDFs QueryBot is a Python-based tool for interacting with multiple PDF documents through natural language queries. Users can ask questions about the content of the PDFs, and the app will deliver relevant answers based on the information within the documents. This application leverages a language model to produce precise responses. Just to let you know, the app's responses are limited to the content of the loaded PDFs.

## How it works?

The application follows these steps to respond to your questions:

- **PDF Loading**: The app reads multiple PDF documents and extracts their text content.
- **Text Chunking**: The extracted text is divided into smaller, manageable chunks for efficient processing.
- **Language Model**: The application employs a language model to create vector representations (embeddings) of the text chunks.
- **Similarity Matching**: When a question is asked, the app compares it to the text chunks and identifies those with the highest semantic similarity.
- **Response Generation**: The selected chunks are input into the language model, which generates a response based on the relevant content from the PDFs.

## Dependencies and Installation

To install the MultiPDF Chat App, please follow these steps:

### Step 1: Clone the repository

    | git clone https://github.com/Bhavik-Jikadara/multiple-pdfs-querybot.git
    | cd multiple-pdfs-querybot/

### Step 2: Create a virtualenv (windows user)

    | pip install virtualenv
    | virtualenv venv
    | source venv/Scripts/activate

### Step 3: Install the requirements libraries using pip

    | pip install -r requirements.txt

### Step 4: Type this command and run the project

    | streamlit run app.py

## License

The Multiple PDFs QueryBot is released under the [Apache License 2.0](https://github.com/Bhavik-Jikadara/multiple-pdfs-querybot/blob/main/LICENSE).
