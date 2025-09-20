
# RAG with HuggingFace Embeddings + Gemini API

This guide explains how to set up a Retrieval-Augmented Generation (RAG)
pipeline using **HuggingFace embeddings** (free & local) for vector
storage, and **Gemini API** for answering questions.

------------------------------------------------------------------------

## 📌 Steps

### 1. Install Required Packages

``` bash
pip install langchain langchain-community langchain-google-genai sentence-transformers faiss-cpu
```

------------------------------------------------------------------------

### 2. Load Documents

``` python
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load documents from ./knowledge folder
documents = []
for root, _, files in os.walk("./knowledge"):
    for file in files:
        file_path = os.path.join(root, file)
        if file.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file.lower().endswith(".txt"):
            loader = TextLoader(file_path)
            documents.extend(loader.load())

print(f"Loaded {len(documents)} documents.")
```

------------------------------------------------------------------------

### 3. Split into Chunks

``` python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks.")
```

------------------------------------------------------------------------

### 4. Create Vector Store with HuggingFace Embeddings

``` python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
```

------------------------------------------------------------------------

### 5. Connect Gemini API as the LLM

``` python
from langchain_google_genai import ChatGoogleGenerativeAI

GOOGLE_GEMINI_API_KEY = "YOUR_API_KEY"

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_GEMINI_API_KEY
)
```

------------------------------------------------------------------------

### 6. Build RAG Chain

``` python
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

prompt_template = """
You are a helpful assistant. 
Answer the user question using only the following context: 
{context}

Question: {question}
If the answer is not in the context, say "I don't know".
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

------------------------------------------------------------------------

### 7. Ask Questions

``` python
question = "Who is Dasun?"
answer = rag_chain.invoke(question)
print("Answer:", answer)
```

------------------------------------------------------------------------

## ✅ Summary

-   **Embeddings**: HuggingFace (local & free)\
-   **Vector DB**: FAISS\
-   **LLM**: Gemini (free API)\
-   **RAG Flow**: Documents → Chunks → Embeddings → FAISS → Retriever →
    Gemini → Answer

This avoids Gemini embedding quota issues while still using Gemini as
the reasoning engine.