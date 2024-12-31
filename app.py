import os
from pathlib import Path
from urllib.parse import urljoin, quote

import aiofiles
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langchain.agents import AgentExecutor, tool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_community.document_loaders.pdf import PyPDFLoader, PDFMinerLoader
from langchain_community.document_loaders.powerpoint import UnstructuredPowerPointLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredWordDocumentLoader
from langchain_core.documents.base import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from models import Conversation, List
from prompts import system_prompt

from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    # allow_origins=[os.environ.get('FRONTEND_URL')],  # Frontend origin
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

llm_global = ChatOpenAI(
    model='gpt-4o-mini',
    temperature = 0
)

embedding_model = OpenAIEmbeddings(
    model='text-embedding-3-small',
    show_progress_bar=True,
    chunk_size=1000,
)

# llm_global = ChatOllama(
#     model="llama3.2"
# )

# embedding_model = OllamaEmbeddings(
#     model= "mxbai-embed-large"
# )

pinecone_client = Pinecone(
    api_key = os.environ.get('PINECONE_API_KEY')
)

# Create pinecone index if not present
# pinecone_client.delete_index(os.environ.get('PINECONE_INDEX'))
if not any(index.get('name') == os.environ.get('PINECONE_INDEX') for index in pinecone_client.list_indexes()):
    pinecone_client.create_index(
        name = os.environ.get('PINECONE_INDEX'),
        dimension=1536,      #for text-embedding-ada-002 the dimension will be 1536. 384 is for sentence-transformer model
        metric="cosine",   # metric and dimensions are predefined for a model. refer the documentation
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

pinecone_vectorstore = PineconeVectorStore(
    embedding = embedding_model,
    pinecone_api_key = os.environ.get('PINECONE_API_KEY'),
    index_name= os.environ.get('PINECONE_INDEX'),
    namespace= os.environ.get('PINECONE_NAMESPACE'),
)

prompt_template = ChatPromptTemplate(messages=[
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

@tool(parse_docstring=True)
def search_cv_bank(query: str, k: int = 4)-> List[Document]:
    """returns k number of similar documents to query (vector search)

    Args:
        query : query to search documents by.
        k : number of results expected. Defaults to 4.

    Returns:
        List[Document]: k number of documents.
    """
    return pinecone_vectorstore.as_retriever(search_kwargs = {'k': k} ).invoke(query)


# Local tools the agent will have access to
tools = [ search_cv_bank ]
llm_with_tools = llm_global.bind_tools(tools)

conversation_agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt_template
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=conversation_agent, tools=tools, verbose=False)

# Create document loading tools and starategies for different file types
pdf_loader = lambda file_path: PDFMinerLoader(file_path).load()
docx_loader = lambda file_path: Docx2txtLoader(file_path).load()
doc_loader = lambda file_path: UnstructuredWordDocumentLoader(file_path, mode="single").load()
pptx_loader = lambda file_path: UnstructuredPowerPointLoader(file_path, mode="single").load()
text_loader = lambda file_path: TextLoader(file_path).load()

document_loaders = {
    'txt': text_loader,
    'ppt': pptx_loader,
    'pptx': pptx_loader,
    'doc': doc_loader,
    'docx': docx_loader,
    'pdf': pdf_loader
}


# Endpoint for conversation
@app.post('/api/chat')
async def chat_api(conversation: Conversation):
    conversation = [
        { 'role': message.role, 'content': message.content }
            for message in conversation.conversation
    ]
    
    response = agent_executor.invoke({
        'chat_history': conversation[:-1],
        'input': conversation[-1]['content']
    })['output']
    return {'role': 'assistant', 'content': response}

# Endpoint for indexing documents
@app.post('/form/index-documents')
async def index_documents(request: Request, files: List[UploadFile] = File(...)):
    try:
        documents: List[Document] = []
        exceptions = []
        for file in files:
            try:
                # Save the incomming file (local storage is this case, use cloud storage for production)
                with open(f"documents/{file.filename}", "wb") as f:
                    f.write(await file.read())
                
                # Type of the file (pdf, pptx, ppt, docx, doc, txt)
                file_type = file.filename.lower().split('.')[-1]
                candidate_docs = document_loaders.get(
                    file_type,
                    lambda x: [] # return empty list if file type not supported
                )(f"documents/{file.filename}")
                
                # for this use case keep one CV (file) in a single document instead of multiple chunks.
                # Since CV is expected to be small, and accuracy of search will reduce if chunked
                if candidate_docs: # if candidate docs is not empty, merge documents into one
                    documents.append(
                        Document(
                            page_content='\n\n'.join([doc.page_content for doc in candidate_docs]),
                            metadata={
                                "source": urljoin(
                                    str(request.base_url),   # base url of backend server
                                    quote(candidate_docs[0].metadata['source']) # urlencoded file path
                                )
                            }
                        )
                    )
            except Exception as e:
                print(f"Failed to process file {file.filename}", e)
                exceptions.append({'file': file.filename, 'exception': str(e)})
        
        # Push all the CVs to vectorstore (Pinecone in this case)
        pinecone_vectorstore.add_documents(documents)
        return {'status': 'complete', 'success': len(documents), 'failure': len(files) - len(documents), 'exceptions': exceptions}
    except Exception as e:
        print(e)
        return HTTPException(500, detail=f"Error uploading files: {str(e)}")

# Endpoint for serving static files
@app.get("/documents/{filename}")
async def download_file(filename: str):
    file_path = Path(f"documents/{filename}")  # Path to your files directory
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Stream the file content
    async def file_streamer(file_path: str):
        async with aiofiles.open(file_path, mode="rb") as f:
            while chunk := await f.read(1024 * 1024):  # Read 1 MB at a time
                yield chunk
    # Get file size for headers (optional)
    file_size = os.path.getsize(file_path)
    
    return StreamingResponse(
        file_streamer(file_path),
        media_type="application/octet-stream",
        headers={"Content-Length": str(file_size)},  # Optional
    )


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0')
