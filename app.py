from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from langchain_groq import ChatGroq

import gradio as gr


pdf_path = "pdf/RP.pdf"  
loader = PyPDFLoader(pdf_path)
documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)


embeddings = OllamaEmbeddings(model="nomic-embed-text")


vectorstore = InMemoryVectorStore.from_texts(
    texts=[doc.page_content for doc in chunks],
    embedding=embeddings,
)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})


llm_compression = ChatGroq(
    api_key="votre-cle_api",
    model="llama-3.3-70b-versatile"
)
compressor_llm = LLMChainExtractor.from_llm(llm_compression)
advanced_retriever = ContextualCompressionRetriever(
    base_compressor=compressor_llm,
    base_retriever=base_retriever
)


rag_template = """Rôle : Expert en enseignement supérieur au Maroc (en français).
Mission : Fournir des informations exactes et à jour basées sur des rapports officiels.

Historique : {history}
Connaissances : {context}
Question : {question}

Consignes :
- Répondez uniquement avec les informations disponibles
- En cas de hors-sujet : "Je n'ai pas cette information. Voulez-vous une question sur les universités marocaines ?"
- Ton professionnel et clair
"""

rag_prompt = ChatPromptTemplate.from_template(rag_template)

llm_response = ChatGroq(
    temperature=0,
    api_key="votre-cle_api",
    model="llama-3.3-70b-versatile"
)

memory = ConversationBufferMemory(memory_key="history", return_messages=False)


def retrieve_context(query):
    docs = advanced_retriever.get_relevant_documents(query)
    return "\n\n".join([f"- {doc.page_content}" for doc in docs])

def generate_response(user_message):
    conversation_history = memory.buffer
    context = retrieve_context(user_message)
    prompt = rag_prompt.format(
        history=conversation_history,
        context=context,
        question=user_message
    )
    response = llm_response([HumanMessage(content=prompt)]).content
    memory.save_context({"input": user_message}, {"output": response})
    return response


with gr.Blocks() as demo:
    gr.Markdown("## Consultant en éducation au Maroc")
    chatbot = gr.Chatbot()
    state = gr.State([])

    with gr.Row():
        txt_input = gr.Textbox(placeholder="Posez votre question ici...", show_label=False)
        send_btn = gr.Button("Envoyer")

    def user_interaction(user_message, history):
        response = generate_response(user_message)
        history.append((user_message, response))
        return "", history

    txt_input.submit(user_interaction, [txt_input, state], [txt_input, chatbot])
    send_btn.click(user_interaction, [txt_input, state], [txt_input, chatbot])

demo.launch()
