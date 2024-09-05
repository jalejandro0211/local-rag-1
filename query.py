import os
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from get_vector_db import get_vector_db

#LLM_MODEL = os.getenv('LLM_MODEL', 'qwen2:1.5b')
LLM_MODEL = "qwen2:1.5b"

# Function to get the prompt templates for generating alternative questions and answering based on context
def get_prompt():
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""Eres un asistente de modelo de lenguaje de IA. 
        Tu tarea es generar una respuesta concreta a la pregunta del usuario para recuperar documentos
        relevantes de una base de datos vectorial. Tu objetivo es ayudar al usuario a superar algunas de las
        limitaciones de la búsqueda de similitud basada en distancia. 
        Siempre debes responder exclusivamente en español. 
        No vas a responder consultas sobre cultura general solo temas de Leasinf Financiero, activos y su  Arrendamiento o uso, tanto temas legasles, financieros y demas peros solo asociado con Leasing
        Si no encuentras la respuesta, solo responde: "La información no está en los documentos suministrados."
        Pregunta original: {question}""",
    )

    # Ajustamos el template para respuestas cortas y estrictamente basadas en el contexto
    template = """Responde la siguiente pregunta basado *exclusivamente* en el contexto proporcionado a continuación.
    Proporciona una respuesta *corta y concisa*. Si no encuentras la respuesta en el contexto, responde:
    "La información no está en los documentos suministrados."
    Contexto: {context}
    Pregunta: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    return QUERY_PROMPT, prompt

# Main function to handle the query process
def query(input):
    if input:
        # Initialize the language model with the specified model name and temperature set to 0
        llm = ChatOllama(model=LLM_MODEL, temperature=0)  # Temperature set to 0 for strict answers
        # Get the vector database instance
        db = get_vector_db()
        # Get the prompt templates
        QUERY_PROMPT, prompt = get_prompt()

        # Set up the retriever to generate multiple queries using the language model and the query prompt
        retriever = MultiQueryRetriever.from_llm(
            db.as_retriever(), 
            llm,
            prompt=QUERY_PROMPT
        )

        # Define the processing chain to retrieve context, generate the answer, and parse the output
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        response = chain.invoke(input)
        
        return response

    return None
