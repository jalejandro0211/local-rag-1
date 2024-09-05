import gradio as gr
import requests
import json
import os

def hacer_request(query):
    url = 'http://localhost:8080/query'
    data = {'query': query}
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        json_response = response.json()
        return json_response.get("message", "No se encontró el campo 'message' en la respuesta")
    except requests.exceptions.RequestException as e:
        return f"Error en la solicitud: {str(e)}"
    except json.JSONDecodeError:
        return "Error al decodificar JSON en la respuesta del servidor"

def eliminar_embeddings():
    url = 'http://localhost:8080/delete'
    try:
        response = requests.delete(url)
        response.raise_for_status()
        return json.dumps(response.json(), indent=2)
    except requests.exceptions.RequestException as e:
        return f"Error al eliminar embeddings: {str(e)}"
    except json.JSONDecodeError:
        return "Error al decodificar JSON en la respuesta del servidor"

def cargar_pdf(file):
    if file is None:
        return "No se ha seleccionado ningún archivo."
    
    url = 'http://localhost:8080/embed'
    try:
        file_path = os.path.abspath(file.name)
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'application/pdf')}
            response = requests.post(url, files=files)
            response.raise_for_status()
            return json.dumps(response.json(), indent=2)
    except requests.exceptions.RequestException as e:
        return f"Error al cargar el archivo PDF: {str(e)}"
    except json.JSONDecodeError:
        return "Error al decodificar JSON en la respuesta del servidor"

# Crea la interfaz de Gradio
with gr.Blocks() as iface:
    gr.Markdown("# Interfaz para Consultas, Eliminación de Embeddings y Carga de PDF")
    
    with gr.Row():
        query_input = gr.Textbox(label="Ingresa tu consulta")
        query_output = gr.Textbox(label="Respuesta")
    
    query_button = gr.Button("Enviar Consulta")
    query_button.click(fn=hacer_request, inputs=query_input, outputs=query_output)
    
    delete_button = gr.Button("Eliminar Archivo de Embeddings")
    delete_output = gr.Textbox(label="Resultado de la eliminación")
    delete_button.click(fn=eliminar_embeddings, inputs=None, outputs=delete_output)
    
    pdf_button = gr.Button("Cargar PDF")
    pdf_input = gr.File(label="Selecciona un archivo PDF", file_types=[".pdf"])
    pdf_output = gr.Textbox(label="Resultado de la carga")
    pdf_button.click(fn=cargar_pdf, inputs=pdf_input, outputs=pdf_output)

# Inicia la interfaz
if __name__ == "__main__":
    iface.launch()