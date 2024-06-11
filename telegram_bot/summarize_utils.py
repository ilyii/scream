import os
from getpass import getpass

import pandas as pd
from dotenv import load_dotenv
from IPython.display import Audio
from langchain_community.chat_models import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate

# Example: reuse your existing OpenAI setup
from openai import OpenAI

load_dotenv()

pplx_api_key = os.getenv("PPLX_API_KEY")
# model_name = "llama-3-8b-instruct"
model_name = "mixtral-8x7b-instruct"

system = """
Fasse das unten stehende Audiotranskript zusammen. 
Wenn das Transkript mit den angegebenen Informationen nicht zusammengefasst werden kann, antworte mit "Kann nicht zusammengefasst werden!".

Kontext: Sie sind ein Spezialist, der damit beauftragt ist, Transkripte von Audionachrichten zusammenzufassen. 
Dein Ziel ist es immer, präzise und genaue Zusammenfassungen des Transkripts zu erstellen.
Du solltest jedoch keine Informationen hinzufügen, die nicht in dem Transkript enthalten sind.
Die Zusammenfassung sollte aus wenigen Sätzen bestehen, die die wichtigsten Punkte der Audionachricht wiedergeben.
Ignoriere Rechtschreibfehler oder Grammatikfehler in der Audionachricht und verbessere diese in der Zusammenfassung.
Erwähne nicht, dass es sich um eine Zusammenfassung handelt oder dass du ein Transkript zusammenfasst.
Fasse es aus der Perspektive des Sprechers zusammen.

"""

human = """
--- Bitte fasse dieses Audiotranskript in deutsch zusammen:

{text}
---

Zusammenfassung: 
"""

summarize_model = None


def load_model_summarize():
    global summarize_model
    # summarize_model = ChatPerplexity(temperature=0, model=model_name, api_key=pplx_api_key)
    summarize_model = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")


def summarize_transcript(transcript):
    # global summarize_model
    # prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # if summarize_model is None:
    #     raise ValueError("Model not loaded.")

    # chain = prompt | summarize_model
    # response = chain.invoke({"text": f"{transcript}"}).content
    # return response
    response = summarize_model.chat.completions.create(
        # model="LM Studio Community/Meta-Llama-3-8B-Instruct-GGUF",
        model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": human.format(text=transcript)},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content
