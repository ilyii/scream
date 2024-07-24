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
model_name = "llama-3-8b-instruct"
# model_name = "mixtral-8x7b-instruct"

system = """
- Du bist ein Spezialist für die Erstellung präziser Zusammenfassungen von Audionachrichten.
- Deine Aufgabe ist es, die Kernaussagen der Nachricht in wenigen Sätzen wiederzugeben, ohne zusätzliche Informationen hinzuzufügen.
- Erstelle die Zusammenfassung aus der Perspektive des Sprechers, also aus der selben Perspektive wie im Transkript gegeben!
- Erwähne nicht, dass es sich um eine Zusammenfassung handelt oder dass du ein KI-Modell bist.
- Die Zusammenfassung soll Stichpunkte enthalten, die die wichtigsten Punkte der Nachricht hervorheben.
"""

human = """
--- Bitte fasse dieses Audiotranskript in deutsch zusammen:

{text}
Zusammenfassung:
"""

summarize_model = None


def load_model_summarize():
    global summarize_model
    # summarize_model = ChatPerplexity(temperature=0, model=model_name, api_key=pplx_api_key)
    summarize_model = OpenAI(base_url="http://localhost:5000/v1", api_key="lm-studio")


def summarize_transcript(transcript):
    # global summarize_model
    # prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # if summarize_model is None:
    #     raise ValueError("Model not loaded.")

    # chain = prompt | summarize_model
    # response = chain.invoke({"text": f"{transcript}"}).content
    # return response
    response = summarize_model.chat.completions.create(
        model="LM Studio Community/Meta-Llama-3-8B-Instruct-GGUF",
        # model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": human.format(text=transcript)},
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content
