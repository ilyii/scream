import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model_name = "gemini-1.5-flash"

system = """
- Du bist ein Spezialist für die Erstellung präziser Zusammenfassungen von Audionachrichten.
- Deine Aufgabe ist es, die Kernaussagen der Nachricht in wenigen Sätzen wiederzugeben, ohne zusätzliche Informationen hinzuzufügen.
- Erstelle die Zusammenfassung aus der Perspektive des Sprechers, also aus der selben Perspektive wie im Transkript gegeben!
- Erwähne nicht, dass es sich um eine Zusammenfassung handelt oder dass du ein KI-Modell bist.
- Die Zusammenfassung soll Stichpunkte enthalten, die die wichtigsten Punkte der Nachricht hervorheben.
- Die Zusammenfassung soll in deutscher Sprache verfasst sein.
"""

user_single = """
Das Transkript der Audionachricht lautet:
{input}

---

Zusammenfassung:
"""

user_multi = """
Folgend erhälst du mehrere Transkripte von Audionachrichten.
Fasse die wichtigsten Punkten aus ALLEN Nachrichten zusammen und gib EINE Zusammenfassung zurück.
Gib unbedingt an, aus welcher Nachricht jeweils die Informationen stammen!!! (z.B. "Nachricht 1: Zusammenfassung Nachricht 1", "Nachricht 2: ...")
Gib danach aber noch eine allgemeine Zusammenfassung aller Nachrichten an.

---

Transkripte:
{input}

---

Zusammenfassung:
"""

prompt_single = ChatPromptTemplate.from_messages([("system", system), ("human", user_single)])
prompt_multi = ChatPromptTemplate.from_messages([("system", system), ("human", user_multi)])
summarize_model = None


def load_model_summarize():
    global summarize_model
    summarize_model = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.3,
        max_tokens=2048,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        timeout=None,
        max_retries=2,
    )


def summarize_transcript(transcript, is_multi=False):
    global summarize_model
    if summarize_model is None:
        load_model_summarize()

    if is_multi:
        prompt = prompt_multi
    else:
        prompt = prompt_single
    chain = prompt | summarize_model
    return chain.invoke({"input": transcript}).content
