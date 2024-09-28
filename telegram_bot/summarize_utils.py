"""
Module for summarizing transcripts using Google's Generative AI (Gemini).

This module provides functionality to load the Gemini model and summarize
single or multiple transcripts.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Constants
MODEL_NAME = "gemini-1.5-flash"
API_KEY = os.getenv("GEMINI_API_KEY")

# Prompt templates
SYSTEM_SINGLE = """
You specialize in creating concise summaries of audio messages.
Your task is to capture the core points of the message in as few sentences as possible, without adding any extra information.
Write the summary from the speaker's perspective, exactly as conveyed in the transcript.
Do not mention that this is a summary or that you are an AI.
Use bullet points to highlight the key information.
Keep the summary as short as possible while retaining all essential details.
The summary must be in German.

Example:
Input: "Ich habe heute eine wichtige Besprechung um 14 Uhr, in der geht es um Carla. Carla ist die neue Kollegin, die wir eingestellt haben. Sie hat eine Menge Erfahrung und ich denke, sie wird eine Bereicherung für unser Team sein. Jeder sollte pünktlich sein, damit wir die Besprechung beginnen können. Außerdem gehe ich davon aus, dass wir mindestens eine Stunde brauchen werden, um alles zu besprechen. Später gehe ich noch zum Sport und trainiere für den Marathon. Was soll ich denn heute Abend kochen? Brauchen wir noch etwas vom Supermarkt?"
Summary:
"● Wichtige Besprechung um 14 Uhr, Thema: Carla, neue Kollegin mit viel Erfahrung und Bereicherung fürs Team. Pünktlichkeit wichtig. Dauer mind. 1 Stunde.
● Danach Marathontraining.
● Was soll ich heute Abend kochen?
● Muss noch etwas vom Supermarkt geholt werden?"
"""

USER_SINGLE = """
Here is the transcript of the audio message:
{input}

---

Summary:
"""

SYSTEM_MULTI = """
You are a specialist in creating concise summaries of multiple audio messages.
Your task is to summarize the most important points from ALL messages, providing ONE consolidated summary.
Make sure to indicate which message each piece of information comes from (e.g., "Message 1: Summary of Message 1", "Message 2: Summary of Message 2").
Then, provide an overall summary combining the key points from all messages.
The summary must be written in German and should use bullet points where necessary.
"""

USER_MULTI = """
Below are multiple transcripts from audio messages.
Summarize the key points from ALL the messages and provide ONE summary.
Be sure to indicate which message each piece of information comes from (e.g., "Message 1: Summary of Message 1", "Message 2: Summary of Message 2").
After that, provide an overall summary of all messages.

---

Transcripts:
{input}

---

Summary:
"""

# Create prompt templates
prompt_single = ChatPromptTemplate.from_messages([("system", SYSTEM_SINGLE), ("human", USER_SINGLE)])
prompt_multi = ChatPromptTemplate.from_messages([("system", SYSTEM_MULTI), ("human", USER_MULTI)])

# Global variable for the model
summarize_model: Optional[ChatGoogleGenerativeAI] = None


def load_model_summarize() -> None:
    """
    Load the Gemini model for summarization.

    This function initializes the global summarize_model variable with the Gemini model.

    Raises:
        ValueError: If the GEMINI_API_KEY is not set in the environment variables.
    """
    global summarize_model
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

    summarize_model = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0.2,
        max_tokens=1024,
        google_api_key=API_KEY,
        timeout=None,
        max_retries=2,
    )


def summarize_transcript(transcript: str, is_multi: bool = False) -> str:
    """
    Summarize a transcript or multiple transcripts.

    Args:
        transcript (str): The transcript(s) to summarize.
        is_multi (bool, optional): Whether the input contains multiple transcripts. Defaults to False.

    Returns:
        str: The summarized transcript(s).

    Raises:
        ValueError: If the model hasn't been loaded.
        RuntimeError: If there's an error during summarization.
    """
    global summarize_model
    if summarize_model is None:
        load_model_summarize()

    prompt = prompt_multi if is_multi else prompt_single
    chain = prompt | summarize_model

    try:
        return chain.invoke({"input": transcript}).content
    except Exception as e:
        raise RuntimeError(f"Error summarizing transcript: {str(e)}")
