import os
import sys

cur_dir = os.path.dirname(__file__)
sys.path.append(cur_dir)
import pandas as pd
import telebot
from dotenv import load_dotenv
from pydub import AudioSegment
from summarize_utils import load_model_summarize, summarize_transcript
from transcribe_utils import load_model, transcribe_voice

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_API_KEY")

bot = telebot.TeleBot(BOT_TOKEN)
cache_dir = os.path.join(cur_dir, "cache")
os.makedirs(cache_dir, exist_ok=True)


@bot.message_handler(commands=["start", "hello"])
def send_welcome(message):
    send_help(message)


@bot.message_handler(commands=["help", "hilfe", "h"])
def send_help(message):
    first_name = message.from_user.first_name
    response_text = f"""
    Hey {first_name}, ich kann dir dabei helfen, Sprachnachrichten zu transkribieren und zusammenzufassen.
    Um eine Aufgabe zu erledigen, musst du einfach auf eine geeignete Sprachnachricht in diesem Chat antworten und einen der Befehle unten verwenden:
    
    *Befehle:*
    /transcribe oder /t - Transkribiert die Sprachnachricht und gibt den Text zurück.
    /summarize oder /s - Zusammenfassung der Sprachnachricht.
    """

    response_text = response_text.strip().replace("    ", "")

    bot.send_message(message.chat.id, response_text, parse_mode="Markdown")


def check_for_valid_message(message):
    first_name = message.from_user.first_name

    if not message.reply_to_message:
        print("No reply message found.")
        bot.send_message(message.chat.id, f"Hey {first_name}, please reply to a voice message.")
        return False

    if message.reply_to_message.content_type != "voice":
        print("Replied message is not a voice message.")
        bot.send_message(message.chat.id, f"Hey {first_name}, please reply to a voice message.")
        return False

    # check if the voice message is from the bot itself
    if message.reply_to_message.from_user.id == bot.get_me().id:
        bot.send_message(message.chat.id, f"Hey {first_name}, I can't transcribe my own voice messages.")
        return False

    return True


def save_audio(voice_id, chat_dir):
    voice_dir = os.path.join(chat_dir, "voice_messages")
    voice = bot.get_file(voice_id)
    ogg_path = os.path.join(voice_dir, f"{voice.file_id}.ogg")
    wav_path = ogg_path.replace(".ogg", ".wav")

    if os.path.exists(wav_path):
        print("Voice file already exists.")
        with open(wav_path, "rb") as file:
            voice_file = file.read()
    else:
        print("Downloading voice file...")
        voice_file = bot.download_file(voice.file_path)
        with open(ogg_path, "wb") as file:
            file.write(voice_file)
        ogg_audio = AudioSegment.from_ogg(ogg_path)
        ogg_audio.export(wav_path, format="wav")
        os.remove(ogg_path)

    return wav_path


def save_transcript(voice_id, transcript, chat_dir):
    transcript_dir = os.path.join(chat_dir, "transcripts")
    transcript_path = os.path.join(transcript_dir, f"{voice_id}.txt")
    with open(transcript_path, "w") as file:
        file.write(transcript)

    return transcript_path


def load_transcript(voice_id, chat_dir):
    transcript_dir = os.path.join(chat_dir, "transcripts")
    transcript_path = os.path.join(transcript_dir, f"{voice_id}.txt")
    if not os.path.exists(transcript_path):
        return None

    with open(transcript_path, "r") as file:
        transcript = file.read()

    return transcript


def save_summary(voice_id, summary, chat_dir):
    summary_dir = os.path.join(chat_dir, "summaries")
    summary_path = os.path.join(summary_dir, f"{voice_id}.txt")
    with open(summary_path, "w") as file:
        file.write(summary)

    return summary_path


def load_summary(voice_id, chat_dir):
    summary_dir = os.path.join(chat_dir, "summaries")
    summary_path = os.path.join(summary_dir, f"{voice_id}.txt")
    if not os.path.exists(summary_path):
        return None

    with open(summary_path, "r") as file:
        summary = file.read()

    return summary


def create_transcript(voice_id, chat_dir):
    wav_path = save_audio(voice_id, chat_dir)
    transcript = load_transcript(voice_id, chat_dir)
    if transcript is None:
        transcript = transcribe_voice(wav_path)
    else:
        return transcript
    _ = save_transcript(voice_id, transcript, chat_dir)
    return transcript


def create_summary(voice_id, transcript, chat_dir):
    summary = load_summary(voice_id, chat_dir)
    if summary is None:
        summary = summarize_transcript(transcript)
    else:
        return summary
    _ = save_summary(voice_id, summary, chat_dir)
    return summary


def create_chat_dir(chat_id):
    chat_dir = os.path.join(cache_dir, "chats", str(chat_id))
    os.makedirs(chat_dir, exist_ok=True)
    # create some subdirectories
    os.makedirs(os.path.join(chat_dir, "voice_messages"), exist_ok=True)
    os.makedirs(os.path.join(chat_dir, "transcripts"), exist_ok=True)
    os.makedirs(os.path.join(chat_dir, "summaries"), exist_ok=True)
    return chat_dir


@bot.message_handler(commands=["summarize", "s"])
def bot_summarize_transcript(message):
    first_name = message.from_user.first_name

    if not check_for_valid_message(message):
        return
    chat_dir = create_chat_dir(message.chat.id)
    voice_id = message.reply_to_message.voice.file_id

    transcript = create_transcript(voice_id, chat_dir)
    if transcript == "" or transcript is None or transcript.isspace():
        bot.send_message(message.chat.id, f"Hey {first_name}, I couldn't transcribe the voice message.")
        return

    summary = create_summary(voice_id, transcript, chat_dir)
    if summary == "" or summary is None or summary.isspace():
        bot.send_message(message.chat.id, f"Hey {first_name}, I couldn't summarize the voice message.")
        return

    markdown_message = f"📃✨*Zusammenfassung*✨📃\n\n{summary}"
    bot.send_message(message.chat.id, markdown_message, parse_mode="Markdown")


# callback function
# for all replied voice messages with /transcribe command
@bot.message_handler(commands=["transcribe", "t"])
def bot_transcribe_voice(message):
    first_name = message.from_user.first_name

    if not check_for_valid_message(message):
        return
    chat_dir = create_chat_dir(message.chat.id)
    voice_id = message.reply_to_message.voice.file_id

    transcript = create_transcript(voice_id, chat_dir)
    if transcript == "" or transcript is None or transcript.isspace():
        bot.send_message(message.chat.id, f"Hey {first_name}, I couldn't transcribe the voice message.")
        return
    # ret_msg = f"Hey {first_name}, hier ist das Transkript:\n{transcript}"
    # bot.send_message(message.chat.id, ret_msg)

    markdown_message = f"🗣️✨*Transkript*✨️🗣️\n\n{transcript}"
    bot.send_message(message.chat.id, markdown_message, parse_mode="Markdown")


@bot.message_handler(content_types=["voice"])
def all_audio(message):
    if message.from_user.id == bot.get_me().id:
        return

    voice_id = message.voice.file_id

    # first_name = message.from_user.first_name
    chat_dir = create_chat_dir(message.chat.id)
    transcript = create_transcript(voice_id, chat_dir)
    if transcript == "" or transcript is None or transcript.isspace():
        return
    summarize = create_summary(voice_id, transcript, chat_dir)
    if summarize == "" or summarize is None or summarize.isspace():
        return

    print(f"Transcript: {transcript}")
    print(f"Summary: {summarize}")


if __name__ == "__main__":
    load_model(cache_dir)
    load_model_summarize()
    print("Bot is running!")
    bot.infinity_polling()
    print("Bot is stopped!")
