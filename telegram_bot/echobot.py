import os
import sys

cur_dir = os.path.dirname(__file__)
sys.path.append(cur_dir)
import telebot
from dotenv import load_dotenv
from summarize_utils import load_model_summarize
from transcribe_utils import load_model

from utils import (
    check_for_valid_message,
    create_chat_dir,
    create_summary,
    create_summary_today,
    create_transcript,
    get_chat_stats,
)

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


@bot.message_handler(commands=["summarize", "s"])
def bot_summarize_transcript(message):
    first_name = message.from_user.first_name

    if not check_for_valid_message(bot, message):
        return
    chat_dir = create_chat_dir(cache_dir, message.chat.id)
    voice_id = message.reply_to_message.voice.file_id

    transcript = create_transcript(bot, voice_id, chat_dir)
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

    if not check_for_valid_message(bot, message):
        return
    chat_dir = create_chat_dir(cache_dir, message.chat.id)
    voice_id = message.reply_to_message.voice.file_id

    transcript = create_transcript(bot, voice_id, chat_dir)
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
    chat_dir = create_chat_dir(cache_dir, message.chat.id)
    transcript = create_transcript(bot, voice_id, chat_dir)
    if transcript == "" or transcript is None or transcript.isspace():
        return
    summarize = create_summary(voice_id, transcript, chat_dir)
    if summarize == "" or summarize is None or summarize.isspace():
        return


@bot.message_handler(commands=["today"])
def bot_summarize_today(message):
    chat_id = message.chat.id
    chat_dir = create_chat_dir(cache_dir, chat_id)
    summary = create_summary_today(chat_dir)
    if summary == "" or summary is None or summary.isspace():
        bot.send_message(chat_id, "I couldn't summarize the voice messages.")
        return

    markdown_message = f"📃✨*Zusammenfassung heutiger Sprachnachrichten*✨📃\n\n{summary}"
    bot.send_message(chat_id, markdown_message, parse_mode="Markdown")


@bot.message_handler(commands=["stats"])
def bot_show_stats(message):
    chat_id = message.chat.id
    chat_dir = create_chat_dir(cache_dir, chat_id)
    stats = get_chat_stats(chat_dir)
    if stats == "" or stats is None or stats.isspace():
        bot.send_message(chat_id, "I couldn't get the stats.")
        return

    markdown_message = f"📊✨*Statistiken*✨📊\n\n"
    for key, value in stats.items():
        markdown_message += f"{key}: {value}\n"

    bot.send_message(chat_id, markdown_message, parse_mode="Markdown")


if __name__ == "__main__":
    load_model(cache_dir)
    load_model_summarize()
    print("Bot is running!")
    bot.infinity_polling()
    print("Bot is stopped!")
