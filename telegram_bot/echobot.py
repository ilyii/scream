"""
Telegram bot for transcribing and summarizing voice messages.

This module implements a Telegram bot that can transcribe voice messages,
summarize them, and provide chat statistics.
"""

import os
import sys
from pathlib import Path

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

# Load environment variables
load_dotenv()

# Constants
BOT_TOKEN = os.getenv("TELEGRAM_BOT_API_KEY")
CUR_DIR = Path(__file__).parent
CACHE_DIR = CUR_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Add current directory to sys.path
sys.path.append(str(CUR_DIR))

# Initialize bot
bot = telebot.TeleBot(BOT_TOKEN, skip_pending=True)


@bot.message_handler(commands=["start", "hello"])
def send_welcome(message):
    """Send a welcome message and help information."""
    send_help(message)


@bot.message_handler(commands=["help", "hilfe", "h"])
def send_help(message):
    """Send help information to the user."""
    first_name = message.from_user.first_name
    response_text = f"""
    Hey {first_name}, ich kann dir dabei helfen, Sprachnachrichten zu transkribieren und zusammenzufassen.
    Um eine Aufgabe zu erledigen, musst du einfach auf eine geeignete Sprachnachricht in diesem Chat antworten und einen der Befehle unten verwenden:
    
    *Befehle:*
    /transcribe oder /t - Transkribiert die Sprachnachricht und gibt den Text zurück.
    /summarize oder /s - Zusammenfassung der Sprachnachricht.
    /today - Zusammenfassung aller Sprachnachrichten, die heute gesendet wurden.
    /stats - Statistiken zu den Sprachnachrichten in diesem Chat.
    """

    response_text = response_text.strip()

    try:
        bot.send_message(message.chat.id, response_text, parse_mode="Markdown")
    except telebot.apihelper.ApiTelegramException:
        bot.send_message(message.chat.id, response_text)


@bot.message_handler(commands=["summarize", "s"])
def bot_summarize_transcript(message):
    """Summarize a replied voice message."""
    first_name = message.from_user.first_name

    if not check_for_valid_message(bot, message):
        return

    chat_dir = create_chat_dir(CACHE_DIR, message.chat.id)
    voice_id = message.reply_to_message.voice.file_id

    transcript = create_transcript(bot, voice_id, chat_dir)
    if not transcript or transcript.isspace():
        bot.send_message(message.chat.id, f"Hey {first_name}, I couldn't transcribe the voice message.")
        return

    summary = create_summary(voice_id, transcript, chat_dir)
    if not summary or summary.isspace():
        bot.send_message(message.chat.id, f"Hey {first_name}, I couldn't summarize the voice message.")
        return

    markdown_message = f"📃✨Zusammenfassung✨📃\n\n{summary}"
    send_message_with_markdown(message.chat.id, markdown_message)


@bot.message_handler(commands=["transcribe", "t"])
def bot_transcribe_voice(message):
    """Transcribe a replied voice message."""
    first_name = message.from_user.first_name

    if not check_for_valid_message(bot, message):
        return

    chat_dir = create_chat_dir(CACHE_DIR, message.chat.id)
    voice_id = message.reply_to_message.voice.file_id

    transcript = create_transcript(bot, voice_id, chat_dir)
    if not transcript or transcript.isspace():
        bot.send_message(message.chat.id, f"Hey {first_name}, I couldn't transcribe the voice message.")
        return

    markdown_message = f"🗣️✨Transkript✨️🗣️\n\n{transcript}"
    send_message_with_markdown(message.chat.id, markdown_message)


@bot.message_handler(content_types=["voice"])
def all_audio(message):
    """Handle all incoming voice messages."""
    if message.from_user.id == bot.get_me().id:
        return

    voice_id = message.voice.file_id
    chat_dir = create_chat_dir(CACHE_DIR, message.chat.id)

    transcript = create_transcript(bot, voice_id, chat_dir)
    if not transcript or transcript.isspace():
        return

    summarize = create_summary(voice_id, transcript, chat_dir)
    if not summarize or summarize.isspace():
        return

    sender = message.from_user.first_name

    markdown_message = f"""
🗣️ {sender} hat eine Sprachnachricht gesendet 🗣️

🗣️✨Transkript✨️🗣️

{transcript}

📃✨Zusammenfassung✨📃

{summarize}
    """
    send_message_with_markdown(message.chat.id, markdown_message)


@bot.message_handler(commands=["today"])
def bot_summarize_today(message):
    """Summarize all voice messages sent today."""
    chat_id = message.chat.id
    chat_dir = create_chat_dir(CACHE_DIR, chat_id)
    summary = create_summary_today(chat_dir)

    if not summary or summary.isspace():
        bot.send_message(chat_id, "I couldn't summarize the voice messages.")
        return

    markdown_message = f"📃✨Zusammenfassung heutiger Sprachnachrichten✨📃\n\n{summary}"
    send_message_with_markdown(chat_id, markdown_message)


@bot.message_handler(commands=["stats"])
def bot_show_stats(message):
    """Show statistics for the chat."""
    chat_id = message.chat.id
    chat_dir = create_chat_dir(CACHE_DIR, chat_id)
    stats = get_chat_stats(chat_dir)

    if not stats:
        bot.send_message(chat_id, "I couldn't get the stats.")
        return

    markdown_message = "📊✨Statistiken✨📊\n\n"
    for key, value in stats.items():
        markdown_message += f" {key}:\n"
        for k, v in value.items():
            markdown_message += f"        {k}: {v}\n"
        markdown_message += "\n"

    send_message_with_markdown(chat_id, markdown_message)


def send_message_with_markdown(chat_id, message):
    """Send a message with Markdown formatting, falling back to plain text if it fails."""
    try:
        bot.send_message(chat_id, message, parse_mode="Markdown")
    except telebot.apihelper.ApiTelegramException:
        bot.send_message(chat_id, message)


if __name__ == "__main__":
    load_model(CACHE_DIR)
    load_model_summarize()
    print("Bot is running!")
    bot.infinity_polling()
    print("Bot is stopped!")
