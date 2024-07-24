import os
import sys

cur_dir = os.path.dirname(__file__)
sys.path.append(cur_dir)
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


# @bot.message_handler(commands=["summarize", "s"])
# def summarize_voice(message):
#     bot.send_message(message.chat.id, "Sorry, I can't summarize voice messages yet.")


# callback function for all voice messages
@bot.message_handler(content_types=["voice"])
def voice_handler(message):
    first_name = message.from_user.first_name
    bot.send_message(message.chat.id, f"Hey {first_name}, there's a voice message. Please reply to it with a command.")


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


def save_audio(message):
    voice_dir = os.path.join(cache_dir, "voice_messages")
    voice = bot.get_file(message.reply_to_message.voice.file_id)
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


def create_transcript(message):
    wav_path = save_audio(message)
    transcript = transcribe_voice(wav_path)
    return transcript


@bot.message_handler(commands=["summarize", "s"])
def bot_summarize_transcript(message):
    first_name = message.from_user.first_name

    if not check_for_valid_message(message):
        return

    transcript = create_transcript(message)
    if transcript == "" or transcript is None or transcript.isspace():
        bot.send_message(message.chat.id, f"Hey {first_name}, I couldn't transcribe the voice message.")
        return

    summary = summarize_transcript(transcript)
    if summary == "" or summary is None or summary.isspace():
        bot.send_message(message.chat.id, f"Hey {first_name}, I couldn't summarize the voice message.")
        return

    ret_msg = f"Hey {first_name}, here's the summary of the voice message:\n{summary}"
    bot.send_message(message.chat.id, ret_msg)


# callback function
# for all replied voice messages with /transcribe command
@bot.message_handler(commands=["transcribe", "t"])
def bot_transcribe_voice(message):
    first_name = message.from_user.first_name

    if not check_for_valid_message(message):
        return

    transcript = create_transcript(message)
    if transcript == "" or transcript is None or transcript.isspace():
        bot.send_message(message.chat.id, f"Hey {first_name}, I couldn't transcribe the voice message.")
        return
    ret_msg = f"Hey {first_name}, here's the transcription of the voice message:\n{transcript}"
    bot.send_message(message.chat.id, ret_msg)


if __name__ == "__main__":
    load_model(cache_dir)
    load_model_summarize()
    print("Bot is running!")
    bot.infinity_polling()
    print("Bot is stopped!")
