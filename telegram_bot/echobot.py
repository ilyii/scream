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


@bot.message_handler(commands=["summarize", "s"])
def summarize_voice(message):
    bot.send_message(message.chat.id, "Sorry, I can't summarize voice messages yet.")


# @bot.message_handler(commands=["horoscope"])
# def sign_handler(message):
#     text = "What's your zodiac sign?\nChoose one: *Aries*, *Taurus*, *Gemini*, *Cancer,* *Leo*, *Virgo*, *Libra*, *Scorpio*, *Sagittarius*, *Capricorn*, *Aquarius*, and *Pisces*."
#     sent_msg = bot.send_message(message.chat.id, text, parse_mode="Markdown")
#     bot.register_next_step_handler(sent_msg, day_handler)


# def day_handler(message):
#     sign = message.text
#     text = (
#         "What day do you want to know?\nChoose one: *TODAY*, *TOMORROW*, *YESTERDAY*, or a date in format YYYY-MM-DD."
#     )
#     sent_msg = bot.send_message(message.chat.id, text, parse_mode="Markdown")
#     bot.register_next_step_handler(sent_msg, fetch_horoscope, sign.capitalize())


# def fetch_horoscope(message, sign):
#     day = message.text
#     horoscope_message = "Haha, I'm sorry, I don't have that information."
#     bot.send_message(message.chat.id, "Here's your horoscope!")
#     bot.send_message(message.chat.id, horoscope_message, parse_mode="Markdown")


# @bot.message_handler(func=lambda msg: True)
# def echo_all(message):
#     is_reply = message.reply_to_message is not None
#     bot.reply_to(message, f"Hey, you said: {message.text} & is_reply: {is_reply}")


# callback function for all voice messages
@bot.message_handler(content_types=["voice"])
def voice_handler(message):
    # first_name = message.from_user.first_name
    # bot.send_message(message.chat.id, f"Hey {first_name}, you little freak! I can't transcribe voice messages yet.")
    bot_transcribe_voice(message)


def bot_summarize_transcript(transcript):
    return "Sorry, I can't summarize the transcript yet."


# callback function
# for all replied voice messages with /transcribe command
@bot.message_handler(commands=["transcribe", "t"])
def bot_transcribe_voice(message):
    first_name = message.from_user.first_name

    # check if the message is a reply
    if not message.reply_to_message:
        print("No reply message found.")
        bot.send_message(message.chat.id, f"Hey {first_name}, please reply to a voice message.")
        return

    if message.reply_to_message.content_type != "voice":
        print("Replied message is not a voice message.")
        bot.send_message(message.chat.id, f"Hey {first_name}, please reply to a voice message.")
        return

    # check if the voice message is from the bot itself
    if message.reply_to_message.from_user.id == bot.get_me().id:
        bot.send_message(message.chat.id, f"Hey {first_name}, I can't transcribe my own voice messages.")
        return

    voice_dir = os.path.join(cache_dir, "voice_messages")
    if not os.path.exists(voice_dir):
        os.makedirs(voice_dir)

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

    transcript = transcribe_voice(wav_path)
    if transcript == "" or transcript is None or transcript.isspace():
        bot.send_message(message.chat.id, f"Hey {first_name}, I couldn't transcribe the voice message.")
        return
    summarization = summarize_transcript(transcript)
    ret_msg = f"Hey {first_name}, here's the transcription of the voice message:\n{transcript}\n---\n{summarization}"
    bot.send_message(message.chat.id, ret_msg)


if __name__ == "__main__":
    load_model(cache_dir)
    load_model_summarize()
    print("Bot is running!")
    bot.infinity_polling()
    print("Bot is stopped!")
