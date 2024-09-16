import os

import pandas as pd
from pydub import AudioSegment
from summarize_utils import summarize_transcript
from transcribe_utils import transcribe_voice

CHAT_META_COLUMNS = [
    "voice_id",
    "transcript_date",
    "summary_date",
    "transcript_path",
    "summary_path",
    "transcript_words",
    "summary_words",
]


def check_for_valid_message(bot, message):
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


def save_audio(bot, voice_id, chat_dir):
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


def create_transcript(bot, voice_id, chat_dir):
    wav_path = save_audio(bot, voice_id, chat_dir)
    transcript = load_transcript(voice_id, chat_dir)
    if transcript is None:
        transcript = transcribe_voice(wav_path)
    else:
        return transcript
    _ = save_transcript(voice_id, transcript, chat_dir)
    _ = save_to_chat_meta(chat_dir, voice_id, transcript=transcript)
    return transcript


def create_summary(voice_id, transcript, chat_dir):
    summary = load_summary(voice_id, chat_dir)
    if summary is None:
        summary = summarize_transcript(transcript)
    else:
        return summary
    _ = save_summary(voice_id, summary, chat_dir)
    _ = save_to_chat_meta(chat_dir, voice_id, transcript=transcript, summary=summary)
    return summary


def create_chat_dir(cache_dir, chat_id):
    chat_dir = os.path.join(cache_dir, "chats", str(chat_id))
    os.makedirs(chat_dir, exist_ok=True)
    # create some subdirectories
    os.makedirs(os.path.join(chat_dir, "voice_messages"), exist_ok=True)
    os.makedirs(os.path.join(chat_dir, "transcripts"), exist_ok=True)
    os.makedirs(os.path.join(chat_dir, "summaries"), exist_ok=True)
    return chat_dir


def save_to_chat_meta(chat_dir, voice_id, transcript=None, summary=None):
    chat_id = os.path.basename(chat_dir)
    chat_meta_path = os.path.join(chat_dir, f"{chat_id}.csv")

    if not os.path.exists(chat_meta_path):
        chat_meta = pd.DataFrame(columns=CHAT_META_COLUMNS)
        chat_meta.to_csv(chat_meta_path, index=False)

    chat_meta = pd.read_csv(chat_meta_path)

    if transcript is not None:
        transcript_date = pd.Timestamp.now()
        transcript_path = os.path.join(chat_dir, "transcripts", f"{voice_id}.txt")
        transcript_words = len(transcript.split())
    else:
        transcript_date = None
        transcript_path = None
        transcript_words = None

    if summary is not None:
        summary_date = pd.Timestamp.now()
        summary_path = os.path.join(chat_dir, "summaries", f"{voice_id}.txt")
        summary_words = len(summary.split())
    else:
        summary_date = None
        summary_path = None
        summary_words = None

    if voice_id not in chat_meta["voice_id"].values:
        new_row = {
            "voice_id": voice_id,
            "transcript_date": transcript_date,
            "summary_date": summary_date,
            "transcript_path": transcript_path,
            "summary_path": summary_path,
            "transcript_words": transcript_words,
            "summary_words": summary_words,
        }
        new_row = pd.DataFrame([new_row])
        chat_meta = pd.concat([chat_meta, new_row], ignore_index=True)
    else:
        idx = chat_meta[chat_meta["voice_id"] == voice_id].index[0]
        chat_meta.at[idx, "transcript_date"] = transcript_date
        chat_meta.at[idx, "transcript_path"] = transcript_path
        chat_meta.at[idx, "transcript_words"] = transcript_words
        chat_meta.at[idx, "summary_date"] = summary_date
        chat_meta.at[idx, "summary_path"] = summary_path
        chat_meta.at[idx, "summary_words"] = summary_words

    chat_meta.to_csv(chat_meta_path, index=False)
    return chat_meta


def get_chat_meta(chat_dir):
    chat_id = os.path.basename(chat_dir)
    chat_meta_path = os.path.join(chat_dir, f"{chat_id}.csv")

    if not os.path.exists(chat_meta_path):
        return None

    chat_meta = pd.read_csv(chat_meta_path)
    return chat_meta


def transcripts_from(chat_dir, from_date, to_date, just_today=False):
    chat_meta = get_chat_meta(chat_dir)
    if chat_meta is None:
        return None

    # drop rows with missing transcript_date
    chat_meta = chat_meta.dropna(subset=["transcript_date"])
    if len(chat_meta) == 0:
        return None

    chat_meta["transcript_date"] = pd.to_datetime(chat_meta["transcript_date"])

    if just_today:
        today = pd.Timestamp.now().date()
        transcripts = chat_meta[chat_meta["transcript_date"].dt.date == today]
    else:
        transcripts = chat_meta[
            (chat_meta["transcript_date"].dt.date >= from_date) & (chat_meta["transcript_date"].dt.date <= to_date)
        ]

    loaded_transcripts = []
    for idx, row in transcripts.iterrows():
        transcript = load_transcript(row["voice_id"], chat_dir)
        if transcript is not None:
            loaded_transcripts.append(transcript)

    return loaded_transcripts


def summaries_from(chat_dir, from_date, to_date, just_today=False):
    chat_meta = get_chat_meta(chat_dir)
    if chat_meta is None:
        return None

    # drop rows with missing summary_date
    chat_meta = chat_meta.dropna(subset=["summary_date"])
    if len(chat_meta) == 0:
        return None

    chat_meta["summary_date"] = pd.to_datetime(chat_meta["summary_date"])

    if just_today:
        today = pd.Timestamp.now().date()
        summaries = chat_meta[chat_meta["summary_date"].dt.date == today]
    else:
        summaries = chat_meta[
            (chat_meta["summary_date"].dt.date >= from_date) & (chat_meta["summary_date"].dt.date <= to_date)
        ]

    loaded_summaries = []
    for idx, row in summaries.iterrows():
        summary = load_summary(row["voice_id"], chat_dir)
        if summary is not None:
            loaded_summaries.append(summary)

    return loaded_summaries


def get_transcripts_last_7_days(chat_dir):
    today = pd.Timestamp.now().date()
    last_7_days = today - pd.DateOffset(days=7)
    return transcripts_from(chat_dir, last_7_days, today)


def create_summary_today(chat_dir):
    transcripts = transcripts_from(chat_dir, None, None, just_today=True)
    str_transcripts = ""
    for i, transcript in enumerate(transcripts, start=1):
        str_transcripts += f"Transcript {i}:\n{transcript}\n\n"

    return summarize_transcript(str_transcripts, is_multi=True)


def create_summary_7_days(chat_dir):
    transcripts = get_transcripts_last_7_days(chat_dir)
    str_transcripts = ""
    for i, transcript in enumerate(transcripts, start=1):
        str_transcripts += f"Transcript {i}:\n{transcript}\n\n"

    return summarize_transcript(str_transcripts)


def get_summaries_last_7_days(chat_dir):
    today = pd.Timestamp.now().date()
    last_7_days = today - pd.DateOffset(days=7)
    return summaries_from(chat_dir, last_7_days, today)


def get_chat_stats(chat_dir):
    chat_meta = get_chat_meta(chat_dir)
    if chat_meta is None:
        return None

    today = pd.Timestamp.now().date()
    last_7_days = today - pd.DateOffset(days=7)
    last_30_days = today - pd.DateOffset(days=30)

    today_chat_meta = chat_meta[chat_meta["transcript_date"].dt.date == today]
    last_7_days_chat_meta = chat_meta[chat_meta["transcript_date"].dt.date >= last_7_days]
    last_30_days_chat_meta = chat_meta[chat_meta["transcript_date"].dt.date >= last_30_days]

    count_today_transcripts = len(today_chat_meta)
    count_last_7_days_transcripts = len(last_7_days_chat_meta)
    count_last_30_days_transcripts = len(last_30_days_chat_meta)
    count_all_time_transcripts = len(chat_meta)

    today_summaries = today_chat_meta[today_chat_meta["summary_date"].notnull()]
    last_7_days_summaries = last_7_days_chat_meta[last_7_days_chat_meta["summary_date"].notnull()]
    last_30_days_summaries = last_30_days_chat_meta[last_30_days_chat_meta["summary_date"].notnull()]
    all_time_summaries = chat_meta[chat_meta["summary_date"].notnull()]

    count_today_summaries = len(today_summaries)
    count_last_7_days_summaries = len(last_7_days_summaries)
    count_last_30_days_summaries = len(last_30_days_summaries)
    count_all_time_summaries = len(all_time_summaries)

    today_transcript_words = today_chat_meta["transcript_words"].sum()
    last_7_days_transcript_words = last_7_days_chat_meta["transcript_words"].sum()
    last_30_days_transcript_words = last_30_days_chat_meta["transcript_words"].sum()
    all_time_transcript_words = chat_meta["transcript_words"].sum()

    today_summary_words = today_chat_meta["summary_words"].sum()
    last_7_days_summary_words = last_7_days_chat_meta["summary_words"].sum()
    last_30_days_summary_words = last_30_days_chat_meta["summary_words"].sum()
    all_time_summary_words = chat_meta["summary_words"].sum()

    return {
        "today": {
            "transcripts": count_today_transcripts,
            "summaries": count_today_summaries,
            "transcript_words": today_transcript_words,
            "summary_words": today_summary_words,
            "word_difference": today_transcript_words - today_summary_words,
        },
        "last_7_days": {
            "transcripts": count_last_7_days_transcripts,
            "summaries": count_last_7_days_summaries,
            "transcript_words": last_7_days_transcript_words,
            "summary_words": last_7_days_summary_words,
            "word_difference": last_7_days_transcript_words - last_7_days_summary_words,
        },
        "last_30_days": {
            "transcripts": count_last_30_days_transcripts,
            "summaries": count_last_30_days_summaries,
            "transcript_words": last_30_days_transcript_words,
            "summary_words": last_30_days_summary_words,
            "word_difference": last_30_days_transcript_words - last_30_days_summary_words,
        },
        "all_time": {
            "transcripts": count_all_time_transcripts,
            "summaries": count_all_time_summaries,
            "transcript_words": all_time_transcript_words,
            "summary_words": all_time_summary_words,
            "word_difference": all_time_transcript_words - all_time_summary_words,
        },
    }
