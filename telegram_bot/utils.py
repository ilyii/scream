"""
Utility functions for managing voice message transcripts and summaries.

This module provides functions for handling voice messages, transcripts, and summaries
in a Telegram bot context. It includes functionality for file management, data persistence,
and statistical analysis of chat data.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
from pydub import AudioSegment
from pandas.tseries.offsets import DateOffset

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


def check_for_valid_message(bot: Any, message: Any) -> bool:
    """
    Check if the replied message is a valid voice message.

    Args:
        bot: The Telegram bot instance.
        message: The message to check.

    Returns:
        bool: True if the message is valid, False otherwise.
    """
    first_name = message.from_user.first_name

    if not message.reply_to_message:
        print("No reply message found.")
        bot.send_message(message.chat.id, f"Hey {first_name}, please reply to a voice message.")
        return False

    if message.reply_to_message.content_type != "voice":
        print("Replied message is not a voice message.")
        bot.send_message(message.chat.id, f"Hey {first_name}, please reply to a voice message.")
        return False

    if message.reply_to_message.from_user.id == bot.get_me().id:
        bot.send_message(message.chat.id, f"Hey {first_name}, I can't transcribe my own voice messages.")
        return False

    return True


def save_audio(bot: Any, voice_id: str, chat_dir: str) -> str:
    """
    Save the audio file from a voice message.

    Args:
        bot: The Telegram bot instance.
        voice_id: The ID of the voice message.
        chat_dir: The directory for the chat.

    Returns:
        str: The path to the saved WAV file.
    """
    voice_dir = Path(chat_dir) / "voice_messages"
    voice_dir.mkdir(parents=True, exist_ok=True)

    voice = bot.get_file(voice_id)
    wav_path = voice_dir / f"{voice.file_id}.wav"

    if wav_path.exists():
        print("Voice file already exists.")
        return str(wav_path)

    print("Downloading voice file...")
    voice_file = bot.download_file(voice.file_path)
    ogg_path = voice_dir / f"{voice.file_id}.ogg"

    with ogg_path.open("wb") as file:
        file.write(voice_file)

    ogg_audio = AudioSegment.from_ogg(str(ogg_path))
    ogg_audio.export(str(wav_path), format="wav")
    ogg_path.unlink()

    return str(wav_path)


def save_transcript(voice_id: str, transcript: str, chat_dir: str) -> str:
    """
    Save the transcript of a voice message.

    Args:
        voice_id: The ID of the voice message.
        transcript: The transcript text.
        chat_dir: The directory for the chat.

    Returns:
        str: The path to the saved transcript file.
    """
    transcript_dir = Path(chat_dir) / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = transcript_dir / f"{voice_id}.txt"

    with transcript_path.open("w", encoding="utf-8") as file:
        file.write(transcript)

    return str(transcript_path)


def load_transcript(voice_id: str, chat_dir: str) -> Optional[str]:
    """
    Load the transcript of a voice message.

    Args:
        voice_id: The ID of the voice message.
        chat_dir: The directory for the chat.

    Returns:
        Optional[str]: The transcript text if it exists, None otherwise.
    """
    transcript_path = Path(chat_dir) / "transcripts" / f"{voice_id}.txt"

    if not transcript_path.exists():
        return None

    with transcript_path.open("r", encoding="utf-8") as file:
        return file.read()


def save_summary(voice_id: str, summary: str, chat_dir: str) -> str:
    """
    Save the summary of a voice message.

    Args:
        voice_id: The ID of the voice message.
        summary: The summary text.
        chat_dir: The directory for the chat.

    Returns:
        str: The path to the saved summary file.
    """
    summary_dir = Path(chat_dir) / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{voice_id}.txt"

    with summary_path.open("w", encoding="utf-8") as file:
        file.write(summary)

    return str(summary_path)


def load_summary(voice_id: str, chat_dir: str) -> Optional[str]:
    """
    Load the summary of a voice message.

    Args:
        voice_id: The ID of the voice message.
        chat_dir: The directory for the chat.

    Returns:
        Optional[str]: The summary text if it exists, None otherwise.
    """
    summary_path = Path(chat_dir) / "summaries" / f"{voice_id}.txt"

    if not summary_path.exists():
        return None

    with summary_path.open("r", encoding="utf-8") as file:
        return file.read()


def create_transcript(bot: Any, voice_id: str, chat_dir: str) -> str:
    """
    Create or load a transcript for a voice message.

    Args:
        bot: The Telegram bot instance.
        voice_id: The ID of the voice message.
        chat_dir: The directory for the chat.

    Returns:
        str: The transcript text.
    """
    wav_path = save_audio(bot, voice_id, chat_dir)
    transcript = load_transcript(voice_id, chat_dir)

    if transcript is None:
        transcript = transcribe_voice(wav_path)
        save_transcript(voice_id, transcript, chat_dir)
        save_to_chat_meta(chat_dir, voice_id, transcript=transcript)

    return transcript


def create_summary(voice_id: str, transcript: str, chat_dir: str) -> str:
    """
    Create or load a summary for a voice message.

    Args:
        voice_id: The ID of the voice message.
        transcript: The transcript text.
        chat_dir: The directory for the chat.

    Returns:
        str: The summary text.
    """
    summary = load_summary(voice_id, chat_dir)

    if summary is None:
        summary = summarize_transcript(transcript)
        save_summary(voice_id, summary, chat_dir)
        save_to_chat_meta(chat_dir, voice_id, transcript=transcript, summary=summary)

    return summary


def create_chat_dir(cache_dir: str, chat_id: int) -> str:
    """
    Create the directory structure for a chat.

    Args:
        cache_dir: The base cache directory.
        chat_id: The ID of the chat.

    Returns:
        str: The path to the created chat directory.
    """
    chat_dir = Path(cache_dir) / "chats" / str(chat_id)
    for subdir in ["voice_messages", "transcripts", "summaries"]:
        (chat_dir / subdir).mkdir(parents=True, exist_ok=True)
    return str(chat_dir)


def save_to_chat_meta(
    chat_dir: str, voice_id: str, transcript: Optional[str] = None, summary: Optional[str] = None
) -> pd.DataFrame:
    """
    Save metadata about a voice message to the chat's metadata file.

    Args:
        chat_dir: The directory for the chat.
        voice_id: The ID of the voice message.
        transcript: The transcript text (optional).
        summary: The summary text (optional).

    Returns:
        pd.DataFrame: The updated chat metadata.
    """
    chat_id = Path(chat_dir).name
    chat_meta_path = Path(chat_dir) / f"{chat_id}.csv"

    if not chat_meta_path.exists():
        chat_meta = pd.DataFrame(columns=CHAT_META_COLUMNS)
    else:
        chat_meta = pd.read_csv(chat_meta_path)

    new_row = {
        "voice_id": voice_id,
        "transcript_date": pd.Timestamp.now() if transcript else None,
        "summary_date": pd.Timestamp.now() if summary else None,
        "transcript_path": str(Path(chat_dir) / "transcripts" / f"{voice_id}.txt") if transcript else None,
        "summary_path": str(Path(chat_dir) / "summaries" / f"{voice_id}.txt") if summary else None,
        "transcript_words": len(transcript.split()) if transcript else None,
        "summary_words": len(summary.split()) if summary else None,
    }

    if voice_id not in chat_meta["voice_id"].values:
        chat_meta = pd.concat([chat_meta, pd.DataFrame([new_row])], ignore_index=True)
    else:
        idx = chat_meta[chat_meta["voice_id"] == voice_id].index[0]
        chat_meta.loc[idx, new_row.keys()] = new_row.values()

    chat_meta.to_csv(chat_meta_path, index=False)
    return chat_meta


def get_chat_meta(chat_dir: str) -> Optional[pd.DataFrame]:
    """
    Load the metadata for a chat.

    Args:
        chat_dir: The directory for the chat.

    Returns:
        Optional[pd.DataFrame]: The chat metadata if it exists, None otherwise.
    """
    chat_id = Path(chat_dir).name
    chat_meta_path = Path(chat_dir) / f"{chat_id}.csv"

    if not chat_meta_path.exists():
        return None

    return pd.read_csv(chat_meta_path)


def transcripts_from(
    chat_dir: str, from_date: Optional[pd.Timestamp], to_date: Optional[pd.Timestamp], just_today: bool = False
) -> Optional[List[str]]:
    """
    Get transcripts from a specified date range.

    Args:
        chat_dir: The directory for the chat.
        from_date: The start date for the range.
        to_date: The end date for the range.
        just_today: If True, only get transcripts from today.

    Returns:
        Optional[List[str]]: A list of transcripts, or None if no transcripts are found.
    """
    chat_meta = get_chat_meta(chat_dir)
    if chat_meta is None or chat_meta.empty:
        return None

    chat_meta = chat_meta.dropna(subset=["transcript_date"])
    chat_meta["transcript_date"] = pd.to_datetime(chat_meta["transcript_date"])

    if just_today:
        today = pd.Timestamp.now().date()
        transcripts = chat_meta[chat_meta["transcript_date"].dt.date == today]
    else:
        transcripts = chat_meta[
            (chat_meta["transcript_date"].dt.date >= from_date) & (chat_meta["transcript_date"].dt.date <= to_date)
        ]

    return [
        load_transcript(row["voice_id"], chat_dir)
        for _, row in transcripts.iterrows()
        if load_transcript(row["voice_id"], chat_dir) is not None
    ]


def summaries_from(
    chat_dir: str, from_date: Optional[pd.Timestamp], to_date: Optional[pd.Timestamp], just_today: bool = False
) -> Optional[List[str]]:
    """
    Get summaries from a specified date range.

    Args:
        chat_dir: The directory for the chat.
        from_date: The start date for the range.
        to_date: The end date for the range.
        just_today: If True, only get summaries from today.

    Returns:
        Optional[List[str]]: A list of summaries, or None if no summaries are found.
    """
    chat_meta = get_chat_meta(chat_dir)
    if chat_meta is None or chat_meta.empty:
        return None

    chat_meta = chat_meta.dropna(subset=["summary_date"])
    chat_meta["summary_date"] = pd.to_datetime(chat_meta["summary_date"])

    if just_today:
        today = pd.Timestamp.now().date()
        summaries = chat_meta[chat_meta["summary_date"].dt.date == today]
    else:
        summaries = chat_meta[
            (chat_meta["summary_date"].dt.date >= from_date) & (chat_meta["summary_date"].dt.date <= to_date)
        ]

    return [
        load_summary(row["voice_id"], chat_dir)
        for _, row in summaries.iterrows()
        if load_summary(row["voice_id"], chat_dir) is not None
    ]


def get_transcripts_last_7_days(chat_dir: str) -> Optional[List[str]]:
    """
    Get transcripts from the last 7 days.

    Args:
        chat_dir: The directory for the chat.

    Returns:
        Optional[List[str]]: A list of transcripts from the last 7 days, or None if no transcripts are found.
    """
    today = pd.Timestamp.now().date()
    last_7_days = today - pd.DateOffset(days=7)
    return transcripts_from(chat_dir, last_7_days, today)


def create_summary_today(chat_dir: str) -> Optional[str]:
    """
    Create a summary of all transcripts from today.

    Args:
        chat_dir: The directory for the chat.

    Returns:
        Optional[str]: A summary of today's transcripts, or None if no transcripts are found.
    """
    transcripts = transcripts_from(chat_dir, None, None, just_today=True)
    if not transcripts:
        return None

    str_transcripts = "\n\n".join(f"Transcript {i}:\n{transcript}" for i, transcript in enumerate(transcripts, start=1))
    return summarize_transcript(str_transcripts, is_multi=True)


def create_summary_7_days(chat_dir: str) -> Optional[str]:
    """
    Create a summary of all transcripts from the last 7 days.

    Args:
        chat_dir: The directory for the chat.

    Returns:
        Optional[str]: A summary of the last 7 days' transcripts, or None if no transcripts are found.
    """
    transcripts = get_transcripts_last_7_days(chat_dir)
    if not transcripts:
        return None

    str_transcripts = "\n\n".join(f"Transcript {i}:\n{transcript}" for i, transcript in enumerate(transcripts, start=1))
    return summarize_transcript(str_transcripts)


def get_summaries_last_7_days(chat_dir: str) -> Optional[List[str]]:
    """
    Get summaries from the last 7 days.

    Args:
        chat_dir: The directory for the chat.

    Returns:
        Optional[List[str]]: A list of summaries from the last 7 days, or None if no summaries are found.
    """
    today = pd.Timestamp.now().date()
    last_7_days = today - pd.DateOffset(days=7)
    return summaries_from(chat_dir, last_7_days, today)


def get_chat_stats(chat_dir: str) -> Optional[Dict[str, Dict[str, int]]]:
    """
    Get statistics for the chat.

    Args:
        chat_dir: The directory for the chat.

    Returns:
        Optional[Dict[str, Dict[str, int]]]: A dictionary containing chat statistics,
        or None if no data is available.
    """
    chat_meta = get_chat_meta(chat_dir)
    if chat_meta is None or chat_meta.empty:
        return None

    today = pd.Timestamp.now().normalize()
    last_7_days = today - DateOffset(days=7)
    last_30_days = today - DateOffset(days=30)

    chat_meta["transcript_date"] = pd.to_datetime(chat_meta["transcript_date"]).dt.normalize()
    chat_meta["summary_date"] = pd.to_datetime(chat_meta["summary_date"]).dt.normalize()

    def get_period_stats(df: pd.DataFrame) -> Dict[str, int]:
        """
        Calculate statistics for a given period.

        Args:
            df: DataFrame containing the data for the period.

        Returns:
            Dict[str, int]: A dictionary of statistics for the period.
        """
        return {
            "transcripts": len(df),
            "summaries": df["summary_date"].notna().sum(),
            "transcript_words": int(df["transcript_words"].sum()),
            "summary_words": int(df["summary_words"].sum()),
        }

    return {
        "today": get_period_stats(chat_meta[chat_meta["transcript_date"] == today]),
        "last_7_days": get_period_stats(chat_meta[chat_meta["transcript_date"] >= last_7_days]),
        "last_30_days": get_period_stats(chat_meta[chat_meta["transcript_date"] >= last_30_days]),
        "all_time": get_period_stats(chat_meta),
    }
