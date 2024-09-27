import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import librosa
import soundfile as sf
from pydub import AudioSegment
# from pytube import YouTube
from pytubefix import YouTube

from src.text import get_transcript, group_transcript, recalculate_durations
from src.utils import delete_video_files

PRIORITY_MAP = {
    (True, False): 1,  # Specified language, not generated
    (True, True): 2,  # Specified language, generated
    (False, False): 3,  # Any language, not generated
    (False, True): 4,  # Any language, generated
}


def download_audio(url: str, audio_outfile: str) -> int:
    """Downloads audio from YouTube video using PyTube and saves it in mp3 format."""
    try:
        yt = YouTube(url, use_oauth=True, allow_oauth_cache=True)
        audio_stream = yt.streams.filter(only_audio=True).first()

        if not audio_stream:
            logging.error(f"No audio stream found for {url}")
            return -1

        downloaded_path = audio_stream.download(output_path=os.path.dirname(audio_outfile), filename=os.path.splitext(audio_outfile)[0])
        audio = AudioSegment.from_file(downloaded_path)
        audio.export(audio_outfile, format="mp3")
        os.remove(downloaded_path)

        logging.info(f"Downloaded audio for video: {url}")
        return 0
    except Exception as e:
        logging.error(f"Failed to download audio from {url}: {str(e)}")
        return -1


def extract_segments(audio_outfile: str, segment_audio_outfiles: list) -> None:
    """Extracts audio segments based on provided start and end times."""
    try:
        audio, sr = librosa.load(audio_outfile, sr=None)
        for segment in segment_audio_outfiles:
            start_sample = int(segment["start"] * sr)
            end_sample = int(segment["end"] * sr)
            segment_audio = audio[start_sample:end_sample]
            os.makedirs(os.path.dirname(segment["outfile"]), exist_ok=True)
            with open(segment["outfile"], "w", encoding="utf-8") as f:
                sf.write(segment["outfile"], segment_audio, sr)
        logging.info(f"Extracted {len(segment_audio_outfiles)} segments from {audio_outfile}")
    except Exception as e:
        logging.error(f"Error extracting segments from {audio_outfile}: {str(e)}")


def synthesize(**kwargs):
    """Main synthesis function to process video segments."""
    source_path = kwargs.get("source")
    num_workers = kwargs.get("num_workers", 1)
    try:
        with open(source_path, "r") as f:
            video_ids = f.read().splitlines()

        if kwargs.get("multiprocessing"):
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for video_id in video_ids:
                    futures.append(executor.submit(process_video, video_id, **kwargs))

                for future in as_completed(futures):
                    future.result()  # handle results and potential exceptions here
        else:
            for video_id in video_ids:
                process_video(video_id, **kwargs)
    except Exception as e:
        logging.error(f"Error processing video segments: {str(e)}")
        delete_video_files(kwargs.get("destination"))

def process_video(video_id: str, **kwargs):
    """Processes individual video and handles audio and transcript processing."""
    destination = kwargs.get("destination")
    audio_outfile = os.path.join(destination, "audios", f"{video_id}.mp3")

    # Download audio
    url = f"https://www.youtube.com/watch?v={video_id}"
    if download_audio(url, audio_outfile) == -1:

        delete_video_files(os.path.join(destination, video_id))
        return

    # Fetch transcript
    transcripts = get_transcript(video_id, kwargs.get("allow_autotranscript"))
    if not transcripts:
        logging.warning(f"Transcript not found for {video_id}")
        return

    # Process transcript and extract segments
    transcripts.sort(key=lambda t: PRIORITY_MAP.get((t["language"] == kwargs.get("language"), t["is_generated"]), float("inf")))
    transcript = transcripts[0]
    transcript["duration"] = transcript["text"][-1]["start"] + transcript["text"][-1]["duration"]
    transcript = recalculate_durations(transcripts[0])
    full_video_length = transcript["duration"]

    # Filter segments based on duration
    segments = group_transcript(transcript, kwargs.get("seg_length", 10))
    original_num_segments = len(segments)
    segment_audio_outfiles = [
        {"start": seg["start"], "end": seg["start"] + seg["duration"], "outfile": f"{destination}/segments/{video_id}_{i}.mp3"}
        for i, seg in enumerate(segments)
    ]

    # Extract audio segments
    extract_segments(audio_outfile, segment_audio_outfiles)

    info_file_path = os.path.join(destination, "info", f"{video_id}.txt")
    with open(info_file_path, "w", encoding="utf-8") as f:
        f.write(f"Video ID: {video_id}\n")
        f.write(f"URL: {url}\n")
        f.write(f"Language: {transcript['language'] if 'language' in transcript else 'N/A'}\n")
        f.write(f"Is generated: {transcript['is_generated'] if 'is_generated' in transcript else 'N/A'}\n")
        f.write(f"Full video length (seconds): {full_video_length}\n")
        f.write(f"Full video length (minutes): {full_video_length / 60:.2f}\n")
        f.write(f"Number of segments: {original_num_segments if segments else 'N/A'}\n")
        f.write(f"Number of segments after filtering: {len(segments if segments else 'N/A')}\n")
        f.write(f"Target segment length: {seg_length}\n")
        f.write(f"Segment durations: {segment_durations if segment_durations else 'N/A'}\n")
