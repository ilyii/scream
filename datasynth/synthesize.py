import argparse
import json
import math
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import librosa
import numpy as np
import soundfile as sf
import yt_dlp
from pydub import AudioSegment
from pytube import YouTube
from tqdm import tqdm
from youtube_transcript_api import NoTranscriptFound, YouTubeTranscriptApi

YTLINK = r"^(https?://)?(www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)$"
VID = r"^([a-zA-Z0-9_-]+)$"


def get_transcript(vid: str, allow_auto_transcript: bool = False):
    transcripts = list()
    try:
        tx = YouTubeTranscriptApi.list_transcripts(vid)
        languages = [t.language_code for t in tx]
        is_generated = [t.is_generated for t in tx]
        scripts = [t.fetch() for t in tx]
    except NoTranscriptFound:
        print(f"{vid}: No transcript found or transcript disabled.")
        return None

    for lg, g, s in zip(languages, is_generated, scripts):
        if not allow_auto_transcript and g:
            continue
        transcripts.append({"text": s, "is_generated": g, "language": lg})

    if len(transcripts) == 0:
        print("No transcript available.")

    return transcripts


def extract_segments(audio_outfile, segment_audio_outfiles):
    audio, sr = librosa.load(audio_outfile, sr=None)

    for segment_audio_outfile in segment_audio_outfiles:
        start_sample = int(segment_audio_outfile["start"] * sr)
        end_sample = int(segment_audio_outfile["end"] * sr)

        segment = audio[start_sample:end_sample]
        sf.write(segment_audio_outfile["outfile"], segment, sr)


def download_audio(url, audio_outfile):
    yt = YouTube(url)

    # download the highest quality audio stream available in mp3 format
    audio = yt.streams.filter(only_audio=True).first()
    audio_name = audio_outfile.split(".")[0]
    audio_dir = os.path.dirname(audio_outfile)
    downloaded_path = audio.download(audio_dir, filename=audio_name, skip_existing=True)
    base, ext = os.path.splitext(downloaded_path)

    audio = AudioSegment.from_file(downloaded_path)
    audio.export(base + ".mp3", format="mp3")

    os.remove(downloaded_path)


def recalculate_durations(transcript):
    # Because mismatched durations can cause issues with segmenting, we recalculate the durations
    # https://github.com/jdepoix/youtube-transcript-api/issues/21

    for i, t in enumerate(transcript["text"]):
        if i == len(transcript["text"]) - 1:
            t["recalc_duration"] = transcript["duration"] - t["start"]
        else:
            t["recalc_duration"] = transcript["text"][i + 1]["start"] - t["start"]

    return transcript


def group(transcript, target_seg_length=10):
    texts, starts, durations = zip(*[(t["text"], t["start"], t["recalc_duration"]) for t in transcript["text"]])

    segments = list()
    segment_length = 0
    start_idx = 0
    for i, start in enumerate(starts):
        cur_duration = durations[i]
        segment_length += cur_duration

        # not only check if the segment is longer than the target length, but also check if it is close enough
        # (shorter with a tolerance of 1 second)
        if segment_length >= target_seg_length or math.isclose(segment_length, target_seg_length, abs_tol=1):
            segment_indices = list(range(start_idx, i + 1))
            new_segment = {
                "start": starts[start_idx],
                "duration": segment_length,
                "text": " ".join(texts[j].replace("\n", " ") for j in segment_indices),
            }
            segments.append(new_segment)

            segment_length = 0
            start_idx = i + 1

    return segments


def synthesize_segment(
    video_id,
    lang,
    destination,
    seg_length,
    num_segments_per_video,
    do_shuffle_segments,
    allow_autotranscript,
    download,
    delete_full_audio_after_segmentation,
    skip_existing,
    language,
):
    output_folder = os.path.join(destination, video_id)
    audio_folder = os.path.join(output_folder, "audios")
    segment_folder = os.path.join(output_folder, "segments")
    transcript_folder = os.path.join(output_folder, "transcripts")

    os.makedirs(audio_folder, exist_ok=True)
    os.makedirs(segment_folder, exist_ok=True)
    os.makedirs(transcript_folder, exist_ok=True)

    audio_outfile = os.path.join(audio_folder, f"{video_id}.mp3")

    if os.path.exists(audio_outfile) and skip_existing:
        return
    url = f"https://www.youtube.com/watch?v={video_id}"

    if download:
        download_audio(url, audio_outfile)

    # Transcript
    transcripts = get_transcript(video_id, allow_autotranscript)

    if transcripts is None:
        return

    # Find the best transcript
    priority_map = {
        (True, False): 1,  # Specified language, not generated
        (True, True): 2,  # Specified language, generated
        (False, False): 3,  # Any language, not generated
        (False, True): 4,  # Any language, generated
    }

    transcripts.sort(key=lambda t: priority_map.get((t["language"] == lang, t["is_generated"]), float("inf")))
    transcript = transcripts[0]
    transcript["duration"] = transcript["text"][-1]["start"] + transcript["text"][-1]["duration"]
    transcript = recalculate_durations(transcript)
    full_video_length = transcript["duration"]

    # Segmenting
    segments = group(transcript, seg_length)
    original_num_segments = len(segments)
    if num_segments_per_video and num_segments_per_video < len(segments):
        indices = (
            np.random.choice(len(segments), num_segments_per_video, replace=False)
            if do_shuffle_segments
            else range(num_segments_per_video)
        )
        segments = [segments[i] for i in indices]
    segment_durations = [s["duration"] for s in segments]

    # collect segment audios to just load the full audio once
    segment_audio_outfiles = []
    for i, segment in enumerate(segments):
        segment_audio_outfile = os.path.join(segment_folder, f"segment_{i:02}.mp3")
        segment_audio_outfiles.append(
            {
                "outfile": segment_audio_outfile,
                "start": segment["start"],
                "end": segment["start"] + segment["duration"],
            }
        )

        segment_transcript_outfile = os.path.join(transcript_folder, f"segment_{i:02}.txt")

        with open(segment_transcript_outfile, "w", encoding="utf-8") as f:
            f.write(segment["text"])

    # extract segments
    extract_segments(audio_outfile, segment_audio_outfiles)

    # write info file for the video
    info_file_path = os.path.join(output_folder, "info.txt")

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

    if delete_full_audio_after_segmentation:
        if os.path.exists(audio_outfile):
            os.remove(audio_outfile)


def synthesize(
    num_workers: int,
    source: str,
    destination: str,
    seg_length: int = 10,
    num_segments_per_video: int = None,
    do_shuffle_segments: bool = False,
    allow_autotranscript: bool = False,
    download: bool = True,
    delete_full_audio_after_segmentation: bool = False,
    skip_existing: bool = True,
    language: str = "en",
):
    if isinstance(source, str) and os.path.exists(source):
        with open(source, "r") as f:
            video_ids = f.read().splitlines()
    elif isinstance(source, list):
        video_ids = source
    else:
        raise ValueError(f"Invalid source: {source}")

    if not os.path.exists(destination):
        print(f"Creating directory: {destination}")
        os.makedirs(destination)

    with tqdm(total=len(video_ids), desc="Processing URLs") as pbar, ThreadPoolExecutor(
        max_workers=num_workers
    ) as executor:
        futures = []
        for video_id in video_ids:
            try:
                video_id, lang = video_id.split(",")
            except ValueError:
                lang = language

            if not re.match(VID, video_id):
                print(f"Invalid VID: {video_id}")
                continue

            futures.append(
                executor.submit(
                    synthesize_segment,
                    video_id,
                    lang,
                    destination,
                    seg_length,
                    num_segments_per_video,
                    do_shuffle_segments,
                    allow_autotranscript,
                    download,
                    delete_full_audio_after_segmentation,
                    skip_existing,
                    language,
                )
            )

        for future in as_completed(futures):
            pbar.update(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-cp" "--cf_path",
        required=False,
        help="Path to the configuration file",
        default="datasynth/synthesize_configs.json",
    )
    args = ap.parse_args()

    if not os.path.exists(args.cp__cf_path):
        raise FileNotFoundError(f"Configuration file not found: {args.cp__cf_path}")

    with open(args.cp__cf_path, "r") as f:
        config = json.load(f)

    synthesize(**config)
