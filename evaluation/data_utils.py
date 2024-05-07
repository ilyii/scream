import os
import re
from ast import literal_eval
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torchaudio as ta
from pandarallel import pandarallel
from torch.utils.data import DataLoader, Dataset


def validate_audio_file(audio_file):
    try:
        ta.info(audio_file)
        return True
    except Exception as e:
        print(f"Error in file {audio_file}: {e}")
        return False


def yt_data_to_df(_path, do_load_transcripts=True, do_validate_audio_files=True):
    # _path must be a directory containing video directories created by the synthesize.py script
    # Each video directory [vid] contains the following directories (+) and files (-):
    # + audios (contains the full audio file as mp3)
    #   - [vid].mp3
    # + segments (contains the audio segments as mp3 files)
    #   - segment_00.mp3
    #   - segment_01.mp3
    #   - ...
    #   - segment_N.mp3
    # + transcripts (contains the transcript as txt file)
    #   - segment_00.txt
    #   - segment_01.txt
    #   - ...
    #   - segment_N.txt
    # - info.txt (contains various information about the video and the segments)
    if not os.path.isdir(_path):
        raise FileNotFoundError(f"Directory '{_path}' not found.")

    video_data = defaultdict(list)
    segment_data = defaultdict(list)

    for vid in os.listdir(_path):
        vid_path = os.path.join(_path, vid)
        if not os.path.isdir(vid_path):
            continue

        info_path = os.path.join(vid_path, "info.txt")
        if not os.path.isfile(info_path):
            continue
        with open(info_path, "r") as f:
            info = f.read().splitlines()

        # create dict from info
        info_dict = {}
        for i in info:
            key, value = i.split(": ")
            info_dict[key] = value.strip()

        video_data["video_id"].append(vid)
        video_data["video_path"].append(vid_path)

        # Extract video information
        # 1. Video URL
        video_data["video_url"].append(info_dict["URL"])

        # 2. language (info[2] = "Language: [language]")
        video_data["language"].append(info_dict["Language"])

        # 3. boolean if transcription is generated (info[3] = "Is generated: [True/False]")
        is_generated = info_dict["Is generated"] == "True"
        video_data["is_generated"].append(is_generated)

        # 4. Number of segments (info[6] = "Number of segments after filtering: [N]")
        num_segments = int(info_dict["Number of segments after filtering"])
        video_data["num_segments"].append(num_segments)

        # 5. Segment durations (info[8] = "Segment durations: [duration]") -> list of floats
        # video_data["segment_durations"].append(info_dict["Segment durations"])
        try:
            segment_durations = literal_eval(info_dict["Segment durations"])
        except Exception as e:
            print(f"Error in video {vid}: {e}\n{info_dict['Segment durations']}")
            segment_durations = [0.0] * num_segments
        video_data["segment_durations"].append(segment_durations)

        # Now save the segment data
        for seg_id in range(num_segments):
            seg_path = os.path.join(vid_path, "segments", f"segment_{seg_id:02d}.mp3")
            trans_path = os.path.join(vid_path, "transcripts", f"segment_{seg_id:02d}.txt")
            if not os.path.isfile(seg_path) or not os.path.isfile(trans_path):
                continue
            segment_duration = segment_durations[seg_id]

            segment_data["video_id"].append(vid)
            segment_data["segment_id"].append(seg_id)
            segment_data["segment_path"].append(seg_path)
            segment_data["transcript_path"].append(trans_path)
            segment_data["segment_duration"].append(segment_duration)

    video_df = pd.DataFrame(video_data)
    segment_df = pd.DataFrame(segment_data)

    # load transcript texts
    if do_load_transcripts:
        pandarallel.initialize(progress_bar=True)
        segment_df["transcript"] = segment_df["transcript_path"].parallel_apply(
            lambda x: open(x, "r", encoding="utf-8").read().strip()
        )

    # validate audio files
    if do_validate_audio_files:
        pandarallel.initialize(progress_bar=True)
        segment_df["valid_audio"] = segment_df["segment_path"].parallel_apply(validate_audio_file)

    yt_df = video_df.merge(segment_df, on="video_id", how="inner")
    return yt_df, video_df, segment_df


def filter_yt_df(
    df,
    min_segment_length=None,
    max_segment_length=None,
    language=None,
    use_auto_generated=None,
    min_words=None,
    max_words=None,
):
    # Filter segments by duration
    if min_segment_length is not None:
        df = df[df["segment_duration"] >= min_segment_length]
    if max_segment_length is not None:
        df = df[df["segment_duration"] <= max_segment_length]

    # Filter by language
    if language is not None:
        df = df[df["language"] == language]

    # Filter by auto-generated transcripts
    if use_auto_generated is not None:
        df = df[df["is_generated"] == use_auto_generated]

    # Filter by number of words
    if "transcript" in df.columns:
        df["num_words"] = df["transcript"].apply(lambda x: len(re.findall(r"\b\w+\b", x)))

    if min_words is not None and "num_words" in df.columns:
        df = df[(df["num_words"] >= min_words)]

    if max_words is not None and "num_words" in df.columns:
        df = df[(df["num_words"] <= max_words)]

    return df.reset_index(drop=True)


class SpeechDataset(Dataset):
    def __init__(
        self,
        df,
        processor,
        processor_args,
        target_sr=16000,
    ):
        self.audio_file_paths = df["segment_path"]
        self.transcripts = df["transcript"]
        self.processor = processor
        self.processor_args = processor_args
        self.target_sr = target_sr

    def __len__(self):
        return len(self.audio_file_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_file_paths[idx]
        transcript = self.transcripts[idx]

        # Load audio
        waveform, sample_rate = ta.load(audio_path)
        if sample_rate != self.target_sr:
            waveform = ta.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sr)(waveform)
        # convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)

        # Process audio and transcript
        try:
            input_features = self.processor(waveform, **self.processor_args).input_features
        except Exception as e:
            input_features = self.processor(waveform, **self.processor_args).input_values

        return {
            "input_features": input_features,
            "labels": self.processor.tokenizer(transcript)["input_ids"],
            "audio_path": audio_path,
            "transcript": transcript,
        }
