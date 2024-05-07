import os
import re
from ast import literal_eval
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


def data_to_df(_path):
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
        video_data["is_generated"].append(info_dict["Is generated"])

        # 4. Number of segments (info[6] = "Number of segments after filtering: [N]")
        num_segments = int(info_dict["Number of segments after filtering"])
        video_data["num_segments"].append(num_segments)

        # 5. Segment durations (info[8] = "Segment durations: [duration]") -> list of floats
        # video_data["segment_durations"].append(info_dict["Segment durations"])
        segment_durations = literal_eval(info_dict["Segment durations"])
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
    return video_df, segment_df


class SpeechDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
