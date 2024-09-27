import argparse
import os
import shutil
import random
import pandas as pd
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import utils

print("Imported utils from:", os.path.abspath(utils.__file__))

def create_dataset(root_folder, audio_paths, transcript_files, output_dir, splits: list=[0.8, 0.1, 0.1]):
    
    os.makedirs(output_dir, exist_ok=True)

    assert len(splits) <=3, "splits must be a list of length 1, 2 or 3."
    assert len(audio_paths) == len(transcript_files), f"Number of audio files ({len(audio_paths)}) and transcript files ({len(transcript_files)}) do not match."

    combined = list(zip(audio_paths, transcript_files))

    if len(splits) == 1:
        splitnames = ['data']
        data = [combined]
        
        
    elif len(splits) == 2:
        splitnames = ['train', 'test']
        random.shuffle(combined)
        train_data = combined[:int(splits[0]*len(combined))]
        test_data = combined[int(splits[0]*len(combined)):]

        data = [train_data, test_data]
        
    elif len(splits) == 3:
        splitnames = ['train', 'val', 'test']
        random.shuffle(combined)
        train_data = combined[:int(splits[0]*len(combined))]
        vaL_data = combined[int(splits[0]*len(combined)):int((splits[0]+splits[1])*len(combined))]
        test_data = combined[int((splits[0]+splits[1])*len(combined)):]

        data = [train_data, vaL_data, test_data]

    for split, data in zip(splitnames, data):
        audio_paths, transcript_files = zip(*data)
        metadata = []
        split_folder = os.path.join(output_dir, split)
        os.makedirs(split_folder, exist_ok=True)

        for audio_path, tranascript_path in zip(audio_paths, transcript_files):    
            vid = os.path.basename(os.path.dirname(os.path.dirname(audio_path)))
            new_audio_path = os.path.join(split_folder, f'{vid}_{os.path.basename(audio_path)}')   
            shutil.copy(audio_path, new_audio_path)
            with open(tranascript_path, 'r', encoding='utf-8') as f:
                transcript_txt = f.read().strip()
            metadata.append([os.path.basename(new_audio_path), transcript_txt])
        metadata_df = pd.DataFrame(metadata, columns=['file_name', 'transcription'])
        metadata_df.to_csv(os.path.join(split_folder, 'metadata.csv'), index=False)



def load_audio_and_transcripts(root_folder):
    all_audio_files = []
    all_transcript_files = []

    for vid in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, vid)
        if os.path.isdir(folder_path):
            audio_files = []
            transcript_files = []

            segments_folder = os.path.join(folder_path, 'segments')
            transcripts_folder = os.path.join(folder_path, 'transcripts')

            for filename in sorted(os.listdir(segments_folder)):
                if filename.endswith('.mp3'):
                    audio_files.append(os.path.join(segments_folder, filename))

            for filename in sorted(os.listdir(transcripts_folder)):
                if filename.endswith('.txt'):
                    transcript_files.append(os.path.join(transcripts_folder, filename))

            all_audio_files.extend(audio_files)
            all_transcript_files.extend(transcript_files)

    return all_audio_files, all_transcript_files


def split_data(folder_path, train_percentage, val_percentage, test_percentage):
    train_folder = os.path.join(folder_path, 'train')
    data_folder = os.path.join(folder_path, 'data')

    os.makedirs(data_folder, exist_ok=True)

    metadata_df = pd.read_csv(os.path.join(train_folder, 'metadata.csv'))
    num_files = len(metadata_df)
    indices = list(range(num_files))
    random.shuffle(indices)

    train_size = int(train_percentage * num_files)
    val_size = int(val_percentage * num_files)
    test_size = num_files - train_size - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]

    for index, split_name in zip([train_indices, val_indices, test_indices], ['train', 'val', 'test']):
        split_folder = os.path.join(data_folder, split_name)
        os.makedirs(split_folder, exist_ok=True)
        for i in index:
            audio_file = metadata_df.iloc[i]['file_name']
            shutil.copy(os.path.join(train_folder, audio_file), split_folder)

def main(opt):
    audio_files, transcript_files = load_audio_and_transcripts(opt.root_dir)
    assert len(audio_files) == len(transcript_files), f"Number of audio files ({len(audio_files)}) and transcript files ({len(transcript_files)}) do not match."

    create_dataset(opt.root_dir, audio_files, transcript_files, opt.output_dir, opt.splits)



def get_args():
    parser = argparse.ArgumentParser(description="Create a dataset from audio files and their corresponding transcriptions.")
    parser.add_argument("--root_dir", "-r", type=str, required=True, help="Path to the root folder containing the audio files and their transcriptions.")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Path to the output folder.")
    parser.add_argument("--splits", "-s", type=int, nargs='+', default=[0.8, 0.1, 0.1], help="Percentage of data to be used for training, validation and testing.")
    args = parser.parse_args()
    return utils.DotDict(args)

if __name__ == "__main__":
    """
    This script is used to create a HuggingFace dataset from audio files and their corresponding transcriptions.

    audio_paths: list of paths to audio files
    transcript_txts: list of transcription files corresponding to the audio files
    
    """
    opt = get_args()
    main(opt)