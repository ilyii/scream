# YouTube Audio Segment Synthesis

This environment constitues a data synthesis tool that extracts segments of audio from YouTube videos. It uses YouTube transcripts to divide the audio into meaningful segments, processes the audio, and saves it to disk. The tool is configurable and can handle multiple videos concurrently using multithreading.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Error Handling & Logging](#error-handling--logging)
- [Dependencies](#dependencies)
- [License](#license)

## Features
- Download audio from YouTube videos.
- Fetch and process transcripts (auto-generated or human-created).
- Segment audio based on transcript timings.
- Concurrent processing with configurable number of workers.
- Flexible configuration with options for downloading, segmenting, and transcript handling.
- Automatic deletion of temporary files after processing.

## Installation

### 1. Clone the repository:
```bash
git clone https://github.com/your-repo/youtube-audio-synthesis.git
cd youtube-audio-synthesis
```

### 2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # For Linux/MacOS
# or
.\venv\Scripts\activate  # For Windows
```

### 3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The tool can be run from the command line by executing `main.py`. The configuration for the processing is provided via a YAML or JSON file.

### Basic Command:
```bash
python main.py --config path/to/config.yaml
```

### Command-Line Arguments:

- `--config` (optional): Path to the configuration file (default: `datasynth/synthesize_configs.yaml`).


## Configuration

The tool's behavior is controlled through a configuration file. The config file should be in either **YAML** or **JSON** format and contain the following fields:

```yaml
num_workers: 4                          # Number of threads for concurrent processing
source: 'video_ids.txt'                  # Path to a file or a list of video IDs to process
destination: 'output_folder'             # Output folder for processed files
seg_length: 10                           # Target length of audio segments (in seconds)
num_segments_per_video: null             # Limit on the number of segments per video (optional)
do_shuffle_segments: false               # Whether to randomly shuffle segments
allow_autotranscript: false              # Allow auto-generated transcripts
download: true                           # Whether to download audio
delete_full_audio_after_segmentation: false # Whether to delete full audio after segmentation
skip_existing: true                      # Skip processing of videos that have already been processed
language: 'en'                           # Preferred transcript language (if available)
```

### Input File (source)
The `source` can be:
- A text file with one YouTube video ID per line.
- A list of video IDs within the configuration file.

Example of a valid `video_ids.txt`:
```
dQw4w9WgXcQ
9bZkp7q19f0
M7lc1UVf-VE
```

## Error Handling & Logging

- **Logging**: The tool logs all events and errors to the console. You can adjust the log level by modifying the `logging.basicConfig()` call in `main.py`.
- **Error Handling**: If a video or transcript can't be processed, the error is logged, and the tool moves on to the next video. Any invalid video ID or transcript-related issue is handled gracefully.
