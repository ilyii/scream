# SCREAM: SpeeCh Recognition and Enhancement for Audio Messages

## Project Overview

SCREAM (SpeeCh Recognition and Enhancement for Audio Messages) is designed to enhance and optimize digital communication by transcribing and refining spoken content. In a society that increasingly relies on fast, efficient communication through digital messenger services, SCREAM leverages advanced speech recognition and content enhancement algorithms to accelerate the exchange of information.

---

## Project Structure

- **`datasynth/`**: Custom dataset synthesis for training and evaluation purposes.
- **`evaluation/`**: Scripts for assessing transcription and speech processing performance.
- **`notebooks/`**: Misc Jupyter notebooks
- **`telegram_bot/`**: Telegram bot for transcribing and summarizing audio messages.
- **`utils/`**: General utility functions that support various modules of the project.

## Key Functionalities

SCREAM is tailored for processing spoken content, making it ideal for enhancing communication in digital formats like audio messages. Key functionalities include:

1. **Automatic Speech Recognition (ASR)**: SCREAM downloads and transcribes audio messages using state-of-the-art speech recognition models like OpenAI Whisper and its optimized variants.
  
2. **Content Enhancement**: Post-processing algorithms refine the transcription by removing filler words, pauses, and other superfluous elements, resulting in concise and coherent text.

3. **Audio Segmentation**: Long audio messages are segmented into manageable parts, allowing for better organization and easier processing of content.

4. **Evaluation**: Includes quality metrics and tools for evaluating the performance of the transcription models, ensuring high accuracy and reliability.

5. **Telegram Integration**: SCREAM includes a Telegram bot that allows users to transcribe and summarize audio messages directly from the messaging platform.

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
```

### 2. Set up the virtual environment

Create and activate a virtual environment for package management:

```bash
# macOS/Linux
python -m venv .venv
source .venv/bin/activate   
```

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate      
```

### 3. Install dependencies

Once the virtual environment is activated, install all required packages:

```bash
pip install -r requirements.txt
```

Or install the packages with uv:

```bash
pip install uv
uv sync
```

### 4. Set up the .env file

Create a `.env` file in the root directory and add the following environment variables:

```bash
TELEGRAM_BOT_API_KEY="your_telegram_bot_api_key"
GEMINI_API_KEY="your_gemini_api_key"
```

## Usage

SCREAM facilitates speech-to-text transcriptions for audio messages, processes and refines the transcriptions, and evaluates the output for accuracy. Here is how you can use the key functionalities:

1. **Download and Process YouTube Audio**: Extract and segment audio content from YouTube videos using SCREAM’s built-in downloading functions.
  
2. **Transcription and Refinement**: Use advanced speech recognition algorithms to transcribe audio content into text, followed by refinement to remove unnecessary elements.

3. **Evaluation**: Assess the quality of the transcriptions using evaluation scripts and metrics provided in the `evaluation/` directory.

4. **Telegram Bot Integration**: Deploy the Telegram bot to transcribe and summarize audio messages directly from the messaging platform.
