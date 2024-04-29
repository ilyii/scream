import librosa
import os
import numpy as np
import yt_dlp
import soundfile as sf
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
import re

YTLINK = r'^(https?://)?(www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)$'
VID = r'^([a-zA-Z0-9_-]+)$'


def get_transcript(vid:str, allow_auto_transcript:bool=False):
    transcripts = list()
    try:
        tx = YouTubeTranscriptApi.list_transcripts(vid)
        languages = [t.language_code for t in tx]
        is_generated = [t.is_generated for t in tx]
        scripts = [t.fetch() for t in tx]

        for l, g, s in zip(languages, is_generated, scripts):        
            transcripts.append({"text": s, "is_generated": g, "language": l})

        
        if not allow_auto_transcript:
            for i,t in enumerate(transcripts):
                if t["is_generated"]:
                    del transcripts[i]
                    

        if len(transcripts) == 0:
            print("No transcript available.")

        return transcripts    
    

    except NoTranscriptFound:
        print(f"{vid}: No transcript found or transcript disabled.")
        return None
    
    


def trim_audio(input_file, output_dir, segment_duration=10):
    # Load the audio file
    audio, sr = librosa.load(input_file, sr=None)
    
    # Calculate the number of segments
    num_segments = len(audio) // (sr * segment_duration)
    
    # Trim the audio into segments
    for i in range(num_segments):
        start_sample = i * sr * segment_duration
        end_sample = (i + 1) * sr * segment_duration
        segment = audio[start_sample:end_sample]
        
        # Save each segment as a separate file
        output_file = os.path.join(output_dir, f"segment_{i}.wav")
        sf.write(output_file, segment, sr)


def extract_segment(input_file, output_file, start_time, end_time):
    audio, sr = librosa.load(input_file, sr=None)
    
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    segment = audio[start_sample:end_sample]
    sf.write(output_file, segment, sr)


def download_audio(url, output_template):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',  
            'preferredquality': '192',
        }],
        'outtmpl': output_template.split(".")[0],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def resample_audio(input_file, output_file, sr=16000):
    audio, sr = sf.read(input_file)
    audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    sf.write(output_file, audio_resampled, sr)


def binary_search(arr:list, x:float):
    left = 0
    right = len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            left = mid + 1
        else:
            right = mid - 1

    if right < 0:
        return left
    if left >= len(arr):
        return right
    
    if x - arr[right] <= arr[left] - x:
        return right
    else:
        return left



def synthesize(source:str, 
               destination:str, 
               seg_length:int=10,          
               num_segments:int=None,     
               allow_autotranscript:bool=False, 
               samplerate:int=None, 
               download:bool=True,
               skip_existing:bool=True
               ):
    
    if isinstance(source, str) and os.path.exists(source):
        with open(source, "r") as f:
            urls = f.read().splitlines()
    elif isinstance(source, list):
        urls = source
    else:
        raise ValueError(f"Invalid source: {source}")
    
    os.makedirs(destination, exist_ok=True)

    for url in urls:
        segment_length = seg_length
        try:
            url, lang = url.split(",")
        except ValueError:
            lang = "en"
        if re.match(YTLINK, url):
            # url = url
            vid = url.split("?v=")[1].split("&")[0]
        elif re.match(VID, url):
            vid = url
            url = f"https://www.youtube.com/watch?v={vid}"
        else:
            raise ValueError(f"Invalid URL: {url}")

        output_folder = os.path.join(destination, vid)
        audio_folder = os.path.join(output_folder, "audios")
        segment_folder = os.path.join(output_folder, "segments")
        transcript_folder = os.path.join(output_folder, "transcripts")

        os.makedirs(audio_folder, exist_ok=True)
        os.makedirs(segment_folder, exist_ok=True)
        os.makedirs(transcript_folder, exist_ok=True)

        audio_outfile = os.path.join(audio_folder, f"{vid}.mp3")


        if os.path.exists(audio_outfile) and skip_existing:
            continue

        if download:
            download_audio(url, audio_outfile)

        # Transcript
        transcripts = get_transcript(vid, allow_autotranscript)

        if transcripts is None:
            continue

        # Find the best transcript
        priority_map = {
            (True, False): 1,  # Specified language, not generated
            (True, True): 2,   # Specified language, generated
            (False, False): 3, # Any language, not generated
            (False, True): 4   # Any language, generated
        }

        transcripts.sort(key=lambda t: priority_map.get((t["language"] == lang, t["is_generated"]), float("inf")))
        transcript = transcripts[0]
        
        # Segmenting
        texts, starts, durations = zip(*[(t['text'], t['start'], t['duration']) for t in transcript["text"]])
        # print(vid)
        # print(starts)
        done = False
        segment_idx, start_idx = 0, 0


        # Without looping
        def group(transcript):
            texts, starts, durations = zip(*[(t['text'], t['start'], t['duration']) for t in transcript["text"]])

            segments = list()
            segment_length = 0
            start_idx = 0                        
            for i, start in enumerate(starts):                
                segment_length += durations[i]
                if segment_length >= seg_length:
                    segment_indices = list(range(start_idx, i+1))
                    new_segment = {
                        "start": starts[start_idx],
                        "duration": sum(durations[i] for i in segment_indices),
                        "text": " ".join(texts[i].replace("\n", " ") for i in segment_indices)
                    }
                    segments.append(new_segment)

                    segment_length = 0
                    start_idx = i+1

            return segments


        segments = group(transcript)
        if num_segments and num_segments < len(segments):
            random_indices = np.random.choice(len(segments), num_segments, replace=False)
            segments = [segments[i] for i in random_indices]
        
        for i, segment in enumerate(segments):
            segment_outfile = os.path.join(segment_folder, f"segment_{i:02}.wav")
            extract_segment(audio_outfile, segment_outfile, segment["start"], segment["start"] + segment["duration"])

            with open(os.path.join(transcript_folder, f"segment_{i:02}.txt"), "w", encoding="utf-8") as f:
                f.write(segment["text"])

if __name__ == "__main__":
    synthesize( source="urls.txt", 
                destination=r"D:\Database\yt-s2t", 
                seg_length=10,
                num_segments=10,
                allow_autotranscript=True, 
                # samplerate=16000,                
                download=True, 
                skip_existing=True)

                # source:str, 
                # destination:str, 
                # seg_length:int=10,          
                # num_segments:int=None,     
                # allow_autotranscript:bool=False, 
                # samplerate:int=None, 
                # download:bool=True,
                # skip_existing:bool=True


