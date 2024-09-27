import logging

from youtube_transcript_api import NoTranscriptFound, YouTubeTranscriptApi


def get_transcript(video_id: str, allow_auto_transcript: bool = False) -> list:
    """Fetches the transcript for the given YouTube video."""
    transcripts = []
    try:
        tx = YouTubeTranscriptApi.list_transcripts(video_id)
        for transcript in tx:
            if not allow_auto_transcript and transcript.is_generated:
                continue
            transcripts.append({
                "text": transcript.fetch(),
                "is_generated": transcript.is_generated,
                "language": transcript.language_code
            })
        if not transcripts:
            logging.info(f"No suitable transcript found for video ID: {video_id}")
    except NoTranscriptFound:
        logging.warning(f"No transcript found for video ID: {video_id}")
        return None
    except Exception as e:
        logging.error(f"Error fetching transcript for {video_id}: {str(e)}")
        return None

    return transcripts


def recalculate_durations(transcript: dict) -> dict:
    """Recalculates the durations for transcript segments."""
    try:
        for i, segment in enumerate(transcript["text"]):
            if i == len(transcript["text"]) - 1:
                segment["recalc_duration"] = transcript["duration"] - segment["start"]
            else:
                segment["recalc_duration"] = transcript["text"][i + 1]["start"] - segment["start"]
        return transcript
    except Exception as e:
        logging.error(f"Error recalculating durations: {str(e)}")
        raise e


def group_transcript(transcript: dict, target_seg_length: int = 10) -> list:
    """Groups transcript text into segments of a specified target length."""
    texts, starts, durations = zip(*[(t["text"], t["start"], t["recalc_duration"]) for t in transcript["text"]])
    segments = []
    segment_length = 0
    start_idx = 0

    for i, start in enumerate(starts):
        cur_duration = durations[i]
        segment_length += cur_duration

        if segment_length >= target_seg_length or abs(segment_length - target_seg_length) <= 1:
            segment = {
                "start": starts[start_idx],
                "duration": segment_length,
                "text": " ".join(texts[j].replace("\n", " ") for j in range(start_idx, i + 1)),
            }
            segments.append(segment)
            segment_length = 0
            start_idx = i + 1

    return segments
