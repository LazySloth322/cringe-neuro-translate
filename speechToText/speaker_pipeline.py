import argparse
import gc
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
import whisper
from nemo.collections.asr.models import SortformerEncLabelModel


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

CHUNK_SECONDS = 5 * 60
CHUNK_OVERLAP_SECONDS = 15.0

DIARIZATION_MODEL = "nvidia/diar_sortformer_4spk-v1"
WHISPER_MODEL = "tiny"

MIN_SPEAKER_MATCH_OVERLAP = 0.5  # seconds
SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def run_cmd(cmd):
    """Run a subprocess and raise a readable error on failure."""
    try:
        return subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Required executable was not found: {cmd[0]!r}. "
            "Install FFmpeg and make sure ffmpeg/ffprobe are in PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(
            f"Command failed with exit code {exc.returncode}:\n"
            f"{' '.join(map(str, cmd))}\n\n{details}"
        ) from exc


def ensure_ffmpeg():
    """Check that ffmpeg and ffprobe are available."""
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(
            "FFmpeg is required. Missing: "
            + ", ".join(missing)
            + ". Install FFmpeg and add it to PATH."
        )


def get_audio_duration(audio_path):
    """Get audio duration in seconds using ffprobe."""
    result = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
    )
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(
            f"Could not determine audio duration for {audio_path}."
        ) from exc


def safe_filename(name):
    """Make a speaker name safe for use as a filename."""
    return re.sub(r"[^0-9A-Za-zА-Яа-я._-]+", "_", name).strip("_") or "Speaker"


def format_srt_time(seconds):
    """Convert float seconds to HH:MM:SS,mmm."""
    total_ms = int(round(max(0.0, seconds) * 1000))
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, millis = divmod(total_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def build_chunks(total_duration, chunk_seconds, overlap_seconds):
    """
    Return chunk definitions.

    Each chunk has a 5-minute "core" that is kept in the final result and a
    small amount of overlap used only as context for diarization/transcription.

    Chunk dict:
        index, file_start, file_end, core_start, core_end
    """
    if total_duration <= chunk_seconds:
        return [
            {
                "index": 0,
                "file_start": 0.0,
                "file_end": total_duration,
                "core_start": 0.0,
                "core_end": total_duration,
            }
        ]

    chunks = []
    core_start = 0.0
    index = 0

    while core_start < total_duration:
        core_end = min(core_start + chunk_seconds, total_duration)

        file_start = max(0.0, core_start - overlap_seconds)
        file_end = min(total_duration, core_end + overlap_seconds)

        chunks.append(
            {
                "index": index,
                "file_start": file_start,
                "file_end": file_end,
                "core_start": core_start,
                "core_end": core_end,
            }
        )

        core_start = core_end
        index += 1

    return chunks


def extract_chunk_to_wav(audio_path, chunk, output_path):
    """
    Extract a chunk as mono 16 kHz PCM WAV.

    Converting once to WAV makes both NeMo and Whisper use the same normalized
    chunk and avoids keeping the whole source audio in RAM.
    """
    duration = chunk["file_end"] - chunk["file_start"]

    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{chunk['file_start']:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(audio_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            "-loglevel",
            "error",
            str(output_path),
        ]
    )


# ---------------------------------------------------------------------------
# NeMo diarization
# ---------------------------------------------------------------------------

def parse_nemo_segments(predicted_segments):
    """
    Parse NeMo Sortformer output into:
        [{"start": float, "end": float, "speaker": "Speaker N"}, ...]

    The original script expects a nested list of strings in the form:
        "start end speaker"
    """
    segments = []

    if predicted_segments is None:
        return segments

    for speaker_list in predicted_segments:
        if isinstance(speaker_list, str):
            entries = [speaker_list]
        else:
            entries = speaker_list

        for entry in entries:
            if not isinstance(entry, str):
                continue

            parts = entry.split()
            if len(parts) < 3:
                continue

            try:
                start = float(parts[0])
                end = float(parts[1])
            except ValueError:
                continue

            speaker = parts[2].strip()
            if speaker.startswith("speaker_"):
                speaker = "Speaker " + speaker[len("speaker_"):]
            elif speaker.lower().startswith("speaker"):
                # Normalize odd variants while preserving the numeric/id part.
                match = re.search(r"(\d+)$", speaker)
                speaker = f"Speaker {match.group(1)}" if match else speaker

            if end > start:
                segments.append(
                    {
                        "start": start,
                        "end": end,
                        "speaker": speaker,
                    }
                )

    return sorted(segments, key=lambda item: item["start"])


def calc_overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def map_chunk_speakers(
    local_segments,
    chunk_file_start,
    core_start,
    core_end,
    previous_global_segments,
    next_global_speaker_id,
    overlap_seconds,
):
    """
    Map NeMo's local speaker IDs to stable global IDs.

    NeMo may restart speaker_0/speaker_1/... in every chunk. We use the overlap
    between adjacent chunks to keep IDs stable. Matching is based on how many
    seconds each local speaker overlaps each previously known global speaker.

    A speaker not seen in the overlap receives a new global ID because there is
    no reliable acoustic identity available from the two supplied scripts.
    """
    overlap_start = max(core_start - overlap_seconds, 0.0)
    overlap_end = min(core_end, chunk_file_start + overlap_seconds)

    local_speakers = sorted(
        {seg["speaker"] for seg in local_segments},
        key=lambda value: (not value.startswith("Speaker "), value),
    )

    if not previous_global_segments:
        mapping = {}
        for speaker in local_speakers:
            mapping[speaker] = f"Speaker {next_global_speaker_id}"
            next_global_speaker_id += 1
        return mapping, next_global_speaker_id

    # score[(local_speaker, global_speaker)] = total temporal overlap
    score = {}
    for local_seg in local_segments:
        local_start = chunk_file_start + local_seg["start"]
        local_end = chunk_file_start + local_seg["end"]

        clipped_start = max(local_start, overlap_start)
        clipped_end = min(local_end, overlap_end)
        if clipped_end <= clipped_start:
            continue

        for previous_seg in previous_global_segments:
            overlap = calc_overlap(
                clipped_start,
                clipped_end,
                previous_seg["start"],
                previous_seg["end"],
            )
            if overlap <= 0:
                continue

            key = (local_seg["speaker"], previous_seg["speaker"])
            score[key] = score.get(key, 0.0) + overlap

    # Greedy one-to-one assignment. Sortformer is limited to four speakers in
    # the selected model, so this is small and deterministic.
    mapping = {}
    used_global = set()

    candidates = [
        (overlap, local_speaker, global_speaker)
        for (local_speaker, global_speaker), overlap in score.items()
        if overlap >= MIN_SPEAKER_MATCH_OVERLAP
    ]
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    for _, local_speaker, global_speaker in candidates:
        if local_speaker in mapping:
            continue
        if global_speaker in used_global:
            continue

        mapping[local_speaker] = global_speaker
        used_global.add(global_speaker)

    for speaker in local_speakers:
        if speaker not in mapping:
            mapping[speaker] = f"Speaker {next_global_speaker_id}"
            next_global_speaker_id += 1

    return mapping, next_global_speaker_id


def convert_chunk_diarization_to_global(
    local_segments,
    chunk,
    previous_global_segments,
    next_global_speaker_id,
    overlap_seconds,
):
    mapping, next_global_speaker_id = map_chunk_speakers(
        local_segments=local_segments,
        chunk_file_start=chunk["file_start"],
        core_start=chunk["core_start"],
        core_end=chunk["core_end"],
        previous_global_segments=previous_global_segments,
        next_global_speaker_id=next_global_speaker_id,
        overlap_seconds=overlap_seconds,
    )

    global_segments = []

    for seg in local_segments:
        seg_start = chunk["file_start"] + seg["start"]
        seg_end = chunk["file_start"] + seg["end"]

        # Only the 5-minute core becomes part of the final diarization.
        clipped_start = max(seg_start, chunk["core_start"])
        clipped_end = min(seg_end, chunk["core_end"])

        if clipped_end <= clipped_start:
            continue

        global_segments.append(
            {
                "start": clipped_start,
                "end": clipped_end,
                "speaker": mapping[seg["speaker"]],
            }
        )

    return global_segments, next_global_speaker_id


# ---------------------------------------------------------------------------
# Whisper transcription
# ---------------------------------------------------------------------------

def choose_speaker_for_interval(start, end, diar_segments):
    """
    Choose the speaker with the largest temporal overlap.

    This is more robust than the original midpoint-only strategy when a Whisper
    segment crosses a speaker-change boundary.
    """
    overlap_by_speaker = {}

    for diar_seg in diar_segments:
        overlap = calc_overlap(start, end, diar_seg["start"], diar_seg["end"])
        if overlap <= 0:
            continue

        speaker = diar_seg["speaker"]
        overlap_by_speaker[speaker] = overlap_by_speaker.get(speaker, 0.0) + overlap

    if overlap_by_speaker:
        return max(
            overlap_by_speaker.items(),
            key=lambda item: (item[1], item[0]),
        )[0]

    # Fallback matching the spirit of the original script.
    midpoint = (start + end) / 2.0
    for seg in diar_segments:
        if seg["start"] <= midpoint <= seg["end"]:
            return seg["speaker"]

    return "Unknown Speaker"


# ---------------------------------------------------------------------------
# SRT output
# ---------------------------------------------------------------------------

def write_srt(transcript_segments, output_srt):
    with open(output_srt, "w", encoding="utf-8") as f:
        for index, seg in enumerate(transcript_segments, start=1):
            f.write(f"{index}\n")
            f.write(
                f"{format_srt_time(seg['start'])} --> "
                f"{format_srt_time(seg['end'])}\n"
            )
            f.write(f"[{seg['speaker']}]: {seg['text']}\n\n")


# ---------------------------------------------------------------------------
# Speaker audio extraction
# ---------------------------------------------------------------------------

def extract_speaker_audio(audio_path, speaker, segments, output_dir):
    OUTPUT_SAMPLE_RATE = 44_100
    """
    Extract and concatenate the audio covered by the speaker-labeled SRT
    segments, preserving the behavior of the original audiosplitter.py.
    """
    if not segments:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    final_output_path = output_dir / f"{safe_filename(speaker)}.wav"

    with tempfile.TemporaryDirectory(prefix="speaker_segments_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        concat_list_path = temp_dir_path / "concat_list.txt"

        valid_segment_files = []

        with concat_list_path.open("w", encoding="utf-8") as concat_file:
            for index, seg in enumerate(segments):
                start = float(seg["start"])
                duration = float(seg["end"]) - start

                if duration <= 0.01:
                    continue

                temp_segment_path = temp_dir_path / f"seg_{index:06d}.wav"

                run_cmd(
                    [
                        "ffmpeg",
                        "-y",
                        "-ss",
                        f"{start:.3f}",
                        "-t",
                        f"{duration:.3f}",
                        "-i",
                        str(audio_path),
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        str(OUTPUT_SAMPLE_RATE),  # ← 44100 вместо 16000
                        "-c:a",
                        "pcm_s16le",             # ← lossless PCM
                        "-af",
                        "afade=t=in:st=0:d=0.01,afade=t=out:st={:.3f}:d=0.01".format(
                            max(0, duration - 0.01)
                        ),  # ← микро-fade для устранения щелчков
                        "-loglevel",
                        "error",
                        str(temp_segment_path),
                    ]
                )

                safe_path = str(temp_segment_path).replace("'", "'\\''")
                concat_file.write(f"file '{safe_path}'\n")
                valid_segment_files.append(temp_segment_path)

        if not valid_segment_files:
            return None

        run_cmd(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list_path),
                "-c:a",
                "pcm_s16le",                     # ← lossless
                "-ar",
                str(OUTPUT_SAMPLE_RATE),          # ← 44100
                "-loglevel",
                "error",
                str(final_output_path),
            ]
        )

    return final_output_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_audio(
    audio_path,
    output_dir,
    whisper_model_name=WHISPER_MODEL,
    diarization_model_name=DIARIZATION_MODEL,
    chunk_seconds=CHUNK_SECONDS,
    overlap_seconds=CHUNK_OVERLAP_SECONDS,
    device=None,
    language=None,
):
    audio_path = Path(audio_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    ensure_ffmpeg()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Audio: {audio_path}")

    duration = get_audio_duration(audio_path)
    print(f"Audio duration: {duration / 60:.2f} min")

    chunks = build_chunks(
        total_duration=duration,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
    )

    if len(chunks) == 1:
        print("Audio <= 5 minutes: processing as a single chunk.")
    else:
        print(
            f"Audio > 5 minutes: {len(chunks)} chunks, "
            f"{chunk_seconds / 60:.1f} min core + {overlap_seconds:.1f}s overlap."
        )

    # Load both models once. Only one chunk is processed at a time, so the
    # audio data and inference outputs do not accumulate in RAM.
    print("\nLoading NVIDIA NeMo Sortformer model...")
    diar_model = SortformerEncLabelModel.from_pretrained(diarization_model_name)
    diar_model = diar_model.to(device)
    diar_model.eval()

    print("Loading OpenAI Whisper model...")
    whisper_model = whisper.load_model(whisper_model_name, device=device)

    all_diar_segments = []
    all_transcript_segments = []
    next_global_speaker_id = 0

    with tempfile.TemporaryDirectory(prefix="speaker_pipeline_") as temp_dir:
        temp_dir_path = Path(temp_dir)

        for chunk in chunks:
            chunk_number = chunk["index"] + 1
            chunk_wav = temp_dir_path / f"chunk_{chunk['index']:04d}.wav"

            print(
                f"\n=== Chunk {chunk_number}/{len(chunks)} ===\n"
                f"File range: {chunk['file_start']:.2f}s -> {chunk['file_end']:.2f}s\n"
                f"Core range: {chunk['core_start']:.2f}s -> {chunk['core_end']:.2f}s"
            )

            extract_chunk_to_wav(audio_path, chunk, chunk_wav)

            # STEP A: diarization
            print("Diarization...")
            with torch.inference_mode():
                raw_nemo_output = diar_model.diarize(
                    audio=str(chunk_wav),
                    batch_size=1,
                )

            local_diar_segments = parse_nemo_segments(raw_nemo_output)

            global_chunk_diar, next_global_speaker_id = (
                convert_chunk_diarization_to_global(
                    local_segments=local_diar_segments,
                    chunk=chunk,
                    previous_global_segments=all_diar_segments,
                    next_global_speaker_id=next_global_speaker_id,
                    overlap_seconds=overlap_seconds,
                )
            )

            all_diar_segments.extend(global_chunk_diar)

            # STEP B: Whisper
            print("Transcription...")
            transcribe_kwargs = {
                "task": "transcribe",
                "fp16": device.startswith("cuda"),
                "verbose": False,
                "condition_on_previous_text": True,
            }
            if language:
                transcribe_kwargs["language"] = language

            whisper_result = whisper_model.transcribe(
                str(chunk_wav),
                **transcribe_kwargs,
            )

            # STEP C: attach global speaker labels and keep only core region
            chunk_transcript_count = 0

            for whisper_seg in whisper_result.get("segments", []):
                local_start = float(whisper_seg["start"])
                local_end = float(whisper_seg["end"])

                global_start = chunk["file_start"] + local_start
                global_end = chunk["file_start"] + local_end

                if global_end <= global_start:
                    continue

                midpoint = (global_start + global_end) / 2.0

                # The midpoint decides which chunk owns a segment. This avoids
                # duplicate Whisper text caused by the overlap.
                if not (chunk["core_start"] <= midpoint < chunk["core_end"]):
                    continue

                text = whisper_seg.get("text", "").strip()
                if not text:
                    continue

                speaker = choose_speaker_for_interval(
                    start=global_start,
                    end=global_end,
                    diar_segments=global_chunk_diar,
                )

                all_transcript_segments.append(
                    {
                        "start": global_start,
                        "end": global_end,
                        "speaker": speaker,
                        "text": text,
                    }
                )
                chunk_transcript_count += 1

            print(
                f"Chunk complete: "
                f"{len(global_chunk_diar)} diarization segments, "
                f"{chunk_transcript_count} transcript segments."
            )

            # Explicit cleanup is useful on machines with limited RAM/VRAM.
            del raw_nemo_output
            del local_diar_segments
            del whisper_result

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Sort once more in case there are tiny timing edge cases at boundaries.
    all_diar_segments.sort(key=lambda item: (item["start"], item["end"]))
    all_transcript_segments.sort(key=lambda item: (item["start"], item["end"]))

    output_srt = output_dir / "output.srt"
    speakers_dir = output_dir / "speaker_outputs"

    write_srt(all_transcript_segments, output_srt)

    # Speaker tracks are extracted from the final labeled transcript segments.
    by_speaker = {}
    for seg in all_transcript_segments:
        by_speaker.setdefault(seg["speaker"], []).append(seg)

    print(f"\nSRT saved: {output_srt}")
    print("Extracting separate speaker tracks...")

    created_tracks = []
    for speaker in sorted(by_speaker):
        result = extract_speaker_audio(
            audio_path=audio_path,
            speaker=speaker,
            segments=by_speaker[speaker],
            output_dir=speakers_dir,
        )
        if result is not None:
            created_tracks.append(result)
            print(f"Saved: {result}")

    # Save a small run summary for convenience.
    summary_path = output_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Source: {audio_path}\n")
        f.write(f"Duration: {duration:.3f} seconds\n")
        f.write(f"Chunks: {len(chunks)}\n")
        f.write(f"Chunk core: {chunk_seconds:.3f} seconds\n")
        f.write(f"Overlap: {overlap_seconds:.3f} seconds\n")
        f.write(f"Device: {device}\n")
        f.write(f"Whisper model: {whisper_model_name}\n")
        f.write(f"NeMo model: {diarization_model_name}\n")
        f.write(f"Speakers found: {len(by_speaker)}\n")
        f.write(f"Transcript segments: {len(all_transcript_segments)}\n")

    print(
        f"\nDone. Found {len(by_speaker)} speakers. "
        f"Output directory: {output_dir}"
    )

    return {
        "output_dir": output_dir,
        "srt": output_srt,
        "speaker_tracks": created_tracks,
        "speaker_count": len(by_speaker),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Speaker diarization + Whisper transcription + separate speaker "
            "audio tracks. Long audio is processed in 5-minute chunks."
        )
    )

    parser.add_argument(
        "audio",
        help="Path to the input audio file (mp3/wav/m4a/etc.).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Default: <input_folder>/<input_stem>_output"
        ),
    )
    parser.add_argument(
        "--whisper-model",
        default=WHISPER_MODEL,
        help="Whisper model name (default: tiny).",
    )
    parser.add_argument(
        "--diarization-model",
        default=DIARIZATION_MODEL,
        help=f"NeMo Sortformer model (default: {DIARIZATION_MODEL}).",
    )
    parser.add_argument(
        "--chunk-minutes",
        type=float,
        default=5.0,
        help="Core chunk length in minutes (default: 5).",
    )
    parser.add_argument(
        "--overlap-seconds",
        type=float,
        default=CHUNK_OVERLAP_SECONDS,
        help=(
            "Overlap between neighboring chunks used for speaker-ID matching "
            "(default: 15 seconds)."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default=None,
        help="Inference device. Default: cuda if available, otherwise cpu.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Whisper language code, e.g. ru. Default: auto-detect.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.chunk_minutes <= 0:
        raise ValueError("--chunk-minutes must be > 0.")
    if args.overlap_seconds < 0:
        raise ValueError("--overlap-seconds must be >= 0.")
    if args.overlap_seconds >= args.chunk_minutes * 60:
        raise ValueError(
            "--overlap-seconds must be smaller than --chunk-minutes."
        )

    audio_path = Path(args.audio).expanduser().resolve()

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = (
            audio_path.parent
            / f"{audio_path.stem}_output"
        )

    process_audio(
        audio_path=audio_path,
        output_dir=output_dir,
        whisper_model_name=args.whisper_model,
        diarization_model_name=args.diarization_model,
        chunk_seconds=args.chunk_minutes * 60.0,
        overlap_seconds=args.overlap_seconds,
        device=args.device,
        language=args.language,
    )


if __name__ == "__main__":
    main()
