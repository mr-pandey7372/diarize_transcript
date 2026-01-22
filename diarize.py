from pyannote.audio import Pipeline
import whisperx
import torch
import json

# -----------------------------
# CONFIG
# -----------------------------
AUDIO_FILE = "test6.wav"
OUTPUT_JSON = "test6.json"

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
whisper_device = "cuda" if torch.cuda.is_available() else "cpu"

TIME_TOLERANCE = 0.05  # 50ms tolerance for word-speaker overlap

# -----------------------------
# 1. LOAD PYANNOTE DIARIZATION
# -----------------------------
print("Loading PyAnnote diarization model...")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
pipeline.to(torch_device)

print("Running speaker diarization...")
diarization = pipeline(AUDIO_FILE)

# -----------------------------
# 2. LOAD WHISPERX & TRANSCRIBE
# -----------------------------
print("Loading WhisperX model...")
model = whisperx.load_model(
    "base",
    device=whisper_device,
    compute_type="float32"
)

audio = whisperx.load_audio(AUDIO_FILE)

print("Transcribing audio...")
result = model.transcribe(audio, language="en")

# -----------------------------
# 3. WORD ALIGNMENT
# -----------------------------
print("Aligning words...")
align_model, align_metadata = whisperx.load_align_model(
    language_code="en",
    device=whisper_device
)

result = whisperx.align(
    result["segments"],
    align_model,
    align_metadata,
    audio,
    whisper_device
)

words = result["word_segments"]

# -----------------------------
# 4. MERGE SPEAKERS + WORDS
# -----------------------------
print("Merging speakers with transcript...")
final_output = []

for turn, _, speaker in diarization.itertracks(yield_label=True):
    seg_start = turn.start
    seg_end = turn.end

    segment_words = [
        w["word"]
        for w in words
        if w.get("start") is not None
        and w.get("end") is not None
        and (seg_start - TIME_TOLERANCE) <= w["start"] <= (seg_end + TIME_TOLERANCE)
    ]

    transcript = " ".join(segment_words).strip()

    # Skip empty segments
    if not transcript:
        continue

    final_output.append({
        "start": round(seg_start, 2),
        "end": round(seg_end, 2),
        "speaker": speaker,
        "transcript": transcript,
        "sentiment": None,
        "ser": None
    })

# -----------------------------
# 5. SAVE OUTPUT
# -----------------------------
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=4, ensure_ascii=False)

print("\n✅ Done!")
print(f"Output saved as: {OUTPUT_JSON}")
