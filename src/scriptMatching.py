import json
import re
import os
import argparse
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from moviepy.editor import AudioFileClip
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from utils.video_cache import iter_registered_videos, video_id_for_path

try:
    from nltk.corpus import wordnet as wn
    _ = wn.synsets("example")
    WORDNET_AVAILABLE = True
except Exception:  # pragma: no cover - graceful fallback if wordnet data missing
    wn = None
    WORDNET_AVAILABLE = False

try:
    from nltk.stem import PorterStemmer

    STEMMER = PorterStemmer()
except Exception:  # pragma: no cover - nltk optional at runtime
    STEMMER = None


STOP_WORDS = set(ENGLISH_STOP_WORDS)
SYNONYM_CACHE: Dict[str, Sequence[str]] = {}


def load_selected_videos(selected_videos_path: Optional[str]) -> List[Dict[str, Any]]:
    if not selected_videos_path:
        return []

    if not os.path.exists(selected_videos_path):
        print(f"Selected videos manifest not found at {selected_videos_path}. Proceeding without it.")
        return []

    try:
        with open(selected_videos_path, "r", encoding="utf-8") as manifest_file:
            payload = json.load(manifest_file)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Unable to read selected videos manifest: {exc}")
        return []

    videos = payload.get("videos", [])
    normalized: List[Dict[str, Any]] = []

    for video in videos:
        path = video.get("path")
        if path:
            path = os.path.abspath(path)

        video_id = video.get("id") or (video_id_for_path(path) if path else None)
        if not video_id:
            continue

        normalized.append(
            {
                "id": video_id,
                "path": path,
                "filename": video.get("filename") or (os.path.basename(path) if path else video_id),
                "metadata_file": video.get("metadata_file"),
                "root": video.get("root") or (os.path.dirname(path) if path else None),
            }
        )

    return normalized


def _tokenize(text: str) -> List[str]:
    """Return alphabetic tokens from text in lowercase."""

    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


def _filter_tokens(tokens: Sequence[str]) -> List[str]:
    """Remove stopwords and optionally stem tokens."""

    if STEMMER is None:
        return [token for token in tokens if token not in STOP_WORDS]
    return [STEMMER.stem(token) for token in tokens if token not in STOP_WORDS]


def stem_tokenizer(text: str) -> List[str]:
    """Tokenizer used by the vectorizer (lowercase, stopword removal, stemming)."""

    return _filter_tokens(_tokenize(text))


def get_synonyms(token: str) -> Sequence[str]:
    """Fetch synonym candidates for a token using WordNet, if available."""

    if not WORDNET_AVAILABLE:
        return ()

    if token in SYNONYM_CACHE:
        return SYNONYM_CACHE[token]

    synonyms = set()
    try:
        for syn in wn.synsets(token, pos=getattr(wn, "NOUN", None)):
            for lemma in syn.lemmas():
                candidate = lemma.name().lower()
                if not candidate.isalpha() or candidate in STOP_WORDS or len(candidate) <= 2:
                    continue
                synonyms.add(candidate)
    except Exception:  # pragma: no cover - safety around optional data
        synonyms = set()

    SYNONYM_CACHE[token] = tuple(sorted(synonyms))
    return SYNONYM_CACHE[token]


def augment_text_with_synonyms(text: str) -> str:
    """Append synonym candidates to the text for richer matching."""

    tokens = _tokenize(text)
    base_terms = set(tokens)
    synonym_terms = set()

    for token in base_terms:
        synonym_terms.update(get_synonyms(token))

    additional_terms = synonym_terms - base_terms
    if not additional_terms:
        return text

    return f"{text} {' '.join(sorted(additional_terms))}".strip()


def build_clip_search_index(metadata: Dict[str, Sequence[dict]]) -> Tuple[
    Optional[TfidfVectorizer],
    Optional[Any],
    List[dict],
]:
    """Prepare TF-IDF matrix and clip metadata for similarity search."""

    clip_texts: List[str] = []
    clip_index: List[dict] = []

    for video_name, entries in metadata.items():
        video_base = os.path.splitext(video_name)[0]
        video_context_tokens = _tokenize(video_base)
        video_context_synonyms = set()

        for token in video_context_tokens:
            video_context_synonyms.update(get_synonyms(token))

        video_context_extra = video_context_synonyms - set(video_context_tokens)
        video_context = " ".join(
            [
                *(video_context_tokens or []),
                *(sorted(video_context_extra) if video_context_extra else []),
            ]
        ).strip()

        for entry in entries:
            objects = entry.get("objects", [])
            object_terms: List[str] = []
            for obj in objects:
                object_terms.extend(_tokenize(obj))

            base_text = " ".join(object_terms)
            enriched_text = augment_text_with_synonyms(base_text) if base_text else ""
            combined = " ".join(
                term
                for term in (enriched_text, video_context)
                if term
            ).strip()

            clip_texts.append(combined or video_context or "video")
            clip_index.append(
                {
                    "video": video_name,
                    "timestamp": entry.get("timestamp", 0),
                    "objects": objects,
                }
            )

    if not clip_texts:
        return None, None, clip_index

    vectorizer = TfidfVectorizer(
        tokenizer=stem_tokenizer,
        lowercase=False,
        ngram_range=(1, 2),
        token_pattern=None,
    )

    try:
        metadata_matrix = vectorizer.fit_transform(clip_texts)
    except ValueError:
        return None, None, clip_index

    return vectorizer, metadata_matrix, clip_index


def compute_clip_similarity(
    sentence: str,
    vectorizer: Optional[TfidfVectorizer],
    metadata_matrix,
    clip_index: Sequence[dict],
    used_videos: set,
    available_videos: Sequence[str],
) -> Tuple[Optional[Tuple[str, int]], float]:
    """Return the best matching clip and its similarity score for the sentence."""

    if not vectorizer or metadata_matrix is None or not clip_index:
        return None, 0.0

    augmented_sentence = augment_text_with_synonyms(sentence)
    sentence_vector = vectorizer.transform([augmented_sentence])

    if sentence_vector.nnz == 0:
        return None, 0.0

    similarities = metadata_matrix.dot(sentence_vector.T)
    if similarities.nnz == 0:
        return None, 0.0

    scores = np.asarray(similarities.toarray()).ravel()
    if not np.any(scores):
        return None, 0.0

    candidate_indices = np.argsort(scores)[::-1]
    best_idx: Optional[int] = None

    for idx in candidate_indices:
        score = scores[idx]
        if score <= 0:
            break

        candidate_video = clip_index[idx]["video"]
        if len(used_videos) < len(available_videos) and candidate_video in used_videos:
            continue

        best_idx = idx
        break

    if best_idx is None:
        positive_indices = [idx for idx in candidate_indices if scores[idx] > 0]
        if positive_indices:
            best_idx = positive_indices[0]

    if best_idx is None:
        return None, 0.0

    candidate = clip_index[best_idx]
    return (candidate["video"], candidate["timestamp"]), float(scores[best_idx])


def fallback_best_clip(
    sentence: str,
    metadata: Dict[str, Sequence[dict]],
    available_videos: Sequence[str],
    used_videos: set,
) -> Optional[Tuple[str, int]]:
    """Fallback overlap-based search in case vector search yields nothing."""

    sentence_tokens = set(stem_tokenizer(sentence))
    if not sentence_tokens:
        sentence_tokens = set(_tokenize(sentence))

    best_match: Optional[Tuple[str, int]] = None
    highest_score = -1

    for video_name in available_videos:
        if len(used_videos) < len(available_videos) and video_name in used_videos:
            continue

        for entry in metadata.get(video_name, []):
            entry_tokens = set(stem_tokenizer(" ".join(entry.get("objects", []))))
            if not entry_tokens:
                continue

            overlap = len(sentence_tokens & entry_tokens)
            if overlap > highest_score:
                highest_score = overlap
                best_match = (video_name, entry.get("timestamp", 0))

    if best_match:
        return best_match

    for video_name in available_videos:
        if video_name not in used_videos and metadata.get(video_name):
            return video_name, metadata[video_name][0].get("timestamp", 0)

    if available_videos:
        first_video = available_videos[0]
        if metadata.get(first_video):
            return first_video, metadata[first_video][0].get("timestamp", 0)

    return None


def calculate_sentence_durations(script_text, audio_duration):
    """Calculate how long each sentence should last based on word count."""
    sentences = re.split(r'(?<=[.!?])\s+', script_text.strip())
    word_counts = [len(re.findall(r'\b\w+\b', sentence)) for sentence in sentences]
    total_words = sum(word_counts)

    # Calculate each sentence's duration proportionally to its word count
    sentence_durations = [(words / total_words) * audio_duration for words in word_counts]
    return sentences, sentence_durations

def match_script_to_clips(
    script_path,
    metadata_folder,
    audio_path,
    output_matches_path,
    selected_videos_path: Optional[str] = None,
):
    """Match each sentence in the script to video clips with appropriate durations."""

    # Step 1: Get audio duration from the narration file
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return

    with AudioFileClip(audio_path) as narration_audio:
        audio_duration = narration_audio.duration

    print(f"Total narration audio duration: {audio_duration:.2f} seconds")

    # Step 2: Load the script and calculate sentence durations
    with open(script_path, "r") as file:
        script_text = file.read()

    sentences, sentence_durations = calculate_sentence_durations(script_text, audio_duration)
    print(f"Found {len(sentences)} sentences in the script.")

    # Step 3: Load metadata for selected or cached videos
    metadata: Dict[str, List[Dict[str, Any]]] = {}
    video_lookup: Dict[str, Dict[str, Any]] = {}

    selected_videos = load_selected_videos(selected_videos_path)

    if selected_videos:
        for video in selected_videos:
            video_id = video["id"]
            metadata_file = video.get("metadata_file") or os.path.join(metadata_folder, f"{video_id}_objects.json")

            if not os.path.exists(metadata_file):
                print(f"Metadata missing for {video['filename']} ({metadata_file}). Skipping.")
                continue

            with open(metadata_file, "r", encoding="utf-8") as file:
                metadata[video_id] = json.load(file)

            video_lookup[video_id] = {
                "id": video_id,
                "path": video.get("path"),
                "filename": video.get("filename") or video_id,
                "metadata_file": metadata_file,
            }

    if not metadata:
        for video_id, entry in iter_registered_videos():
            metadata_file = entry.get("metadata_file") or os.path.join(metadata_folder, f"{video_id}_objects.json")
            if not os.path.exists(metadata_file):
                continue

            with open(metadata_file, "r", encoding="utf-8") as file:
                metadata[video_id] = json.load(file)

            filename = entry.get("filename")
            if not filename and entry.get("path"):
                filename = os.path.basename(entry["path"])
            if not filename:
                filename = f"{video_id}.mp4"

            video_lookup[video_id] = {
                "id": video_id,
                "path": entry.get("path"),
                "filename": filename,
                "metadata_file": metadata_file,
            }

    if not metadata:
        for metadata_file in os.listdir(metadata_folder):
            if metadata_file.endswith("_objects.json"):
                video_id = metadata_file.replace("_objects.json", "")
                metadata_path = os.path.join(metadata_folder, metadata_file)
                with open(metadata_path, "r", encoding="utf-8") as file:
                    metadata[video_id] = json.load(file)

                video_lookup[video_id] = {
                    "id": video_id,
                    "path": None,
                    "filename": f"{video_id}.mp4",
                    "metadata_file": metadata_path,
                }

    if not metadata:
        print("No metadata available.")
        return

    friendly_names = [video_lookup.get(video_id, {}).get("filename", video_id) for video_id in metadata.keys()]
    print(f"Available videos: {friendly_names}")

    # Step 4: Prepare enhanced clip search index
    vectorizer, metadata_matrix, clip_index = build_clip_search_index(metadata)
    if vectorizer is None or metadata_matrix is None:
        print("Falling back to basic keyword matching; insufficient metadata for vector search.")

    matched_clips = []
    used_videos = set()
    available_videos = list(metadata.keys())

    for sentence, duration in zip(sentences, sentence_durations):
        if len(used_videos) == len(available_videos):
            print("All videos have been used. Resetting available videos.")
            used_videos.clear()

        match, score = compute_clip_similarity(
            sentence,
            vectorizer,
            metadata_matrix,
            clip_index,
            used_videos,
            available_videos,
        )

        if not match:
            match = fallback_best_clip(sentence, metadata, available_videos, used_videos)
            score = 0.0

        if match:
            video_id, timestamp = match
            video_info = video_lookup.get(video_id, {"filename": video_id, "path": None})
            matched_clips.append(
                {
                    "sentence": sentence,
                    "video": video_info.get("filename", video_id),
                    "video_id": video_id,
                    "video_path": video_info.get("path"),
                    "timestamp": timestamp,
                    "duration": duration,
                }
            )
            used_videos.add(video_id)
            score_text = f" (score: {score:.3f})" if score else ""
            print(
                f"Matched sentence: '{sentence}' to video: '{video_info.get('filename', video_id)}' "
                f"(id: {video_id}) at timestamp: {timestamp} "
                f"for {duration:.2f} seconds{score_text}"
            )
        else:
            print(f"No match found for sentence: '{sentence}'")

    # Step 6: Save the matches to a JSON file
    if matched_clips:
        with open(output_matches_path, "w") as file:
            json.dump(matched_clips, file, indent=4)
        print(f"Matched clips saved to {output_matches_path}")
    else:
        print("No clips were matched to any sentence.")

    return matched_clips

def parse_args():
    parser = argparse.ArgumentParser(description="Match narration sentences to available clips using object metadata.")
    parser.add_argument("script_path", nargs="?", default="data/script/narration_script.txt", help="Path to the narration script text file")
    parser.add_argument("metadata_folder", nargs="?", default="data/metadata", help="Folder containing *_objects.json metadata files")
    parser.add_argument("audio_path", nargs="?", default="data/audio/narration_audio.mp3", help="Narration audio file path")
    parser.add_argument("output_matches_path", nargs="?", default="data/matches/matched_clips.json", help="Where to write the matched clips JSON")
    parser.add_argument("--selected-videos", help="Path to a JSON manifest of the selected videos to use", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    match_script_to_clips(
        script_path=args.script_path,
        metadata_folder=args.metadata_folder,
        audio_path=args.audio_path,
        output_matches_path=args.output_matches_path,
        selected_videos_path=args.selected_videos,
    )
