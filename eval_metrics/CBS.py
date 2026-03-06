import os
import numpy as np
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import librosa
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
import warnings
warnings.filterwarnings("ignore")

def estimate_beats(audio_file):
    downbeat_proc = RNNDownBeatProcessor()
    downbeat_activation = downbeat_proc(audio_file)
    dbn_downbeat = DBNDownBeatTrackingProcessor(
        beats_per_bar=[3, 4], min_bpm=30, max_bpm=300,
        fps=150, transition_lambda=100)
    downbeats = dbn_downbeat(downbeat_activation)
    beat_times = downbeats[:, 0]
    return beat_times

def compute_alignment_metric(
    track_files,
    window_size=0.07
):
    track_beats = [estimate_beats(f) for f in track_files]
    track_lengths = []
    # Using librosa just to get the audio duration accurately
    for f in track_files:
        y, sr_ = librosa.load(f, sr=None)
        track_lengths.append(len(y) / sr_)
    max_length_local = max(track_lengths)

    N = len(track_beats)
    valid_track_count = sum(len(beats) > 0 for beats in track_beats)

    if valid_track_count <= 1:
        return {
            'window_count': np.nan,
            'valid_window_count': np.nan,
            'mean_beat_ratio': np.nan,
            'window_size(ms)': np.nan,
            'valid_track_count': valid_track_count,
        }

    step = window_size
    T = int(np.ceil((max_length_local - window_size) / step)) + 1
    windows = [(i * step, i * step + window_size) for i in range(T)]

    b = np.zeros((T, N), dtype=int)
    for j, beats in enumerate(track_beats):
        if len(beats) == 0:
            continue
        for i, (start, end) in enumerate(windows):
            if np.any((beats >= start) & (beats < end)):
                b[i, j] = 1

    denominator = valid_track_count
    p = np.sum(b, axis=1) / denominator 
    valid_window_mask = np.sum(b, axis=1) >= 1
    total_valid_windows = np.sum(valid_window_mask)
    mean_beat_ratio = float(np.sum(p[valid_window_mask]) / (total_valid_windows + 1e-10)) if total_valid_windows > 0 else 0

    return {
        'mean_beat_ratio': mean_beat_ratio,
    }

def collect_tracks_folder(folder, stems_list):
    stem_files = {}
    for stem in stems_list:
        stem_path = os.path.join(folder, stem)
        if not os.path.exists(stem_path):
            print(f"stem not found: {stem_path}")
            continue
        files = set(f for f in os.listdir(stem_path) if f.endswith('.wav'))
        stem_files[stem] = files
    common_files = set.intersection(*(stem_files[stem] for stem in stems_list if stem in stem_files))
    print(f"found {len(common_files)} tracks")
    all_tracks = []
    for filename in sorted(common_files):
        track_files = [os.path.join(folder, stem, filename) for stem in stems_list]
        all_tracks.append(track_files)
    print(f"found {len(all_tracks)} groups of tracks")
    return all_tracks


def analyze_folder_by_audio(
    folder, stems_list, window_size=0.07, num_workers=4
):
    all_tracks = collect_tracks_folder(folder, stems_list)
    results = []
    future_to_info = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for idx, track_files in enumerate(all_tracks):
            future = executor.submit(
                compute_alignment_metric,
                track_files,
                window_size
            )
            future_to_info[future] = (idx, track_files)
        for f in tqdm(as_completed(list(future_to_info.keys())), total=len(future_to_info), desc="Processing"):
            res = f.result()
            results.append(res)


    avg_mean_beat_ratio = np.nanmean([r['mean_beat_ratio'] for r in results])
    print(f"\nCross-track Beat Synchronization (CBS) evaluation:")
    print(f"mean_beat_ratio: {avg_mean_beat_ratio:.4f}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True, help='Path to wav folder')
    parser.add_argument('--window_size', type=float, default=0.15, help='window length (seconds), e.g. 0.07') 
    parser.add_argument('--num_workers', type=int, default=4, help='number of concurrent processes')

    args = parser.parse_args()

    stems = ["stem_0", "stem_1", "stem_2", "stem_3"]

    if os.path.exists(args.folder):
        analyze_folder_by_audio(
            folder=args.folder,
            stems_list=stems,
            window_size=args.window_size,
            num_workers=args.num_workers
        )
    else:
        print("Please specify a valid --folder")