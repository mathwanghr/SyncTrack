from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
import numpy as np
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def extract_beats_from_wav(audio_file):
    downbeat_proc = RNNDownBeatProcessor()
    downbeat_activation = downbeat_proc(audio_file)
    dbn_downbeat = DBNDownBeatTrackingProcessor(
        beats_per_bar=[3, 4], min_bpm=30, max_bpm=300,
        fps=150, transition_lambda=100)
    downbeats = dbn_downbeat(downbeat_activation)
    beats = downbeats[:, 0]
    return np.array(beats)

def estimate_beat_intervals(audio_file):
    beat_times = extract_beats_from_wav(audio_file)
    if len(beat_times) < 2:
        return [], beat_times.tolist()
    beat_intervals = np.diff(beat_times)
    return beat_intervals.tolist(), beat_times.tolist()

def process_stem_folder(folder_path, stems=('stem_0', 'stem_1', 'stem_2', 'stem_3')):
    """
    
    """
    stem_results = {}
    stem_avg_beat_counts = []
    stem_avg_interval_means = []
    stem_avg_interval_stds = []

    for stem in tqdm(stems):
        stem_path = os.path.join(folder_path, stem)
        if not os.path.exists(stem_path) or not os.path.isdir(stem_path):
            print(f"{stem_path} doesn't exist or is not a directory, skipping.")
            continue
        interval_means = []
        interval_stds = []
        file_count = 0

        for fname in tqdm(os.listdir(stem_path)):
            if fname.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                file_path = os.path.join(stem_path, fname)
                intervals, beats = estimate_beat_intervals(file_path)
                if intervals is not None and len(intervals) > 0:
                    interval_means.append(np.mean(intervals))
                    interval_stds.append(np.std(intervals))
                file_count += 1

        avg_interval_mean = np.mean(interval_means) if interval_means else None
        avg_interval_std = np.mean(interval_stds) if interval_stds else None

        print(f"[{stem}] Processed {file_count} files")
        print(f"[{stem}] Avg interval mean: {avg_interval_mean}")
        print(f"[{stem}] Avg interval std: {avg_interval_std}")

        stem_results[stem] = {
            'file_count': file_count,
            'avg_interval_mean': avg_interval_mean,
            'avg_interval_std': avg_interval_std
        }

        if avg_interval_mean is not None:
            stem_avg_interval_means.append(avg_interval_mean)
        if avg_interval_std is not None:
            stem_avg_interval_stds.append(avg_interval_std)

    overall_avg_interval_mean = np.mean(stem_avg_interval_means) if stem_avg_interval_means else None
    overall_avg_interval_std = np.mean(stem_avg_interval_stds) if stem_avg_interval_stds else None

    print(f"\nIntra-Rhythmic Stability (IRS) evaluation:")
    print(f"Summary:")
    print(f"avg_interval_std: {overall_avg_interval_std}" if overall_avg_interval_std is not None else "avg_interval_std: N/A")
    print(f"avg_interval_mean: {overall_avg_interval_mean}" if overall_avg_interval_mean is not None else "avg_interval_mean: N/A")

    return {
        'stem_results': stem_results,
        'overall_avg_interval_mean': overall_avg_interval_mean,
        'overall_avg_interval_std': overall_avg_interval_std
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help='Path to folder containing stem_x subfolders')
    args = parser.parse_args()

    if os.path.exists(args.folder):
        process_stem_folder(args.folder)
    else:
        print("Please specify a valid --folder")