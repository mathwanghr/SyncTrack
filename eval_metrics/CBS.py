import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import random
import csv
import json
import librosa
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
import warnings
import scipy

def load_beats_from_json(json_path):
    beats_list = []
    names = []
    with open(json_path, "r") as f:
        data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        for item in data:
            beats_list.append(np.array(item["beats"]))
            names.append(item["name"])
    return beats_list, names

def plot_multi_track_beats(track_beats, track_names=None, save_path="multi_track_beats_binary.png", resolution=0.01, window_edges=None, max_length=10.24):
    stem_names = ["bass", "drum", "guitar", "piano"]
    time_points = np.arange(0, max_length, resolution)
    plt.figure(figsize=(16, 6))
    for idx, beats in enumerate(track_beats):
        binary_seq = np.zeros_like(time_points)
        beat_indices = np.searchsorted(time_points, beats)
        beat_indices = beat_indices[beat_indices < len(binary_seq)]
        binary_seq[beat_indices] = 1
        label = stem_names[idx] if idx < len(stem_names) else f"stem_{idx}"
        plt.plot(time_points, binary_seq + idx, label=label)
    plt.yticks(range(len(track_beats)), [stem_names[i] if i < len(stem_names) else f"stem_{i}" for i in range(len(track_beats))])
    plt.xlabel("Time (s)")
    plt.title("Multi-track Beat Binary Time Series")
    plt.legend(loc='upper right')
    if window_edges is not None:
        for edge in window_edges:
            plt.axvline(x=edge, color='grey', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Multi-track beat binary time series plot saved to: {save_path}")

def estimate_beats(audio_file, sr=22050, beat_type="madmom", madmom_fps=150, madmom_transition_lambda=100):
    if beat_type == "madmom":
        downbeat_proc = RNNDownBeatProcessor()
        downbeat_activation = downbeat_proc(audio_file)
        dbn_downbeat = DBNDownBeatTrackingProcessor(
            beats_per_bar=[3, 4], min_bpm=30, max_bpm=300,
            fps=madmom_fps, transition_lambda=madmom_transition_lambda)
        downbeats = dbn_downbeat(downbeat_activation)
        beat_times = downbeats[:, 0]
    else:
        y, sr_ = librosa.load(audio_file, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr_)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr_)
    return beat_times



def compute_alignment_metric(
    track_files=None,
    track_beats=None,
    window_size=0.07,
    overlap=0.0,
    sr=22050,
    plot=True,
    fig_path="multi_track_beats.png",
    track_names=None,
    max_length=None,
    beat_type="madmom",
    madmom_fps=150,
    madmom_transition_lambda=100
):
    if track_beats is None:
        assert track_files is not None
        track_beats = [estimate_beats(f, sr, beat_type, madmom_fps, madmom_transition_lambda) for f in track_files]
        track_lengths = []
        for f in track_files:
            y, _ = librosa.load(f, sr=sr)
            track_lengths.append(len(y) / sr)
        max_length_local = max(track_lengths)
        track_names = track_files
    else:
        if max_length is None:
            raise ValueError("max_length must be specified!")
        max_length_local = max_length
        if track_names is None:
            track_names = [f"track_{i}" for i in range(len(track_beats))]

    N = len(track_beats)
    valid_track_count = sum(len(beats) > 0 for beats in track_beats)

    if valid_track_count <= 1:
        return {
            'window_count': np.nan,
            'valid_window_count': np.nan,
            'mean_beat_ratio': np.nan,
            'window_size(ms)': np.nan,
            'overlap(ms)': np.nan,
            'p_per_window': np.nan,
            'valid_track_count': valid_track_count,
        }

    step = window_size - overlap
    T = int(np.ceil((max_length_local - window_size) / step)) + 1
    windows = [(i * step, i * step + window_size) for i in range(T)]
    window_edges = [w[0] for w in windows]

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

    if plot:
        plot_multi_track_beats(track_beats, track_names, save_path=fig_path, window_edges=window_edges, max_length=max_length_local)

    return {
        'window_count': T,
        'valid_window_count': int(total_valid_windows),
        'mean_beat_ratio': mean_beat_ratio,
        'window_size(ms)': window_size * 1000,
        'overlap(ms)': overlap * 1000,
        'p_per_window': p.tolist(),
        'valid_track_count': valid_track_count,
    }

def print_valid_track_count_distribution(results):
    from collections import Counter
    valid_track_counts = [r['valid_track_count'] for r in results if r['valid_track_count'] is not None]
    dist = Counter(valid_track_counts)
    for k in sorted(dist.keys()):
        print(f"  valid_track_count = {k}: {dist[k]} samples")
    return dist


def collect_tracks_single(folder, filename, stems_list):
    return [os.path.join(folder, stem, filename) for stem in stems_list]

def collect_tracks_folder(folder, stems_list):
    stem_files = {}
    for stem in stems_list:
        stem_path = os.path.join(folder, stem)
        if not os.path.exists(stem_path):
            print(f"stem not found: {stem_path}")
            continue
        files = set(f for f in os.listdir(stem_path) if f.endswith('.wav'))
        stem_files[stem] = files
        print(stem, len(files))
    common_files = set.intersection(*(stem_files[stem] for stem in stems_list if stem in stem_files))
    print(f"found{len(common_files)} tracks")
    all_tracks = []
    for filename in sorted(common_files):
        track_files = [os.path.join(folder, stem, filename) for stem in stems_list]
        all_tracks.append(track_files)
    print(f"found {len(all_tracks)} groups of tracks")
    return all_tracks

def collect_tracks_beats(beats_json_dir, stems_list):
    all_stem_beats = []
    track_names = None
    for stem in stems_list:
        json_path = os.path.join(beats_json_dir, f"{stem}.json")
        if not os.path.exists(json_path):
            print(f"lack json: {json_path}")
            return [], []
        beats_list, names = load_beats_from_json(json_path)
        all_stem_beats.append(beats_list)
        track_names = names 
    n_tracks = len(all_stem_beats[0])
    for stem_beats in all_stem_beats:
        assert len(stem_beats) == n_tracks, "stem not match in number of tracks"
    track_beats_groups = []
    for idx in range(n_tracks):
        track_beats = [all_stem_beats[s][idx] for s in range(len(stems_list))]
        track_beats_groups.append(track_beats)
    return track_beats_groups, track_names

def analyze_folder_by_json(beats_json_dir, stems_list, window_size=0.07, overlap=0.0, sr=22050, plot_folder=None, example_count=10, max_length=None, beat_type="madmom", madmom_fps=150, madmom_transition_lambda=100):
    track_beats_groups, track_names = collect_tracks_beats(beats_json_dir, stems_list)
    results = []
    example_results = []
    example_indices = set(random.sample(range(len(track_beats_groups)), min(example_count, len(track_beats_groups))))
    if plot_folder and not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    for idx, (track_beats, track_name) in enumerate(zip(track_beats_groups, track_names)):
        plot = idx in example_indices
        fig_path = os.path.join(plot_folder, f"{track_name}.png") if (plot and plot_folder) else None
        res = compute_alignment_metric(
            track_files=None,
            track_beats=track_beats,
            window_size=window_size,
            overlap=overlap,
            sr=sr,
            plot=plot,
            fig_path=fig_path,
            track_names=stems_list,
            max_length=max_length,
            beat_type=beat_type,
            madmom_fps=madmom_fps,
            madmom_transition_lambda=madmom_transition_lambda
        )
        results.append(res)
        if plot:
            example_results.append({
                'track_name': track_name,
                'window_count': res['window_count'],
                'valid_window_count': res['valid_window_count'],
                'mean_beat_ratio': res['mean_beat_ratio'],
                'valid_track_count': res['valid_track_count'],
            })
    if plot_folder is not None and len(example_results) > 0:
        example_csv_path = os.path.join(plot_folder, "examples.csv")
        with open(example_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['track_name', 'window_count', 'valid_window_count', 'mean_beat_ratio', 'valid_track_count'])
            writer.writeheader()
            for row in example_results:
                writer.writerow(row)
    if plot_folder is not None and len(results) > 0:
        all_csv_path = os.path.join(plot_folder, "all_results.csv")
        with open(all_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['track_name', 'window_count', 'valid_window_count', 'mean_beat_ratio', 'valid_track_count'])
            writer.writeheader()
            for idx, res in enumerate(results):
                writer.writerow({
                    'track_name': track_names[idx],
                    'window_count': res['window_count'],
                    'valid_window_count': res['valid_window_count'],
                    'mean_beat_ratio': res['mean_beat_ratio'],
                    'valid_track_count': res['valid_track_count'],
                })
    avg_mean_beat_ratio = np.nanmean([r['mean_beat_ratio'] for r in results])
    avg_valid_window_count = np.nanmean([r['valid_window_count'] for r in results])
    avg_valid_track_count = np.nanmean([r['valid_track_count'] for r in results])
    # print_valid_track_count_distribution(results)
    return results

def analyze_folder_by_audio(
    folder, stems_list, window_size=0.07, overlap=0.0, sr=22050, num_workers=4,
    plot_folder=None, example_count=10, beat_type="madmom", madmom_fps=150, madmom_transition_lambda=100
):
    all_tracks = collect_tracks_folder(folder, stems_list)
    results = []
    example_results = []
    example_indices = set(random.sample(range(len(all_tracks)), min(example_count, len(all_tracks))))
    if plot_folder and not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    future_to_info = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for idx, track_files in enumerate(all_tracks):
            plot = idx in example_indices
            fig_path = os.path.join(plot_folder, f"{os.path.basename(track_files[0])}.png") if (plot and plot_folder) else None
            future = executor.submit(
                compute_alignment_metric,
                track_files,
                None,
                window_size,
                overlap,
                sr,
                plot,
                fig_path,
                stems_list,
                None,  
                beat_type,
                madmom_fps,
                madmom_transition_lambda
            )
            future_to_info[future] = (idx, track_files)
        for f in tqdm(as_completed(list(future_to_info.keys())), total=len(future_to_info), desc="Processing"):
            res = f.result()
            results.append(res)
            idx, track_files = future_to_info[f]
            if idx in example_indices and plot_folder is not None:
                wav_name = os.path.basename(track_files[0])
                example_results.append({
                    'track_files': '|'.join(track_files),
                    'window_count': res['window_count'],
                    'valid_window_count': res['valid_window_count'],
                    'mean_beat_ratio': res['mean_beat_ratio'],
                    'valid_track_count': res['valid_track_count'],
                })
    print(f"Analyzed {len(results)} track groups")
    # Save examples
    if plot_folder is not None and len(example_results) > 0:
        example_csv_path = os.path.join(plot_folder, "examples.csv")
        with open(example_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['track_files', 'window_count', 'valid_window_count', 'mean_beat_ratio', 'valid_track_count'])
            writer.writeheader()
            for row in example_results:
                writer.writerow(row)
        print(f"save {len(example_results)} examples metrics to {example_csv_path}")
    # Save all
    if plot_folder is not None and len(results) > 0:
        all_csv_path = os.path.join(plot_folder, "all_results.csv")
        with open(all_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['track_files', 'window_count', 'valid_window_count', 'mean_beat_ratio', 'valid_track_count'])
            writer.writeheader()
            for idx, res in enumerate(results):
                writer.writerow({
                    'track_files': '|'.join(all_tracks[idx]),
                    'window_count': res['window_count'],
                    'valid_window_count': res['valid_window_count'],
                    'mean_beat_ratio': res['mean_beat_ratio'],
                    'valid_track_count': res['valid_track_count'],
                })
        print(f"save all samples metrics to {all_csv_path}")
    avg_mean_beat_ratio = np.nanmean([r['mean_beat_ratio'] for r in results])
    avg_valid_window_count = np.nanmean([r['valid_window_count'] for r in results])
    avg_valid_track_count = np.nanmean([r['valid_track_count'] for r in results])
    print(f"\nmean_beat_ratio: {avg_mean_beat_ratio:.4f}")
    print(f"valid_window_count: {avg_valid_window_count:.2f}")
    print(f"valid_track_count: {avg_valid_track_count:.2f}")
    print_valid_track_count_distribution(results)
    return results

def analyze_single_track(
    folder, filename, stems_list, window_size=0.07, overlap=0.0, sr=22050, plot_folder=None,
    beat_type="madmom", madmom_fps=150, madmom_transition_lambda=100
):
    track_files = collect_tracks_single(folder, filename, stems_list)
    print(f"analyze tracks: {track_files}")
    if plot_folder and not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    fig_path = os.path.join(plot_folder, f"{filename}.png") if plot_folder else None
    result = compute_alignment_metric(
        track_files=track_files,
        track_beats=None,
        window_size=window_size,
        overlap=overlap,
        sr=sr,
        plot=True,
        fig_path=fig_path,
        track_names=stems_list,
        max_length=None, # If given original audio files, deduct max_length automatically
        beat_type=beat_type,
        madmom_fps=madmom_fps,
        madmom_transition_lambda=madmom_transition_lambda
    )
    print(f"window_count: {result['window_count']}")
    print(f"valid_window_count: {result['valid_window_count']}")
    print(f"mean_beat_ratio: {result['mean_beat_ratio']:.4f}")
    print(f"window_size(ms): {result['window_size(ms)']}")
    print(f"overlap(ms): {result['overlap(ms)']}")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=False, help='contain multiple stems subfolders')
    parser.add_argument('--beats_json_dir', required=False, help='contain each stem beats json directory (if available, use this preferentially)')
    parser.add_argument('--track', help='analyze a single filename (e.g. Track02098_from_40.wav), otherwise analyze all')
    parser.add_argument('--window_size', type=float, default=0.15, help='window length (seconds), e.g. 0.07') # 0.1, 0.15, 0.05
    parser.add_argument('--overlap', type=float, default=0.0, help='window overlap (seconds), e.g. 0.035')
    parser.add_argument('--sr', type=int, default=22050, help='sampling rate')
    parser.add_argument('--stems', nargs='+', default=["stem_0", "stem_1", "stem_2", "stem_3"], help='stem subfolder names')
    parser.add_argument('--num_workers', type=int, default=4, help='number of concurrent processes')
    parser.add_argument('--plot_folder', default="path/to/save/example/plots", help='folder path to save example plots')
    parser.add_argument('--example_count', type=int, default=1, help='number of examples to save plots for')
    parser.add_argument('--max_length', type=float, default=10.24, help='maximum duration (seconds) for beats json mode; if not given, deduce automatically')
    parser.add_argument('--beat_type', type=str, default='madmom', choices=['madmom', 'librosa'], help='choose beat detection method')
    parser.add_argument('--madmom_fps', type=int, default=150, help='madmom fps parameter')
    parser.add_argument('--madmom_transition_lambda', type=int, default=100, help='madmom transition_lambda parameter')

    args = parser.parse_args()


    if args.beats_json_dir and os.path.exists(args.beats_json_dir):
        print(f"Batch analyzing beats json directory: {args.beats_json_dir}")
        analyze_folder_by_json(
            beats_json_dir=args.beats_json_dir,
            stems_list=args.stems,
            window_size=args.window_size,
            overlap=args.overlap,
            sr=args.sr,
            plot_folder=args.plot_folder,
            example_count=args.example_count,
            max_length=args.max_length,
            beat_type=args.beat_type,
            madmom_fps=args.madmom_fps,
            madmom_transition_lambda=args.madmom_transition_lambda
        )
    elif args.folder and os.path.exists(args.folder):
        if args.track:
            analyze_single_track(
                folder=args.folder,
                filename=args.track,
                stems_list=args.stems,
                window_size=args.window_size,
                overlap=args.overlap,
                sr=args.sr,
                plot_folder=args.plot_folder,
                beat_type=args.beat_type,
                madmom_fps=args.madmom_fps,
                madmom_transition_lambda=args.madmom_transition_lambda
            )
        else:
            analyze_folder_by_audio(
                folder=args.folder,
                stems_list=args.stems,
                window_size=args.window_size,
                overlap=args.overlap,
                sr=args.sr,
                num_workers=args.num_workers,
                plot_folder=args.plot_folder,
                example_count=args.example_count,
                beat_type=args.beat_type,
                madmom_fps=args.madmom_fps,
                madmom_transition_lambda=args.madmom_transition_lambda
            )
    else:
        print("--beats_json_dir or --folder")