import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from madmom.evaluation.beats import (
    find_closest_matches,
    calc_absolute_errors
)
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
from tqdm import tqdm

def extract_beats_from_wavs(wav_files):
    """
    Extract beat timestamps from all wav files using madmom.
    """
    beats_list = []
    for wav in wav_files:
        downbeat_proc = RNNDownBeatProcessor()
        downbeat_activation = downbeat_proc(wav)
        dbn_downbeat = DBNDownBeatTrackingProcessor(
            beats_per_bar=[3, 4], min_bpm=30, max_bpm=300,
            fps=150, transition_lambda=100)
        downbeats = dbn_downbeat(downbeat_activation)
        beats = downbeats[:, 0]
        beats_list.append(np.array(beats))
    return beats_list

def compute_relative_beat_error(ref_beats, tgt_beats):
    """
    Compute normalized beat error between reference and target track.
    """
    N = len(ref_beats)
    ref_beats = np.asarray(ref_beats)
    tgt_beats = np.asarray(tgt_beats)
    if N < 2 or len(tgt_beats) == 0:
        return []
    matches = find_closest_matches(ref_beats, tgt_beats)
    matched_beats = tgt_beats[matches]
    errors = calc_absolute_errors(ref_beats, tgt_beats, matches)
    intervals = np.diff(ref_beats)
    intervals_prev = np.concatenate([[intervals[0]], intervals])   # I_{n-1}
    intervals_next = np.concatenate([intervals, [intervals[-1]]]) # I_n

    left = ref_beats - intervals_prev * 0.5
    right = ref_beats + intervals_next * 0.5
    in_window = (matched_beats >= left) & (matched_beats < right)
    denominator = np.where(matched_beats >= ref_beats,
                           intervals_next * 0.5,
                           intervals_prev * 0.5)
    norm_errors = np.full(N, np.nan)
    norm_errors[in_window] = errors[in_window] / denominator[in_window]
    results = []
    for idx in range(N):
        results.append({
            'idx': idx,
            'ref': ref_beats[idx],
            'paired': bool(in_window[idx]),
            'error': errors[idx] if in_window[idx] else np.nan,
            'norm_error': norm_errors[idx]
        })
    return results

def summarize_error(results):
    """
    Summarize normalized error metrics.
    """
    paired_errors = [r['norm_error'] for r in results if r['paired']]
    avg_error = np.nanmean(paired_errors) if paired_errors else np.nan
    avg_std_error = np.nanstd(paired_errors) if paired_errors else np.nan
    avg_median_error = np.nanmedian(paired_errors) if paired_errors else np.nan
    return {
        'avg_error': avg_error,
        'avg_std_error': avg_std_error,
        'avg_median_error': avg_median_error
    }

def multi_track_consistency(beats_list):
    """
    Multi-track beat consistency metric.
    beats_list: list, beat timestamps for each track.
    Returns statistics for each reference track.
    """
    N = len(beats_list)
    ref_indices = range(N)
    stats = []
    for i in ref_indices:
        ref_beats = beats_list[i]
        all_results = []
        for j in range(N):
            if i == j: continue
            tgt_beats = beats_list[j]
            results = compute_relative_beat_error(ref_beats, tgt_beats)
            all_results.extend(results)
        stat = summarize_error(all_results)
        stat['reference_track'] = i
        stats.append(stat)
    avg_std_error = np.nanmean([s['avg_std_error'] for s in stats])
    avg_error = np.nanmean([s['avg_error'] for s in stats])
    avg_median_error = np.nanmean([s['avg_median_error'] for s in stats])
    return stats, {
        'avg_std_error': avg_std_error,
        'avg_error': avg_error,
        'avg_median_error': avg_median_error
    }

def collect_wav_groups(folder, stems=('stem_0', 'stem_1', 'stem_2', 'stem_3')):
    """
    Traverse all stems in folder, return paired wav file lists.
    Returns wav_groups and track_names.
    """
    stem_files = []
    for stem in stems:
        stem_path = os.path.join(folder, stem)
        wavs = sorted([f for f in os.listdir(stem_path) if f.endswith('.wav')])
        stem_files.append(wavs)
    common_names = set(stem_files[0])
    for files in stem_files:
        common_names &= set(files)
    common_names = sorted(list(common_names))
    wav_groups = []
    for name in common_names:
        wav_groups.append([os.path.join(folder, stem, name) for stem in stems])
    return wav_groups, [os.path.splitext(n)[0] for n in common_names]

def batch_consistency_eval(beats_groups):
    """
    Batch evaluation: input multiple beats_list groups, output statistics and overall averages.
    """
    all_stats = []
    for beats_list in tqdm(beats_groups):
        stats, overall = multi_track_consistency(beats_list)
        all_stats.append({'stats': stats, 'overall': overall})
    all_avg_std_error = [x['overall']['avg_std_error'] for x in all_stats]
    all_avg_error = [x['overall']['avg_error'] for x in all_stats]
    all_avg_median_error = [x['overall']['avg_median_error'] for x in all_stats]
    summary = {
        'avg_std_error': np.nanmean(all_avg_std_error),
        'avg_error': np.nanmean(all_avg_error),
        'avg_median_error': np.nanmean(all_avg_median_error)
    }
    return all_stats, summary

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help='Path to wav folder')
    args = parser.parse_args()

    stems = ('stem_0', 'stem_1', 'stem_2', 'stem_3')

    if os.path.exists(args.folder):
        wav_groups, track_names = collect_wav_groups(args.folder, stems)
        beats_groups = []
        print("Extracting beats for batch...")
        for wav_files in tqdm(wav_groups):
            print(wav_files)
            beats_list = extract_beats_from_wavs(wav_files)
            beats_groups.append(beats_list)
        all_stats, summary = batch_consistency_eval(beats_groups)
        print("Cross-track Beat Dispersion (CBD) evaluation:")
        print("Summary:")
        for k, v in summary.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    else:
        print("Please specify a valid --folder")


