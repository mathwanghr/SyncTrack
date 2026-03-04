import os
import glob
import numpy as np
import argparse
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
from itertools import product

def extract_beats(audio_file, fps, transition_lambda):
    downbeat_proc = RNNDownBeatProcessor()
    downbeat_activation = downbeat_proc(audio_file)
    dbn_downbeat = DBNDownBeatTrackingProcessor(
        beats_per_bar=[3, 4],
        min_bpm=30,
        max_bpm=300,
        fps=fps,
        transition_lambda=transition_lambda,
    )
    downbeats = dbn_downbeat(downbeat_activation)
    beat_times = downbeats[:, 0]
    return beat_times

def process_single_file(audio_file, fps, transition_lambda):
    try:
        beat_times = extract_beats(audio_file, fps=fps, transition_lambda=transition_lambda)
        base = os.path.splitext(os.path.basename(audio_file))[0]
        return {"name": base, "beats": beat_times}
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None

def process_stem_folder_json_threaded(stem_folder, fps, transition_lambda, ext, out_subdir, max_workers=8):
    audio_files = sorted(glob.glob(os.path.join(stem_folder, f"*.{ext}")))
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_single_file, audio_file, fps, transition_lambda): audio_file
            for audio_file in audio_files
        }
        for future in tqdm(as_completed(future_to_file), total=len(audio_files), desc=f"Processing {os.path.basename(stem_folder)}"):
            result = future.result()
            if result is not None:
                result["beats"] = list(result["beats"])
                results.append(result)
    out_path = os.path.join(out_subdir, f"{os.path.basename(stem_folder)}.json")
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"Saved results to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="Directory containing audio files (e.g., path/to/experiment/output)")
    parser.add_argument("--ext", type=str, default="wav", help="Audio file extension")
    parser.add_argument("--fps", type=int, default=150, help="Frames per second for Madmom processing")
    parser.add_argument("--transition_lambda", type=float, default=100, help="Transition lambda parameter for Madmom's DBN (e.g., 10, 50, 100)")
    parser.add_argument("--out_dir", type=str, default="Beats/beat_results", help="Output directory for extracted beats")
    parser.add_argument("--param_tag", type=str, default="", help="Optional parameter tag for output subfolder")
    parser.add_argument("--max_workers", type=int, default=4, help="Number of threads for parallel processing")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    # Tag by parameters for easy tuning and result comparison
    param_str = f"fps_{args.fps}_tl_{args.transition_lambda}"
    if args.param_tag:
        param_str = args.param_tag + "_" + param_str
    out_subdir = os.path.join(args.out_dir, param_str)
    os.makedirs(out_subdir, exist_ok=True)

    # Automatically detect stem folders (stem_0, stem_1, ...) or process the folder itself
    stem_folders = [os.path.join(args.folder, f"stem_{i}") for i in range(4)]
    found_stems = False
    for stem_folder in stem_folders:
        if os.path.isdir(stem_folder):
            found_stems = True
            process_stem_folder_json_threaded(
                stem_folder,
                fps=args.fps,
                transition_lambda=args.transition_lambda,
                ext=args.ext,
                out_subdir=out_subdir,
                max_workers=args.max_workers
            )
    
    if not found_stems:
        # If no stem_i folders found, process the provided folder directly
        process_stem_folder_json_threaded(
            args.folder,
            fps=args.fps,
            transition_lambda=args.transition_lambda,
            ext=args.ext,
            out_subdir=out_subdir,
            max_workers=args.max_workers
        )

if __name__ == "__main__":
    main()