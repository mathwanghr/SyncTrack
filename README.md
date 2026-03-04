<div align="center">

# SyncTrack: Rhythmic Stability and Synchronization in Multi-Track Music Generation

<!-- <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://lightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a> -->
[![Paper](https://img.shields.io/badge/OpenReview-Paper-b31b1b.svg)](https://openreview.net/forum?id=Jf7i0a8dr0)
[![HomePage](https://img.shields.io/badge/HomePage-SyncTrack-blue)](https://synctrack-v1.github.io/)
[![Conference](https://img.shields.io/badge/ICLR-2026-green)](https://iclr.cc//)

</div>

## 📖 Abstract

> Multi-track music generation has garnered significant research interest due to its precise mixing and remixing capabilities. However, existing models often overlook essential attributes such as **rhythmic stability and synchronization**, leading to a focus on differences between tracks rather than their inherent properties.
> 
> In this paper, we introduce **SyncTrack**, a synchronous multi-track waveform music generation model designed to capture the unique characteristics of multi-track music. SyncTrack features a novel architecture that includes a shared module to establish a common rhythm across all tracks and track-specific modules to accommodate diverse timbres and pitch ranges. The shared module employs two cross-track attention mechanisms to synchronize rhythmic information, while the track-specific modules utilize learnable instrument priors to better represent timbre and other unique features. 
>
> Additionally, we enhance the evaluation of multi-track music quality by introducing rhythmic consistency through three novel metrics: **Inner-track Rhythmic Stability (IRS), Cross-track Beat Synchronization (CBS), and Cross-track Beat Dispersion (CBD)**. Both objective metrics and subjective listening tests demonstrate that SyncTrack significantly improves multi-track music quality by enhancing rhythmic consistency.


<div align="center">
<img src="assets/architecture.png" alt="SyncTrack Architecture" width="800px">
</div>

## 🔥 News
- **[2026.03]** The official implementation and evaluation metrics for SyncTrack are released.
- **[2026.01]** Paper accepted to ICLR 2026!


## 🛠️ Prerequisites

### 1. Environment Setup
Ensure you have the required dependencies installed. The codebase assumes a CUDA-enabled environment for efficient processing.

```bash
# Clone the repository
git clone https://github.com/YangLabHKUST/SyncTrack.git
cd SyncTrack

# Install dependencies 
conda create -n synctrack python=3.9
conda activate synctrack
pip install -r requirements.txt
```
>
> **Note:** For evaluation metrics, ensure you have the necessary audio processing libraries like `madmom` installed on your system.


### 2. Model Checkpoints
To run the model, you must download the pre-trained weights for **VAE**, **HiFi-GAN**, and **MusicLDM**.

- 📥 **Download Link:** [Model Checkpoints](https://zenodo.org/records/18797721?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjlhYzY4MDRiLWE4YzktNGRkOC05MzQwLTEwYzc5ZTI2MjM4MCIsImRhdGEiOnt9LCJyYW5kb20iOiIxOGUxMTQxNTI1ZGQyZmU0NGZjYjFmZjM0OThiNzJlNiJ9.iPuxzy7aIE1HFs0q1EVsVc3r87Mq5FyizZPnDr21-Nu6hpTbfpb7omGNZyg4G3JckMBs_qOZTuLSi5IiWONtqw) 

After downloading, unzip the file and place the contents into the `ckpt/` directory. Ensure your directory structure looks like this:

```text
SyncTrack/
├── ckpt/
│   ├── vae-ckpt.ckpt        # (Example name, matches your unzipped files)
│   ├── hifigan-ckpt.ckpt
│   └── musicldm-ckpt.ckpt
├── config/
├── src/
└── ...
```


## 🚀 Training

To train the SyncTrack model, use the `train_musicldm.py` script. The configuration is managed via YAML files.

### Configuration Setup
In `config/synctrack_train.yaml`, please configure the following paths before starting:
- `data.params.path.train_data`: Path to your **training** dataset.
- `data.params.path.valid_data`: Path to your **validation** dataset.
- `model.params.ckpt_path`: (Optional) Path to a pre-trained checkpoint to resume training.

### Run Training
```bash
python train_musicldm.py --config config/synctrack_train.yaml
```

## ⚡ Inference & Evaluation

To generate samples or evaluate the model on the test set, use the `eval_musicldm.py` script.


### Configuration Setup
In `config/synctrack_eval.yaml`, please configure the following:

- `mode`: Ensure this is set to `test`.
- `data.params.path.valid_data`: Path to your **test** dataset.
- `model.params.ckpt_path`: Path to the model checkpoint.

> 🌟 **Use [Our Pre-trained Model](https://zenodo.org/records/18797721?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjlhYzY4MDRiLWE4YzktNGRkOC05MzQwLTEwYzc5ZTI2MjM4MCIsImRhdGEiOnt9LCJyYW5kb20iOiIxOGUxMTQxNTI1ZGQyZmU0NGZjYjFmZjM0OThiNzJlNiJ9.iPuxzy7aIE1HFs0q1EVsVc3r87Mq5FyizZPnDr21-Nu6hpTbfpb7omGNZyg4G3JckMBs_qOZTuLSi5IiWONtqw)**
> If you wish to evaluate using our pre-trained weights:
> 1. Download **`synctrack_slakh2100_pretrained.ckpt`** from [Insert Link Here].
> 2. Unzip and place the file into the `ckpt/` directory.
> 3. Set the config path: `model.params.ckpt_path: "ckpt/synctrack_slakh2100_pretrained.ckpt"`.

### Run Inference
```bash
python eval_musicldm.py --config config/synctrack_eval.yaml
```

## 📊 Evaluation Metrics

We provide a comprehensive suite of metrics to measure rhythmic stability and synchronization, located in the `eval_metrics` directory.

### 1. Cross-Beat Deviation (CBD)
`CBD.py` measures multi-track beat consistency using relative beat errors. It evaluates the rhythmic consistency between different tracks of a music composition.

**Key Parameters:**
- `--reference_type`: `"all"` (average error across all tracks) or specific index (e.g., `"0"`).
- `--pair_window_ratio`: Valid matching window ratio (default: `0.5`).
- `--beat_type`: `"madmom"` (default) or `"librosa"`.

```bash
python eval_metrics/CBD.py --folder /path/to/generated/stems --beat_type madmom --reference_type all
```

<details>
<summary><b>Click for detailed usage and full options</b></summary>

#### Features
- **Multiple Input Modes**: Support for single file sets, batch processing of WAV folders, and pre-computed JSON beat files.
- **Flexible Beat Detection**: Supports both `madmom` (RNN/DBN) and `librosa` for beat extraction.
- **Cross-Reference Evaluation**: Can evaluate consistency by treating every track as a reference ("all-to-all") or by aligning all tracks to a specific stem.

#### Additional Usage Examples

**1. Evaluate a Single Set of Audio Files**
```bash
python eval_metrics/CBD.py \
  --single_files "path/to/bass.wav" "path/to/drums.wav" "path/to/guitar.wav" "path/to/piano.wav" \
  --beat_type madmom
```

**2. Batch Evaluate Pre-computed JSON Beats**
```bash
python eval_metrics/CBD.py \
  --beats_json_folder "/path/to/json_folder"
```

#### Configuration Arguments
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--reference_type` | `all` | `all`: Compare all tracks against all others. <br>`0`, `1`, etc.: Use specific track index as ground truth. |
| `--beat_type` | `madmom` | Algorithm for beat tracking: `madmom`, `librosa`. |
| `--pair_window_ratio` | `0.5` | Tolerance window for matching beats (relative to beat interval). |
| `--madmom_fps` | `150` | Frames per second for Madmom processing. |
| `--madmom_transition_lambda` | `100` | Transition lambda parameter for Madmom's DBN. |

#### Output Metrics
- **avg_error**: Average normalized time difference between matched beats.
- **avg_std_error**: Standard deviation of beat errors.
- **avg_median_error**: Median error.


</details>



### 2. Cross-Beat Synchrony (CBS)
`CBS.py` evaluates the rhythmic synchronization between multiple tracks by calculating the **beat alignment ratio** within sliding time windows. It determines how frequently beats from different instruments (stems) co-occur within a specific tolerance window.

**Key Parameters:**
- `--window_size`: Length of the sliding window in seconds (default: `0.15`).
- `--overlap`: Overlap between consecutive windows in seconds (default: `0.0`).
- `--beat_type`: `"madmom"` (recommended) or `"librosa"`.

```bash
python eval_metrics/CBS.py --folder /path/to/generated/stems --window_size 0.07 --overlap 0.0
```

<details>
<summary><strong>Click for detailed usage and full options</strong></summary>

#### Features
- **Flexible Input Sources**: Supports raw audio folders, specific single files, or pre-computed Beat JSON files (fastest).
- **Visualization**: Automatically generates binary beat plots (`.png`) showing where beats align across tracks.
- **Parallel Processing**: Uses multi-processing to speed up audio analysis.

#### Additional Usage Examples

**1. Batch Evaluate Pre-computed JSON Beats (Recommended)**
If you have already extracted beats using `batch_beat_extract.py`, use this mode for instant results.
```bash
python eval_metrics/CBS.py \
  --beats_json_dir "Beats/my_experiment" \
  --window_size 0.07
```

**2. Batch Evaluate Audio Folder**
Analyzes all songs in a directory structure containing stem subfolders (e.g., `stem_0`, `stem_1`...).
```bash
python eval_metrics/CBS.py \
  --folder "/path/to/generated/data" \
  --stems stem_0 stem_1 stem_2 stem_3 \
  --num_workers 8
```

**3. Analyze a Single Song**
Useful for debugging or generating a visualization for a specific track.
```bash
python eval_metrics/CBS.py \
  --folder "/path/to/generated/data" \
  --track "Track001.wav" \
  --plot_folder "plots/"
```

#### Configuration Arguments
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--folder` | `None` | Root folder containing stem subdirectories. |
| `--beats_json_dir` | `None` | Path to folder containing pre-computed JSON beat files (takes precedence over `--folder`). |
| `--track` | `None` | Specific filename to analyze (e.g., `song_1.wav`). If omitted, analyzes all files. |
| `--window_size` | `0.15` | Tolerance window size (seconds) for defining synchronous beats. |
| `--overlap` | `0.0` | Overlap size (seconds) for the sliding window. |
| `--stems` | `stem_0`... | List of subfolder names representing different tracks. |
| `--beat_type` | `madmom` | Algorithm for beat tracking: `madmom` or `librosa`. |
| `--plot_folder` | `None` | Directory to save visualization plots (e.g., `multi_track_beats.png`). |

#### Output Metrics
- **mean_beat_ratio**: The primary metric. The average ratio of tracks that have a beat within the valid windows. Higher is better (more synchronized).
- **valid_window_count**: Number of time windows that contained at least one beat.
- **valid_track_count**: Average number of active tracks detected.

</details>



### 3. Intra-Rhythmic Stability (IRS)
`IRS.py` assesses the rhythmic consistency within individual tracks by analyzing the variance of beat intervals. It calculates the standard deviation of time differences between consecutive beats to measure how stable the tempo is within a single stem.

**Key Parameters:**
- `--beat_type`: `"madmom"` (recommended for accuracy) or `"librosa"`.
- `--sr`: Sampling rate (default: `22050`).

```bash
python eval_metrics/IRS.py --folder /path/to/generated/stems --beat_type madmom
```

<details>
<summary><strong>Click for detailed usage and full options</strong></summary>

#### Features
- **Stability Analysis**: Quantifies rhythmic stability using the standard deviation of beat intervals (lower values indicate more stable rhythm).
- **Flexible Inputs**: Supports analysis of raw audio folders, single audio files, or pre-computed JSON beat files.
- **Batch Processing**: Automatically aggregates results across multiple stems (`stem_0` to `stem_3`).

#### Additional Usage Examples

**1. Batch Evaluate Pre-computed JSON Beats (Recommended)**
If you have already extracted beats using `batch_beat_extract.py`, this method is instantaneous.
*Note: Expects files named `stem_0.json`, `stem_1.json`, etc., inside the folder.*
```bash
python eval_metrics/IRS.py \
  --beats_json_folder "Beats/my_experiment"
```

**2. Batch Evaluate Audio Folder**
Analyzes a root directory containing subfolders `stem_0`, `stem_1`, etc.
```bash
python eval_metrics/IRS.py \
  --folder "/path/to/generated/data" \
  --beat_type madmom
```

**3. Analyze a Single Audio File**
Useful for checking the stability of a specific track or loop.
```bash
python eval_metrics/IRS.py \
  --audio "path/to/drums.wav" \
  --beat_type madmom
```

#### Configuration Arguments
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--folder` | `None` | Root folder containing stem subdirectories (e.g., `stem_0/`). |
| `--beats_json_folder` | `None` | Path to folder containing pre-computed JSON beat files. |
| `--audio` | `None` | Path to a single audio file for individual analysis. |
| `--beat_type` | `librosa` | Algorithm for beat tracking: `madmom` (recommended) or `librosa`. |
| `--sr` | `22050` | Sampling rate used for loading audio. |
| `--madmom_fps` | `150` | Frames per second for Madmom processing. |

#### Output Metrics
- **avg_interval_std**: The primary metric for IRS. The standard deviation of the time intervals between beats. **Lower values indicate higher rhythmic stability.**
- **avg_interval_mean**: The average time between beats (correlated with Tempo).
- **avg_beat_count**: Average number of beats detected per track.

</details>

### 4. Batch Beat Extraction (Recommended)
`batch_beat_extract.py` allows you to pre-extract beats for an entire dataset or experiment output. 

We have found that `madmom` (using RNN-based downbeat tracking) provides significantly more precise beat detection for generated music compared to `librosa`. However, madmom's processing speed is relatively slow. Therefore, we **strongly recommend** pre-extracting beats for all generated data and saving them to JSON files before running other evaluation metrics.

**Key Parameters:**
- `--folder`: Path to the directory containing audio files or stem subfolders (`stem_0`, `stem_1`, etc.).
- `--out_dir`: Directory to save the resulting JSON files.
- `--fps`: Frames per second for Madmom (default: `150`).
- `--max_workers`: Number of threads for parallel processing.

```bash
python eval_metrics/batch_beat_extract.py --folder /path/to/generated/data --out_dir Beats/my_experiment --max_workers 8
```

The extracted JSON files can then be used directly with `CBD.py` using the `--beats_json_folder` argument, significantly speeding up the following evaluation process.

## 🔗 Citation

If you find this code or our paper useful for your research, please cite:

```bibtex
@inproceedings{wangsynctrack,
  title={SyncTrack: Rhythmic Stability and Synchronization in Multi-Track Music Generation},
  author={Wang, Hongrui and Zhang, Fan and Yu, Zhiyuan and Zhou, Ziya and Chen, Xi and Yang, Can and Wang, Yang},
  booktitle={The Fourteenth International Conference on Learning Representations}
}
```

## 🙏 Acknowledgements

This repository is built upon [MSG-LD](https://github.com/karchkha/MSG-LD?tab=readme-ov-file) and utilizes [Madmom](https://github.com/CPJKU/madmom) for beat tracking. We thank the authors for their open-source contributions.