<div align="center">

# SyncTrack: Rhythmic Stability and Synchronization in Multi-Track Music Generation

<!-- <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://lightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a> -->
[![Paper](https://img.shields.io/badge/arxiv-Paper-b31b1b.svg)](https://arxiv.org/abs/2603.01101)
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
> **Note:** For evaluation metrics, ensure you have the necessary audio processing libraries `madmom` installed on your system.


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

To train the SyncTrack model, use the `train_synctrack.py` script. The configuration is managed via YAML files.

### Configuration Setup
In `config/synctrack_train.yaml`, please configure the following paths before starting:
- `data.params.path.train_data`: Path to your **training** dataset.
- `data.params.path.valid_data`: Path to your **validation** dataset.
- `model.params.ckpt_path`: (Optional) Path to a pre-trained checkpoint to resume training.

### Run Training
```bash
python train_synctrack.py --config config/synctrack_train.yaml
```

## ⚡ Inference & Evaluation

To generate samples or evaluate the model on the test set, use the `eval_synctrack.py` script.


### Configuration Setup
In `config/synctrack_eval.yaml`, please configure the following:

- `mode`: Ensure this is set to `test`.
- `data.params.path.valid_data`: Path to your **test** dataset.
- `model.params.ckpt_path`: Path to the model checkpoint.

> 🌟 **Use [Our Pre-trained Model](https://zenodo.org/records/18797721?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjlhYzY4MDRiLWE4YzktNGRkOC05MzQwLTEwYzc5ZTI2MjM4MCIsImRhdGEiOnt9LCJyYW5kb20iOiIxOGUxMTQxNTI1ZGQyZmU0NGZjYjFmZjM0OThiNzJlNiJ9.iPuxzy7aIE1HFs0q1EVsVc3r87Mq5FyizZPnDr21-Nu6hpTbfpb7omGNZyg4G3JckMBs_qOZTuLSi5IiWONtqw)**


### Run Inference
```bash
python eval_synctrack.py --config config/synctrack_eval.yaml
```

## 📊 Evaluation Metrics

We provide a comprehensive suite of metrics to measure rhythmic stability and synchronization, located in the `eval_metrics` directory.

### 1. Cross-track Beat Dispersion  (CBD)
`CBD.py` quantifies rhythmic synchronization in multitrack music by measuring the dispersion of beat alignment across all pairs of tracks.

**Key Parameters:**
- `--folder`: Path to the directory containing audio stem subfolders (`stem_0`, `stem_1`, `stem_2` and `stem_3`).

```bash
python eval_metrics/CBD.py --folder /path/to/generated/stems
```



### 2. Cross-track Beat Synchronization (CBS)
`CBS.py` measures rhythmic synchronization among multiple tracks.

**Key Parameters:**
- `--folder`: Path to the directory containing audio stem subfolders (`stem_0`, `stem_1`, `stem_2` and `stem_3`).
- `--window_size`: Length of the sliding window in seconds (default: `0.15`).


```bash
python eval_metrics/CBS.py --folder /path/to/generated/stems
```




### 3. Inner-track Rhythmic Stability (IRS).
`IRS.py` quantifies temporal consistency by averaging the standard deviation of the Inter-Beat Interval across all samples for each track.

**Key Parameters:**
- `--folder`: Path to the directory containing audio stem subfolders (`stem_0`, `stem_1`, `stem_2` and `stem_3`).

```bash
python eval_metrics/IRS.py --folder /path/to/generated/stems
```



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

## 📧 Contact

Please feel free to contact Hongrui Wang (hwangfb@connect.ust.hk), Fan Zhang (mafzhang@ust.hk), or Prof. Can Yang (macyang@ust.hk) if you have any questions.