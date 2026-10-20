<a id="top"></a>

<h1 align="center">
  MSCR: Jointly Balancing Utilization Across Modalities<br>
  and Discovering Synergistic Information
</h1>

<p align="center">
  <strong>Xinyu Chen, Liangjian Wen, Jiang Duan, Dongkai Wang, Yong Dai,<br>
  Jiayu Bai, Jianzhuang Liu, Zhen Tian, Guoping Qiu, Zhao Kang</strong>
</p>

<p align="center">
  <a href="https://neurips.cc/virtual/2026/loc/sydney/poster/150032">
  <img src="https://img.shields.io/badge/NeurIPS-2026-008080?style=flat-square" alt="NeurIPS 2026">
</a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://github.com/chenxinyu991-collab/MSCR"><img src="https://img.shields.io/badge/Code-GitHub-181717?style=flat-square&logo=github" alt="Code on GitHub"></a>
</p>

<p align="center">
  Official PyTorch implementation of <strong>MSCR</strong>, accepted at <strong>NeurIPS 2026</strong>.
</p>

<p align="center">
  <a href="#tldr">TL;DR</a> &nbsp;·&nbsp;
  <a href="#key-results">Key Results</a> &nbsp;·&nbsp;
  <a href="#our-approach">Our Approach</a> &nbsp;·&nbsp;
  <a href="#quick-start">Quick Start</a> &nbsp;·&nbsp;
  <a href="#citation">Citation</a> &nbsp;·&nbsp;
  <a href="#acknowledgments">Acknowledgments</a>
</p>

---

<a id="tldr"></a>

## 🎯 TL;DR

<p align="center">
  <img src="figs/Method_concept_MSCR.png" width="100%" alt="MSCR overview: latent-space permutations, an additive unimodal reference, anti-dominance and synergy objectives, and proficiency-aware gating">
  <br>
  <em>MSCR jointly promotes balanced modality utilization and predictive gains beyond unimodal aggregation.</em>
</p>

> **Balanced modality utilization does not automatically lead to synergistic fusion.**

MSCR (**Multimodal Synergistic Competition Regularizer**) builds on MCR to help the fusion network discover predictive information that independent unimodal evidence cannot provide.

- ✅ **Limit modality dominance:** penalize excessive sensitivity of the fused prediction to any single modality.
- ✅ **Reward synergistic gain:** encourage the joint predictor to outperform an additive unimodal reference by a margin.
- ✅ **Focus on difficult samples:** use a proficiency-aware gate to strengthen regularization when unimodal evidence is insufficient.
- ✅ **Improve across settings:** outperform MCR in all nine evaluated settings across five benchmarks, with higher conditional recovery in every setting.

<a id="key-results"></a>

## 📊 Key Results

### Main Benchmark Results

Accuracy (%) from Table 1 of the paper. Values are mean ± standard deviation over three folds for CREMA-D and UCF101, or three random seeds for AVE, MOSI, and MOSEI. MOSI and MOSEI are trained as regression-style tasks and evaluated by binary accuracy. **V**: video; **A**: audio; **T**: text. Gains are absolute percentage points.

| Dataset | Backbone | Modalities | MCR | MSCR (ours) | Gain |
| :--- | :--- | :---: | ---: | ---: | ---: |
| CREMA-D | ResNet | V-A | 74.5 ± 0.8 | **75.7 ± 1.1** | +1.2 |
| CREMA-D | Conformer | V-A | 85.1 ± 0.5 | **86.4 ± 0.9** | +1.3 |
| AVE | ResNet | V-A | 73.5 ± 0.0 | **74.5 ± 0.2** | +1.0 |
| AVE | Conformer | V-A | 87.4 ± 0.2 | **89.3 ± 0.3** | +1.9 |
| UCF101 | ResNet | V-A | 52.9 ± 2.4 | **54.4 ± 2.4** | +1.5 |
| CMU-MOSI | Transformer | V-T | 75.3 ± 0.7 | **76.8 ± 0.8** | +1.5 |
| CMU-MOSI | Transformer | V-A-T | 78.6 ± 0.2 | **80.1 ± 0.5** | +1.5 |
| CMU-MOSEI | Transformer | V-T | 80.8 ± 0.4 | **81.3 ± 0.5** | +0.5 |
| CMU-MOSEI | Transformer | V-A-T | 81.1 ± 0.4 | **81.8 ± 0.6** | +0.7 |

MCR results above were reproduced in our study. This table compares MSCR with MCR; see the paper for comparisons with all baselines.

### Can Fusion Succeed When Unimodal Predictors Fail?

Conditional recovery measures the accuracy of the joint predictor on samples misclassified by **all auxiliary unimodal predictors of the same trained model**. It is a behavioral proxy for synergistic capability.

| Dataset | Backbone | Modalities | MCR recovery | MSCR recovery |
| :--- | :--- | :---: | ---: | ---: |
| CREMA-D | ResNet | V-A | 10.13% | **18.35%** |
| CREMA-D | Conformer | V-A | 5.92% | **9.98%** |
| AVE | ResNet | V-A | 5.50% | **7.07%** |
| AVE | Conformer | V-A | 6.93% | **12.95%** |
| UCF101 | ResNet | V-A | 10.28% | **15.45%** |
| CMU-MOSI | Transformer | V-T | 4.85% | **34.35%** |
| CMU-MOSI | Transformer | V-A-T | 10.87% | **17.44%** |
| CMU-MOSEI | Transformer | V-T | 7.12% | **12.50%** |
| CMU-MOSEI | Transformer | V-A-T | 6.71% | **13.69%** |

MSCR also improves recovery with Transformer-based fusion in all four tested settings (Table 4). On CREMA-D with Conformer encoders, recovery increases from **11.82% to 25.66%**.

### Independent Cross-Modal Interaction Analysis

Global InterSHAP measures the relative contribution of cross-modal interactions to model outputs. It complements conditional recovery without requiring auxiliary unimodal predictions or ground-truth labels.

| Dataset / Backbone | MCR | MSCR (ours) |
| :--- | ---: | ---: |
| CREMA-D / Conformer | 5.22 ± 1.33 | **10.31 ± 1.89** |
| AVE / Conformer | 4.74 ± 0.93 | **9.16 ± 1.28** |
| UCF101 / ResNet | 12.10 ± 0.67 | **18.21 ± 1.12** |

Values are percentages, reported as mean ± standard deviation in Table 3. All results in this section are from the paper.

<a id="our-approach"></a>

## 🧩 Our Approach

### From Modality Balancing to Synergistic Fusion

1. **Encode and predict.** Modality-specific encoders produce latent features; the fusion network and auxiliary unimodal heads produce joint and unimodal predictions.
2. **Limit dominance.** Permute each modality's latent features within a batch, measure prediction changes using Jensen–Shannon divergence, and penalize the smooth maximum of these sensitivities.
3. **Reward joint predictive gain.** Average the unimodal predictions to form an additive reference, then require the joint predictor to achieve a lower task loss by a margin.
4. **Adapt to sample difficulty.** Increase regularization when even the best unimodal predictor has a high loss, and reduce it for samples already explained by unimodal evidence.

### Training Objective

$$
\mathcal{L}_{\mathrm{total}}=\mathcal{L}_{\mathrm{task}}+\mathcal{L}_{\mathrm{MCR}}+\mathcal{L}_{\mathrm{MSCR}},
\qquad
\mathcal{L}_{\mathrm{MSCR}}=\frac{1}{B}\sum_{b=1}^{B}g^{(b)}\left(\lambda_{\mathrm{dom}}\mathcal{L}_{\mathrm{dom}}^{(b)}+\lambda_{\mathrm{syn}}\mathcal{L}_{\mathrm{syn}}^{(b)}\right).
$$

For two-modality classification, the synergy objective and gate are:

$$
p_{\mathrm{ref}}=\tfrac12(p_1+p_2),\qquad
\mathcal{L}_{\mathrm{syn}}=\max\left(0,m+\ell_{\mathrm{joint}}-\ell_{\mathrm{ref}}\right),\qquad
g=\sigma\left(\kappa(\min(\ell_1,\ell_2)-\tau)\right).
$$

The task loss supervises both joint and unimodal predictions. The MCR regularizer is retained. **MSCR's additional gradients update only the fusion network:** latent features, the unimodal reference, and the gate are detached for this branch. The regularizer operates within a configured activation interval after the initial training stage. The paper also describes trimodal and regression variants.

<a id="quick-start"></a>

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/chenxinyu991-collab/MSCR.git
cd MSCR
```

Use a dedicated Python environment. Install a compatible PyTorch, torchvision, and torchaudio build for your CUDA setup, then install the supplied dependencies:

```bash
python -m pip install -r requirement.txt
```

The dependency file is named **`requirement.txt`** (singular). It does not install PyTorch itself; the training code also imports `torchmetrics`, which must be available in your environment. The supplied dependency list has not yet been validated as a complete installation in a clean environment.

### 2. Prepare Data and Paths

Prepare the datasets using the preprocessing and splits described in Appendix B of the paper. CREMA-D uses actor-disjoint splits; UCF101 uses the 51-category audio-visual subset; MOSI and MOSEI use aligned MultiBench features.

| Dataset / Backbone | Default configuration | MSCR configuration |
| :--- | :--- | :--- |
| CREMA-D / ResNet | [`default_config_cremad_res.json`](configs/CREMA_D/default_config_cremad_res.json) | [`MSCR.json`](configs/CREMA_D/release/res/MSCR.json) |
| CREMA-D / Conformer | [`default_config_cremad_vit.json`](configs/CREMA_D/default_config_cremad_vit.json) | [`MSCR.json`](configs/CREMA_D/release/vit/MSCR.json) |
| AVE / ResNet | [`default_config_ave_res.json`](configs/AVE/default_config_ave_res.json) | [`MSCR.json`](configs/AVE/release/res/MSCR.json) |
| AVE / Conformer | [`default_config_ave_vit.json`](configs/AVE/default_config_ave_vit.json) | [`MSCR.json`](configs/AVE/release/vit/MSCR.json) |
| UCF101 / ResNet | [`default_config_ucf.json`](configs/UCF/default_config_ucf.json) | [`MSCR.json`](configs/UCF/res/MSCR.json) |
| MOSI / V-T | [`default_config_mosi_VT.json`](configs/FactorCL/Mosi/default_config_mosi_VT.json) | [`MSCR.json`](configs/FactorCL/Mosi/release/VT/MSCR.json) |
| MOSI / V-A-T | [`default_config_mosi_VTA.json`](configs/FactorCL/Mosi/default_config_mosi_VTA.json) | [`MSCR.json`](configs/FactorCL/Mosi/release/VTA/MSCR.json) |
| MOSEI / V-T | [`default_config_mosei_VT.json`](configs/FactorCL/Mosei/default_config_mosei_VT.json) | [`MSCR.json`](configs/FactorCL/Mosei/release/VT/MSCR.json) |
| MOSEI / V-A-T | [`default_config_mosei_VTA.json`](configs/FactorCL/Mosei/default_config_mosei_VTA.json) | [`MSCR.json`](configs/FactorCL/Mosei/release/VTA/MSCR.json) |

Before training, update the following fields for your machine:

| Field | Purpose |
| :--- | :--- |
| `dataset.data_roots` | Prepared dataset directory or aligned feature file |
| `dataset.norm_wav_path`, `dataset.norm_face_path` | Normalization files where used |
| `model.save_base_dir` | Checkpoint directory |
| `model.encoders[*].pretrainedEncoder` | Whether to load unimodal checkpoints, and their filenames |
| `model.args.pretraining_paths` | Additional pretrained-model paths where used |
| `training_params.gpu_device` | GPU selection |
| `training_params.wandb_disable` | Set to `true` to disable Weights & Biases logging |

The default configurations contain machine-specific absolute paths. MSCR configurations that enable pretrained encoders require the corresponding unimodal checkpoints. Train the `unimodal_*.json` configurations in the same experiment directory first, or provide compatible checkpoints and update their paths. Checkpoint filenames incorporate fold and applicable command-line overrides.

### 3. Configure MSCR

MSCR retains the internal code name **FSR**, including `MCR_Model_FSR`, `MCR_FSR`, and `fsr_*` configuration keys.

| Key under `model.args.bias_infusion` | Meaning |
| :--- | :--- |
| `use_fsr` | Enable the additional MSCR regularizer |
| `fsr_lambda_dom`, `fsr_lambda_syn` | Anti-dominance and synergy weights |
| `fsr_margin` | Required joint-prediction advantage |
| `fsr_tau_dom` | Smooth-maximum sharpness |
| `fsr_use_gate` | Enable proficiency-aware gating |
| `fsr_gate_kappa`, `fsr_gate_tau` | Gate sharpness and threshold |
| `fsr_starting_epoch`, `fsr_ending_epoch` | Regularization activation interval |

**Gate setting:** some supplied `MSCR.json` files, including CREMA-D / ResNet and MOSI / V-T, set `fsr_use_gate` to `false`. Set it to `true` to enable the paper's gated formulation. Verify the complete hyperparameter configuration against the paper before reproducing a reported result; enabling the gate alone does not establish an exact reproduction configuration.

### 4. Training

After completing the data, checkpoint, and configuration setup, launch an experiment from the repository root:

**CREMA-D · ResNet · Video + Audio**

```bash
python train.py \
  --config configs/CREMA_D/release/res/MSCR.json \
  --default_config configs/CREMA_D/default_config_cremad_res.json \
  --fold 0
```

**CMU-MOSI · Transformer · Video + Text**

```bash
python train.py \
  --config configs/FactorCL/Mosi/release/VT/MSCR.json \
  --default_config configs/FactorCL/Mosi/default_config_mosi_VT.json \
  --fold 0
```

For other experiments, use the corresponding configuration pair in the table above. Experiment-specific settings override the default JSON; command-line options such as `--lr`, `--wd`, and `--batch_size` override the merged settings. These examples launch a single run; reporting paper results requires the appropriate folds or random seeds.

### 5. Evaluation

[`analyze_best_checkpoint.py`](analyze_best_checkpoint.py) evaluates a specified checkpoint. Use the same architecture, data split, and effective configuration as training:

```bash
python analyze_best_checkpoint.py \
  --config configs/CREMA_D/release/res/MSCR.json \
  --default_config configs/CREMA_D/default_config_cremad_res.json \
  --fold 0 \
  --ckpt /path/to/checkpoint.pth.tar \
  --test_only
```

Replace the checkpoint path with your saved model. The evaluation script reports missing and unexpected state-dictionary keys; inspect them before interpreting metrics. Its configuration builder does not apply all of `train.py`'s filename and path substitutions, so resolve any `{}` placeholders in pretrained checkpoint paths in your evaluation configuration beforehand.

<a id="citation"></a>

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{chen2026mscr,
  title={MSCR: Jointly Balancing Utilization Across Modalities and Discovering Synergistic Information},
  author={Chen, Xinyu and Wen, Liangjian and Duan, Jiang and Wang, Dongkai and
          Dai, Yong and Bai, Jiayu and Liu, Jianzhuang and Tian, Zhen and
          Qiu, Guoping and Kang, Zhao},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

<a id="acknowledgments"></a>

## 🙏 Acknowledgments

This work builds on [MCR](https://github.com/kkontras/MCR), *Balancing Multimodal Training Through Game-Theoretic Regularization*. We thank its authors for their research and open-source implementation. Our interaction analysis uses [InterSHAP](https://github.com/LauraWenderoth/InterSHAP).

## ✉️ Contact

For questions or feedback, please [open an issue](https://github.com/chenxinyu991-collab/MSCR/issues) or contact [Xinyu Chen](mailto:chenxinyu991@gmail.com).

<p align="center">
  <a href="#top">↑ Back to top</a>
</p>
