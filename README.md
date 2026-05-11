# ⚡ FengWu-Lite

Lightweight adaptation of the FengWu weather forecasting model, designed to run on **single GPU (RTX A1000 / Colab)**.

---

## 🎯 Goal

Make large-scale weather models:

- Trainable on limited hardware
- Memory efficient
- Adapted to regional data

---

## 🧠 Model

Architecture:

Encoder → Transformer (LG_net) → Decoder

Key features:

- Multi-modal inputs (surface + pressure)
- Window attention (Swin-like)
- Probabilistic output (mean + uncertainty)
- Autoregressive prediction

---

## 📊 Data

ERA5 variables:

- Surface: `u10, v10, t2m, msl`
- Pressure (37 levels): `z, t, u, v, q`

Data format:

```

Input  : [2, 189, 64, 64]
Target : [N, 189, 64, 64]

```

- N = 4 → 1 day
- N = 28 → 7 days

---

## 🚀 Installation

```bash
git clone https://github.com/habibatp/FengWu-RTX-A1000.git
cd FengWu-RTX-A1000
pip install -r requirements.txt
```

---

## 🧪 Test

```bash
python test_memory_gpu.py
```

---

## 🏋️ Training

```bash
python train_optimized.py -c config/fengwu_local_8gb.yaml
```

---

## ⚙️ Key Config

```yaml
input_steps: 2
target_steps: 4
length: 6
sample_stride: 1
```



│   README.md
│   requirements.txt
│   test_memory_gpu.py
│   train_optimized.py
│   
├───config
│       fengwu_local_8gb.yaml
│       
├───datasets
│       era5_npy_f32.py
│       
├───models
│       model.py
│       MTS2d_model.py
│       
├───networks
│   │   LGUnet_all.py
│   │   
│   └───utils
│           Attention.py
│           Blocks.py
│           positional_encodings.py
│           utils.py
│           
├───replay
│       replay_buff.py
│       
└───utils
        builder.py
        distributedsample.py
        logger.py
        misc.py
        __init__.py 

---

## ⚡ Notes

* Use AMP for GPU efficiency
* Start with `target_steps=4`
* Increase for longer forecasts

---

## 📌 Summary

FengWu-Lite enables **weather forecasting with limited resources** while keeping core transformer-based design.

