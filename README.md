# Constraint-Aware Tool Use (CTU)

This repository contains the implementation of **Constraint-Aware Tool Use (CTU)**, a framework for improving the reliability of LLM agents in multi-step tool-use scenarios.

CTU enforces constraints during execution and provides structured signals, such as feedback (reactive) and guidance (proactive), to influence agent behavior.

---

## Features

- Multiple constraint engines:
  - No constraints (baseline)
  - JSON-based constraints
  - Knowledge graph-based constraints
- Support for:
  - Access, temporal, and contextual constraints
  - Execution-time feedback and guidance
- End-to-end evaluation pipeline

---

## Setup

### Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) for local LLM inference

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### Dataset

#### Lightweight Sample (included)

We provide a small sample dataset:

```
data/sample.json
```

This subset is intended for **quick reproducibility** and runs in a few minutes.

---


#### Full Dataset

The full dataset is derived from [ToolBench](https://github.com/OpenBMB/ToolBench).

 Download data.zip from: [Google Drive](https://drive.google.com/drive/folders/1TysbSWYpP8EioFu9xPJtpbJZMLLmwAmL). Extract the required file `data/toolllama_G123_dfs_train.json` to the project folder `data/`.

---


### Configuration

Experiments are controlled via `config.yaml`.

Example:

```
execution:
  constraint_engines: ["kg"]   # no | json | kg

  capabilities:
    enforce: true
    feedback: true
    guidance: true
```

---

### Preprocessing

```bash
python -m src.data_processing.build_dataset
```

---

### Running Experiments

```bash
python main.py
```

---

### Results

Sample results are included under:

```
results/
```

---

### Notes

- Tool outputs are simulated from dataset traces.
- Experiments rely on local inference via Ollama.
- The sample configuration is designed for quick execution; full runs may require significantly more time.
<!--
---

### Citation

```
@inproceedings{ctu2026,
  title={Constraint-Aware Tool Use for LLM Agents},
  author={...},
  booktitle={ICSE},
  year={2026}
}
```
-->