# Iterative Training of a Minesweeper-Playing LLM via Failure-Driven Reward Engineering

[![Model](https://img.shields.io/badge/Model-Llama--3.1--8B--Instruct-blue)](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
[![Framework](https://img.shields.io/badge/Framework-Unsloth%20%2B%20TRL-green)](https://unsloth.ai/)
[![Hardware](https://img.shields.io/badge/Hardware-AMD%20Instinct%20MI300X-red)](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
[![Method](https://img.shields.io/badge/Method-SFT%20%2B%20GRPO-orange)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)]()

> **Submission for the AMD AI Reinforcement Learning Hackathon 2026**
> **Track:** Gaming the Models

---

## 📖 Abstract

We present an iterative, failure-driven approach to training **Llama-3.1-8B-Instruct** to play Minesweeper using pure JSON inputs and outputs. Unlike top-down reinforcement learning strategies, we employed a bottom-up methodology, evolving our pipeline through seven major iterations (v1-v7) on the **AMD Instinct MI300X**.

By combining **Supervised Fine-Tuning (SFT)** on expert demonstrations with **Group Relative Policy Optimization (GRPO)**, we successfully navigated critical failure modes—such as reward variance collapse and recursive flag loops—to produce a functional Minesweeper agent. Our final model achieves a **45% win rate** on standard 6x6 grids, utilizing a novel "Prefix-Forcing" prompt strategy to guarantee 100% syntactic validity.

---

## 🚀 Key Innovations

### 1. The "Prefix-Force" Prompting Strategy
The most significant barrier to RL training was the model's tendency to output conversational "analysis" text, which broke the JSON parser and resulted in a -5.0 reward loop.

* **Problem:** Prompts like "Output JSON" resulted in responses like *"Sure! Here is the JSON: {..."*
* **Solution:** We modified the training and inference prompts to end with the partial string: `Action: {"type": "`.
* **Impact:** This physically forces the model's next token to be a value (e.g., `flag` or `reveal`), rendering it impossible for the model to output conversational text. This single change increased valid generation from **0% to 100%**.

### 2. Logic-Biased Synthetic Data
Random board states are often unsolvable without guessing. Training on random data teaches the model that "guessing is necessary," preventing convergence.

* **Solution:** We implemented a custom `generate_data()` engine using a constraint-satisfaction oracle (CSP). It generates *only* board states where a logical move is guaranteed to exist.
* **Impact:** This ensures the GRPO reward signal is always attributable to skill, not luck.

### 3. Safe-Mode Memory Management
The MI300X is powerful but sensitive to memory fragmentation with PyTorch/Unsloth.

* **Solution:** We implemented a custom "Safe Mode" boot sequence:
    * Explicit `gc.collect()` and `empty_cache()` pre-hooks.
    * Environment variable `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`.
    * Explicit device mapping `device_map={"": 0}` to prevent Meta-Tensor initialization errors.

---

## 🛠️ Technical Architecture

### System Overview

We employed a two-stage training pipeline designed to stabilize the model before teaching it strategy.

1.  **Expert Logic Solver:** A constraint-satisfaction algorithm generates "solvable" board states where no guessing is required.
2.  **SFT (Syntax Grounding):** Teaches the model the JSON schema and the "Prefix-Force" output format.
3.  **GRPO (Strategy Optimization):** Uses Group Relative Policy Optimization to incentivize winning behaviors (flagging mines) over simple validity.

### Hardware Stack Optimization

| Component | Specification | Optimization |
| :--- | :--- | :--- |
| **GPU** | AMD Instinct MI300X | **192GB HBM3** utilized for large batch sizes |
| **Precision** | BFloat16 | Native support for high dynamic range training |
| **Attention** | Eager Implementation | Fixes SDPA crashes on ROCm stack |
| **Library** | Unsloth | 2x Faster training, minimized memory footprint |

---

## ⚙️ Methodology & Pipeline

### Phase 1: Supervised Fine-Tuning (SFT)
* **Objective:** Teach JSON syntax and basic board comprehension.
* **Dataset:** 1,000 samples of Logic-Biased mid-game states.
* **Config:** LoRA Rank 16, Alpha 32.
* **Result:** Model learns to output coordinates within bounds and adhere to the "Prefix-Force" format.

### Phase 2: Group Relative Policy Optimization (GRPO)
* **Objective:** Incentivize winning strategy.
* **Loss Normalization:** Dr. GRPO technique.
* **Reward Function Engineering:**
    * **Valid JSON:** +1.0 (Baseline).
    * **Logic Flag:** +15.0 (Primary goal - Correctly identifying a mine).
    * **Logic Reveal:** +10.0 (Secondary goal - Clearing safe space).
    * **Invalid Move (Death):** -25.0 (Hitting a mine).
    * **Stagnation:** -10.0 (Repeating moves or flagging safe cells).

---

## 🔬 Failure Mode Analysis

Our development was driven by identifying and resolving five specific failure modes:

1.  **Format Hallucination:** Solved via "Prefix-Forcing" prompt strategy.
2.  **Random Selection:** Solved by replacing random rollout data with Logic-Oracle data.
3.  **Reward Variance Collapse:** Solved by removing length-based penalties.
4.  **Recursive Flag Loops:** Solved by implementing state-memory in the environment to penalize toggling.
5.  **KL Divergence Explosion:** Solved by lowering learning rate ($2e^{-5}$) and using gradient clipping.

---

## 📂 Repository Structure

```text
├── minesweeper_grpo.ipynb      # Main training pipeline (SFT + GRPO)
├── minesweeper_model.py        # Inference wrapper for the trained model
├── minesweeper_agent.py        # Agent logic for game interaction
├── agent_server.py             # Server for handling game state I/O
├── documentation.pdf           # Detailed project report and findings
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```
## 💻 Installation & Usage Guide

### Prerequisites
* Python 3.10+
* AMD ROCm 6.0+ (or CUDA if not using MI300X features)
* Unsloth AI

### 1. Environment Preparation
Ensure you are in the root workspace before running the following commands.

```bash
# Verify GPU availability
rocm-smi

# Install dependencies
pip install unsloth trl torch
```
### 2. Training

To replicate the training process, launch the provided Jupyter notebook:

```bash
jupyter notebook minesweeper_grpo.ipynb
```
Note: The notebook includes a "Safe Mode" boot sequence to handle MI300X memory allocation. Ensure you run Cell 1 to reset memory before training.

### 3. Inference
To run the agent against the local evaluation server:
```bash
python agent_server.py --config minesweeper_config.yaml
```

## 👥 Team

- Ritvik Shrivastava  
- J Bharath Reddy  
- Prashasth Immanuel  
- Kamal Enoch  

## 🙏 Acknowledgments

We thank the Yardi School of Artificial Intelligence (IIT Delhi) for organizing this competition.

Special thanks to <b>AMD</b> for providing access to the powerful Instinct MI300X
