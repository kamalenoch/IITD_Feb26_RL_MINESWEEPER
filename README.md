# Iterative Training of a Minesweeper-Playing LLM via Failure-Driven Reward Engineering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model: Llama-3.1-8B](https://img.shields.io/badge/Model-Llama--3.1--8B-blue)](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
[![Framework: Unsloth](https://img.shields.io/badge/Framework-Unsloth-green)](https://unsloth.ai)
[![Hardware: AMD ROCm](https://img.shields.io/badge/Hardware-AMD%20MI300X-red)](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)

> **Submission for the AMD AI Reinforcement Learning Hackathon 2026** > *Hosted by AMD, Yardi School of AI (IIT Delhi), and Unsloth.*

---

## 📖 Abstract

This project presents an iterative, failure-driven approach to training **Llama-3.1-8B-Instruct** to play Minesweeper using pure JSON inputs and outputs. Unlike top-down reinforcement learning strategies, we employed a bottom-up methodology, evolving our pipeline through seven major iterations (v1-v7).

By combining **Supervised Fine-Tuning (SFT)** on expert demonstrations with **Group Relative Policy Optimization (GRPO)**, we successfully navigated critical failure modes—such as reward variance collapse and recursive flag loops—to produce a functional Minesweeper agent.

---

## 🚀 Key Features

* **Hybrid Training Pipeline:** A two-stage approach utilizing SFT for syntax grounding and GRPO for strategic reasoning.
* **Logic-Driven Data Generation:** 8,000+ expert demonstrations generated using a constraint satisfaction oracle (CSP) to prioritize provably safe reveals.
* **Hardware Optimization:** Optimized for **AMD Instinct MI300X** GPUs using the ROCm stack and Unsloth for efficient LoRA training.
* **Custom Reward Engineering:** Implementation of "Reveal-Biased" reward functions with harsh penalties for state-toggling to prevent infinite loops.



---

## 🛠️ Methodology

### 1. Stage 1: Supervised Fine-Tuning (SFT)
Before applying RL, we established a baseline using SFT.
* **Dataset:** 8,000 samples of mid-game states.
* **Objective:** Teach the model strict JSON schema adherence and basic 1-1 logical patterns.
* **Outcome:** Achieved 100% valid JSON output rate, eliminating the "Hallucination" failure mode.

### 2. Stage 2: Group Relative Policy Optimization (GRPO)
We utilized GRPO to incentivize winning behavior over simple validity.
* **Model:** Llama-3.1-8B-Instruct (LoRA Rank 32, Alpha 64).
* **Loss Normalization:** Dr. GRPO technique.
* **Reward Function:**
    * **Safe Reveal:** +20 points (Primary incentive).
    * **Correct Flag:** +15 points.
    * **Flag Toggling:** -15 points (To prevent recursive loops).
    * **Invalid Move:** -25 points.

---

## 🔬 Failure Mode Analysis

Our development was driven by identifying and resolving five specific failure modes:

1.  **Format Hallucination:** Solved via SFT on strict JSON datasets.
2.  **Random Selection:** Solved by replacing random rollout data with Logic-Oracle data.
3.  **Reward Variance Collapse:** Solved by removing length-based penalties.
4.  **Recursive Flag Loops:** Solved by implementing state-memory in the environment to penalize toggling.
5.  **KL Divergence Explosion:** Solved by lowering learning rate ($2e^{-5}$) and using gradient clipping.

---

## 📂 Repository Structure

```text
├── minesweeper_grpo_final.ipynb # Main training pipeline (SFT + GRPO)
├── minesweeper_model.py         # Inference wrapper for the trained model
├── minesweeper_agent.py         # Agent logic for game interaction
├── agent_server.py              # Server for handling game state I/O
├── documentation.pdf            # Detailed project report and findings
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
