# TCN vs RNN for Long-Range Dependencies: A Benchmarking Study

## Overview

This project reproduces and extends the paper *"An Empirical Evaluation of Convolutional vs Recurrent Networks for Sequence Modeling"* by systematically benchmarking **Temporal Convolutional Networks (TCN)** against **RNN, LSTM, and GRU** architectures.

The focus is on evaluating the ability of different sequence models to capture **long-range dependencies** across synthetic and real-world-inspired tasks.

---

## Key Contributions

* Implemented **TCN, RNN, LSTM, and GRU** in PyTorch within a unified experimental framework
* Demonstrated **TCN’s superiority in long-range dependency modeling**, achieving:

  * **100% accuracy on Copy Memory (T ≥ 500)**
  *  Recurrent models failed (~10–15% accuracy)
* Built a **modular sequence modeling framework** supporting:

  * Classification
  * Sequence-to-sequence tasks
  * Language modeling
* Improved training stability using:

  * Causal dilated convolutions
  * Residual connections
  * Gradient clipping
  * Proper weight initialization

---

## Datasets

* **Sequential MNIST** – sequence classification task
* **Copy Memory Task** – long-range dependency benchmark
* **Penn Treebank (PTB)** – language modeling

---

## Tech Stack

* Python
* PyTorch
* NumPy, Pandas
* Matplotlib

---

## Results Summary

| Model | Copy Memory (T=500+) | Key Insight                            |
| ----- | -------------------- | -------------------------------------- |
| TCN   | **100% Accuracy**    | Captures long dependencies effectively |
| RNN   | ~10–15%              | Fails due to vanishing gradients       |
| LSTM  | ~10–15%              | Struggles at very long sequences       |
| GRU   | ~10–15%              | Similar limitations as LSTM            |

---

## Key Insights

* **TCNs outperform recurrent architectures** in long-range sequence tasks
* Dilated causal convolutions enable **large receptive fields with stable gradients**
* Residual connections significantly improve training stability
* Recurrent models struggle despite gating mechanisms

---

## Future Work

* Extend to **real-world time-series anomaly detection (NAB dataset)**
* Benchmark against **Transformer-based architectures**
* Explore hybrid architectures combining convolution + attention

---

## How to Run

```bash
git clone https://github.com/your-username/tcn-vs-rnn-sequence-modeling.git
cd tcn-vs-rnn-sequence-modeling
pip install -r requirements.txt
python train.py
```

---

## Reference

Bai, Shaojie, J. Zico Kolter, and Vladlen Koltun.
*"An Empirical Evaluation of Convolutional vs Recurrent Networks for Sequence Modeling"*
