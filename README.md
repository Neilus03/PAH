# TSR: Dynamic Tangent Space Realignment for Continual Multi-Task Learning with Task Vectors 
![TANGENT SPACE REALIGNMENT](https://github.com/user-attachments/assets/2bbc0243-fc65-422c-a1fe-f51003a21cea)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch 2.0](https://img.shields.io/badge/pytorch-2.0-orange.svg)](https://pytorch.org/)

This repository contains the official PyTorch implementation of **TSR (Tangent Space Realignment)**, a novel continual multi-task learning algorithm introduced in "[Paper Title - Link to arXiv or published paper when available]". TSR leverages task vectors and dynamic tangent space realignment via the Neural Tangent Kernel (NTK) to enable efficient and scalable continual learning without catastrophic forgetting.

## Key Features

* **Adaptive NTK Linearization:** Dynamically recalibrates the tangent space after each task, ensuring task vector compatibility with the evolving model.
* **Task Vector Disentanglement:**  Maintains the distinctness of task-specific knowledge, minimizing interference and promoting weight disentanglement.
* **Memory Efficiency:**  Avoids storing full model copies for each task, making it suitable for memory-constrained environments.
* **Scalability:**  Handles a growing number of tasks efficiently through periodic tangent space updates and task vector reprojection.
* **Robustness:**  Mitigates catastrophic forgetting and enhances performance stability in continual multi-task learning scenarios.
