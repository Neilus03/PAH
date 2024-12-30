# Prototype-Augmented Hypernetworks for Continual Multitask Learning

 
This repository contains the code and experiments for **Prototype-Augmented Hypernetworks (PAH)**, a method aimed at continual multitask learning without large memory overhead or replay buffers.

## What Is PAH?

PAH tackles the challenge of learning multiple tasks over time, where new tasks arrive sequentially and the system must learn them without forgetting what it learned before. Traditional methods often need separate sets of parameters for each task or require replaying old data to avoid forgetting. PAH, on the other hand, avoids these issues by using two key ideas:

1. **Hypernetworks:**  
   Instead of storing a dedicated classifier head for each task, PAH uses a hypernetwork. This hypernetwork generates the classifier weights for any given task as needed. This design keeps the total number of parameters roughly constant, no matter how many tasks you learn.

2. **Learnable Prototypes:**  
   Each task is associated with small prototype images. These prototypes capture the essence of a task’s classes. Because the main network and classifier remain fixed, the prototypes themselves learn to represent each task’s features in a way that fits the model’s stable representation space. This allows PAH to adapt to new tasks without altering previously learned weights and without storing raw data from older tasks.

## Key Advantages

- **Scalability:**  
  As the number of tasks grows, PAH does not need to store a separate head or a large buffer of old samples for each task. The model parameters stay efficient and manageable.

- **Reduced Forgetting:**  
  By fixing the main network and letting prototypes adapt, PAH preserves past knowledge more effectively. It achieves near-zero forgetting while maintaining strong performance on new tasks.

- **Backbone Flexibility:**  
  PAH works with a variety of backbone architectures (like ResNet or MobileNet), demonstrating that the method is flexible and not tied to a specific type of network.

## Results

Experiments on standard benchmarks (like Split-CIFAR100 and TinyImageNet) show that PAH achieves high accuracy and very low forgetting, often outperforming established baselines in continual learning.

## Getting Started

Please refer to the code and instructions in this repository to set up the environment, run the training scripts, and reproduce our results.

---
