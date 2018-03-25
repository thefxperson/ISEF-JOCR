# ISEF-JOCR
Improving Optical Character Recognition (OCR) of Japanese characters through one-shot learning and a context-aware system.
## Results
#### 11/11 MNIST - Maximum Likelihood Estimation - 11.35% accuracy

#### 11/21 MNIST - LSTM (built in Keras) - 83.3% accuracy

200 hidden units - 6 epochs | 37.62m build time - 81.67% build accuracy | 62s test time

#### 11/21 MNIST - LSTM (built in TensorFlow) - 98.55% accuracy

200 hidden units - 6 epochs | 49.5s build time

#### 02/14 Omniglot - MANN (LSTM Controller, 5-hot encoding) - ~50-70% 10th Accuracy

200 hidden units - 1.6m episodes | a long time

## Citations
Some code has been taken or adapted from hmishra2250's github under the MIT License. [[Github](https://github.com/hmishra2250/NTM-One-Shot-TF)]
The Memory Augmented Neural Network was taken and slightly modified from snowkylin's github under the LGPL-3.0 License [[Github](https://github.com/snowkylin/ntm)]

Memory Augmented Neural Network based on the following paper:
Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, Timothy Lillicrap, *One-shot Learning with Memory-Augmented Neural Networks*, [[arXiv](http://arxiv.org/abs/1605.06065)]
