# ISEF-JOCR
Improving Optical Character Recognition (OCR) of Japanese characters through context aware machine learning.

## Results
#### 11/11 MNIST - Maximum Likelihood Estimation - 11.35% accuracy

#### 11/21 MNIST - LSTM (built in Keras) - 83.3% accuracy

200 hidden units - 6 epochs | 37.62m build time - 81.67% build accuracy | 62s test time

#### 11/21 MNIST - LSTM (built in TensorFlow) - 98.55% accuracy

200 hidden units - 6 epochs | 49.5s build time

## Citations
Some code has been taken or adapted from hmishra2250's github under the MIT License. [[Github](https://github.com/hmishra2250/NTM-One-Shot-TF)]

Memory Augmented Neural Network based on the following paper:
Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, Timothy Lillicrap, *One-shot Learning with Memory-Augmented Neural Networks*, [[arXiv](http://arxiv.org/abs/1605.06065)]
