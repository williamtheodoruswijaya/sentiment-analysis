# Sentiment Analysis Model

**Final Project for Artificial Intelligence Course - Binus University**

This repository contains the code and resources for a sentiment analysis AI model developed for the final project of the Artificial Intelligence course at Binus University. The AI was built using Jupyter Notebook and leverages machine learning techniques to classify text into various sentiment categories.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)
- [References](#references)

## Project Overview

The aim of this project is to develop a sentiment analysis model that can classify text into different sentiment categories such as positive, negative, or neutral. The model is trained using a labeled dataset of text data and is implemented using Python in a Jupyter Notebook environment.

## Dataset

The dataset used for this project consists of [insert dataset information: size, source, categories]. It is preprocessed using tokenization, stop word removal, and other text-cleaning techniques. For feature extraction, techniques like word embeddings were used.

## Model Architecture

The sentiment analysis model was built using TensorFlow and contains the following layers:

- **Embedding Layer**: Transforms text input into dense vector representations.
- **LSTM/GRU Layer**: For capturing long-term dependencies in the text sequence.
- **Dense Layers**: For final classification into sentiment categories.

The model was optimized using 'Adam' optimizer and evaluated with the `categorical_crossentropy` loss function.

## Requirements

To run this project, the following dependencies are required:

- Python 3.x
- Jupyter Notebook
- TensorFlow
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

## Installation

1. Clone this repository:

```bash
git clone https://github.com/williamtheodoruswijaya/sentiment_analysis_project.git
```

2. Navigate to the project directory:

```bash
cd sentiment_analysis_project
```

3. Install the dependencies:

```bash
pip install tensorflow
pip install pandas
pip install nltk
pip install numpy
pip install matplotlib
pip install scikit-learn
```

## Usage

1. Open the Jupyter Notebook:

```bash
jupyter notebook
```

2. Run the notebook step by step to train the model and evaluate its performance.

3. To evaluate the model on your custom data, modify the notebook and run the `predict()` function with new text inputs.

## Results

The model achieved the following performance on the test dataset:

- Accuracy: 0.8924
- Loss: 0.3954

## Conclusion

This AI successfully classifies text into sentiment categories and demonstrates the application of machine learning techniques to natural language processing (NLP) tasks. It provides a solid foundation for further research and development in sentiment analysis.

## References

- TensorFlow Documentation: https://www.tensorflow.org/
- Scikit-learn Documentation: https://scikit-learn.org/
- Dataset Source: https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/data
