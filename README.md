# Cost_benefit_of_LLMs
Cost benefit of using LLMs for text classification instead of models such as BERT

# 1. Classification of BERT embeddings using a Neural Network

Previously we have used BERT to create embeddings of e-commerce products descriptions to see if it was possible to use the embeddings for classification.

A few distinct features were visible after doing a T-SNE to represent the embeddings in 2D.

In order to obtain a label from this embeddings, we are using a simple deep neural network model to classify them.

On the validation data, we obtain a 0.93 accuracy

# 2. Classification using a LLM (Mistral7B)

Recently LLMs have been used for a lot of usage, and some suggest it would be simpler to replace traditional text classification approaches by LLMs to reduce the development and maintenance time.

To quickly build a classifier from a LLM, we will use Mistral7B with Ollama.

# Results

The Mistral based model took much longer time to predict the label. In addition the accuracy was lower than the Neural Network used on BERT embeddings.

In both case, the training and model building requires very low time. However BERT was cheaper and faster to run. Therefore, it is still a better option than using a LLM for classifying the e-commerce items.
