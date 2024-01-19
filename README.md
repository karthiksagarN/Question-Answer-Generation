Question Answer Generation with Streamlit:
Overview:
This project is designed to generate answers to user-provided questions using a combination of T5 transformer-based models and Sentence Embeddings. The workflow involves training a T5 model on a dataset containing questions and corresponding answers, saving the model using Pickle, and using Streamlit to create a user interface for interacting with the model.

Dependencies:
Make sure you have the following libraries installed:

pandas
pickle
warnings
re
sentence_transformers
transformers
torch
pytorch_lightning
sklearn
streamlit
You can install them using:
pip install pandas pickle warnings re sentence_transformers transformers torch pytorch_lightning scikit-learn streamlit

Data Preprocessing:
The dataset (Technical_interview.xlsx) is loaded using Pandas. The answers in different columns are preprocessed by removing special characters, converting to lowercase, and creating separate lists for each answer and user input.

Tokenization and Custom Dataset:
The T5 tokenizer is employed to tokenize the questions and contexts. A custom PyTorch dataset (CustomDataset) is created to manage the data and prepare it for training.

Data Splitting:
The dataset is split into training and testing sets using train_test_split from scikit-learn.

Lightning Data Module:
A PyTorch Lightning Data Module (DataModule) is defined to organize data loading for training and testing.

T5 Model Definition:
A custom PyTorch Lightning module (CustomModel) is created, wrapping the T5 model and defining training and validation steps.

Model Training:
The model is trained using PyTorch Lightning, and the best checkpoint is saved.

Streamlit Web App:
A Streamlit web application is developed for user interaction. Users can input questions and answers, and the model generates responses based on similarity measures.

Running the App:
Execute the following command to run the Streamlit app:
streamlit run your_script.py

Loading the Trained Model:
The trained T5 model is loaded using Pickle.

Answer Generation and Similarity Check:
A function (check_similarity) is defined to calculate the similarity between user-provided answers and correct answers using Sentence Embeddings. If the similarity exceeds a threshold, the user is given positive feedback; otherwise, an alternative answer is generated using the T5 model.

Streamlit UI:
The Streamlit app presents a simple UI with input fields for user questions and answers. Clicking the "Generate Result" button triggers the model to process the inputs and provide feedback.

Conclusion:
This README provides an overview of the project structure and functionality. Users can follow these instructions to set up and run the project on their local machines.
