from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')
import re
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from transformers import (AdamW)

app = Flask(__name__)

# Load the dataset and other necessary data
df = pd.read_excel("/Users/karthiksagar/DestinX/Technical_interview.xlsx")
df_copy = df.copy()

# Preprocess the data
df_copy['Answer1'] = df_copy['Answer1'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]', ' ', x))
df_copy['Answer1'] = df_copy['Answer1'].apply(lambda x: x.lower())
df_copy['Answer2'] = df_copy['Answer2'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]', ' ', x))
df_copy['Answer2'] = df_copy['Answer2'].apply(lambda x: x.lower())
df_copy['Answer3'] = df_copy['Answer3'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]', ' ', x))
df_copy['Answer3'] = df_copy['Answer3'].apply(lambda x: x.lower())
df_copy['Answer4'] = df_copy['Answer4'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]', ' ', x))
df_copy['Answer4'] = df_copy['Answer4'].apply(lambda x: x.lower())
df_copy['Answer5'] = df_copy['Answer5'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]', ' ', x))
df_copy['Answer5'] = df_copy['Answer5'].apply(lambda x: x.lower())
df_copy['Answer6'] = df_copy['Answer6'].str.replace('[^a-zA-Z0-9]', ' ')
df_copy['Answer6'] = df_copy['Answer6'].str.lower()
df_copy['user_answer'] = df_copy['user_answer'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]', ' ', x))
df_copy['user_answer'] = df_copy['user_answer'].apply(lambda x: x.lower())

corpus_ans1 = list(df_copy['Answer1'].values)
corpus_ans2 = list(df_copy['Answer2'].values)
corpus_ans3 = list(df_copy['Answer3'].values)
corpus_ans4 = list(df_copy['Answer4'].values)
corpus_ans5 = list(df_copy['Answer5'].values)
corpus_ans6 = list(df_copy['Answer6'].values)
corpus_user = list(df_copy['user_answer'].values)

corpus_questions = list(df_copy['Questions'].values)

tokenizer = T5Tokenizer.from_pretrained("t5-base-tokenizer")

MODEL_NAME = 't5-base'

class CustomModel(pl.LightningModule):

  def __init__(self):
    super().__init__()
    self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict = True)

  def forward(self, input_ids, attention_mask, labels = None, decoder_input_ids=None):
    output = self.model(
      input_ids = input_ids,
      attention_mask = attention_mask,
      labels = labels,
      decoder_input_ids = decoder_input_ids
    )
    return output.loss, output.logits
  def training_step(self, batch, batch_idx):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("train_loss", loss, prog_bar=True, logger=True)
    return loss
  def validation_step(self, batch, batch_idx):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("val_loss", loss, prog_bar=True, logger=True)
    return loss
  def test_step(self, batch, batch_idx):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("test_loss", loss, prog_bar=True, logger=True)
    return loss

  def configure_optimizers(self):
    return AdamW(self.parameters(), lr = 0.0001)

# Load the model
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Define the function to generate answers
def generate_answer(question):
  source_encoding = tokenizer(
      question['Questions'],
      question['CONTEXT'],
      max_length = 396,
      padding = 'max_length',
      truncation = 'only_second',
      return_attention_mask = True,
      add_special_tokens = True,
      return_tensors = 'pt'
  )
  pred_id = loaded_model.model.generate(
      input_ids = source_encoding['input_ids'],
      attention_mask = source_encoding['attention_mask'],
      num_beams = 1,
      max_length = 200,
      repetition_penalty = 2.5,
      length_penalty = 1.0,
      early_stopping = True
  )
  pred =[tokenizer.decode(i, skip_special_tokens=True, clean_up_tokenization_spaces=True) for i in pred_id]
  return (pred[0])

# Define the function to calculate similarity
def calculate_similarity(user_embedding, correct_answer_embedding):
    return util.pytorch_cos_sim(user_embedding, correct_answer_embedding)[0][0].item()

# Define the function to check similarity
def check_similarity(user_answer, correct_answers, index, similarity_threshold=0.98):
    # Load the model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    # model.save("/Users/karthiksagar/DestinX/paraphrase-MiniLM-L6-v2")

    # Ensure user_answer is a string
    user_answer = str(user_answer)

    # Encode the user and correct answers into sentence embeddings
    user_embedding = model.encode(user_answer, convert_to_tensor=True)
    correct_answers_embedding = [model.encode(str(answer), convert_to_tensor=True) for answer in correct_answers]

    # Calculate cosine similarity between the user and each correct answer
    similarity_scores = [calculate_similarity(user_embedding, ans_embedding) for ans_embedding in correct_answers_embedding]
    max_similarity_score = max(similarity_scores)
    # Check if the list is not empty before using max
    genanswer = ""
    finalist=[]

    if max_similarity_score > similarity_threshold:
        # Check if the maximum similarity score is above the threshold
        #print(max_similarity_score)
        finalist.append(max_similarity_score)
        genanswer= "Your answer is good enough...congrats!"
        finalist.append(genanswer)

        return finalist
    else:
        # If the list is empty, all scores are below the threshold
        #print("well tried, instead you can answer to this question in this way, \n")
        sample_question = df.iloc[index]
        genanswer =  generate_answer(sample_question)

        finalist.append(max_similarity_score)
        finalist.append(genanswer)

        #print(finalist)

        return finalist

def find_index(dataset_questions, target_name):
    try:
        index = dataset_questions.index(target_name)
        return index
    except ValueError:
        print(f"{target_name} not found in the list.")
        return None

# Define the function to run the model
def run_model(Question1, Answer1, Question2, Answer2):
    results = []
    input_dict = {Question1: Answer1, Question2: Answer2}
    for i in input_dict:
        index = find_index(corpus_questions, i)
        correct_answers = [corpus_ans1[index], corpus_ans2[index], corpus_ans3[index], corpus_ans4[index], corpus_ans5[index], corpus_ans6[index]]
        user_answer = input_dict[i]
        is_match = check_similarity(user_answer, correct_answers, index)
        dict1 = {"Question": i, "Similarity": is_match[0], "Feedback": is_match[1]}
        results.append(dict1)
    return results

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle form submission
@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        # Get the form data
        Question1 = request.form['Question1']
        Answer1 = request.form['Answer1']
        Question2 = request.form['Question2']
        Answer2 = request.form['Answer2']
        
        # Run the model
        results = run_model(Question1, Answer1, Question2, Answer2)
        
        # Return the results as JSON
        return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
