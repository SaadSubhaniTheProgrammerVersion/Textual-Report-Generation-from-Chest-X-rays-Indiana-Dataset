import streamlit as st
import os
import numpy as np
import time
from PIL import Image
import create_model as cm
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import re
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoModelForTokenClassification
from torch.utils.data import Dataset
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import nltk
from rouge import Rouge

def calculate_rouge_scores(generated_report, actual_report):
    rouge = Rouge()
    scores = rouge.get_scores(generated_report, actual_report)
    return scores

st.title("THIS IS THE TESTING SITE FOR X-rAI")
st.markdown("[Github](https://github.com/SaadSubhaniTheProgrammerVersion)")
st.markdown("\nThis app will generate impression part of an X-ray report.\nYou can upload 2 X-rays that are front view and side view of chest of the same individual.")
st.markdown("The 2nd X-ray is optional.")


col1,col2 = st.columns(2)
image_1 = col1.file_uploader("X-ray 1",type=['png','jpg','jpeg'])
image_2 = None
if image_1:
    image_2 = col2.file_uploader("X-ray 2 (optional)",type=['png','jpg','jpeg'])

col1,col2 = st.columns(2)
predict_button = col1.button('Predict on uploaded files')
test_data = col2.button('Predict on sample data')

@st.cache(allow_output_mutation=True)
def create_model():
    model_tokenizer = cm.create_model()
    return model_tokenizer

def get_report_by_uid(uid):
    # Read the CSV file into a DataFrame
    csv_file_path = "../indiana_reports.csv"  # Adjust this path as necessary
    df = pd.read_csv(csv_file_path)
    
    # Convert uid to int to match the DataFrame's format if necessary
    uid_int = int(uid)
    
    # Find the row with the matching uid
    report_row = df[df['uid'] == uid_int]
    
    # If a matching report is found, extract the findings
    if not report_row.empty:
        findings = report_row.iloc[0]['findings']
        print(f"Findings for UID {uid}: {findings}")
        return findings
    else:
        print(f"No findings found for UID {uid}")
        return None

# Example usage


def predict(image_1,image_2,model_tokenizer,predict_button = predict_button):
    start = time.process_time()
    if predict_button:
        
        if (image_1 is not None):
            # print(f"\n\nUploaded file 1: {image_1.name}")
            match = re.match(r"(\d+)_", image_1.name)
            if match:
                uid = match.group(1)
                print(f"Extracted UID: {uid}")  # For verification
                
                csv_file_path = "../indiana_reports.csv"  # Adjust this path as necessary
                report = get_report_by_uid(uid)
            else:
                print("No UID found in the filename")
                uid = None  # Or handle the case as needed
            start = time.process_time()  
            image_1 = Image.open(image_1).convert("RGB") #converting to 3 channels
            image_1 = np.array(image_1)/255
            if image_2 is None:
                image_2 = image_1
            else:
                # print(f"\n\nUploaded file 2: {image_2.name}")
                image_2 = Image.open(image_2).convert("RGB") #converting to 3 channels
                image_2 = np.array(image_2)/255
            st.image([image_1,image_2],width=300)
            caption = cm.function1([image_1],[image_2],model_tokenizer)

                        # Load the language model and tokenizer
            model_directory = "/FYP_DATASET/results/"

            loaded_model = AutoModelForCausalLM.from_pretrained(model_directory)
            tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")

            if torch.cuda.is_available():
                device = torch.device("cuda")
                print("Using GPU:", torch.cuda.get_device_name(0))
            else:
                device = torch.device("cpu")
                print("GPU is not available, using CPU instead. Inference might be slow.")

            # Move the model to the GPU
            loaded_model = loaded_model.to(device)

            # Get user input
            user_input = caption[0]
            # user_input=user_input.replace(';', ' and ')

            input_ids = tokenizer.encode(user_input, return_tensors="pt")

            # Move the input_ids to the same device as the model
            input_ids = input_ids.to(device)


            # Generate output from the language model
            output = loaded_model.generate(
                input_ids,
                max_length=200,  # Adjust the maximum length here
                num_return_sequences=1,  # Increase if you need multiple sequences
                pad_token_id=tokenizer.eos_token_id,  # Padding token ID
                do_sample=True,  # Enable sampling
                temperature=0.7,  # Control randomness of sampling
                top_k=50,  # Filter top-k tokens to sample from
                top_p=0.95,  # Filter cumulative probability for top-p sampling
                repetition_penalty=1.0,  # Adjusts for repeating tokens
            )
            # Decode the output tokens to text
            decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
            print("User input:", user_input)
            # Display the generated text
            print("Generated output:", decoded_output)
            caption= decoded_output
            print("Actual Report:",report)
            rouge_scores = calculate_rouge_scores(caption, report)
            print("ROUGE scores:", rouge_scores)
            generated_report = decoded_output.split()  # Tokenize the generated report
            actual_report = [report.split()]  # Tokenize the actual report and wrap it in another list

            bleu_score = sentence_bleu(actual_report, generated_report)
            print(f"BLEU score for this report: {bleu_score}")
            

            # Calculate METEOR score
            # Note: METEOR expects the actual and predicted to be strings, not tokenized.
            # meteorScore = meteor_score([" ".join(actual_report[0])], " ".join(generated_report))
            # print(f"METEOR score for this report: {meteorScore}")



            st.markdown(" ### **Predicted Report:**")
            impression = st.empty()
            impression.write(caption)
            time_taken = "Time Taken for prediction: %i seconds"%(time.process_time()-start)
            st.write(time_taken)
            st.markdown(" ### **Actual Report:**")
            st.write(report)
            del image_1,image_2
            # st.markdown(" ### **BLEU Score for this report:**")
            # st.write(bleu_score)
            st.markdown(" ### **ROUGE Scores for this report:**")
            st.write(rouge_scores)
        else:
            st.markdown("## Upload an Image")


def predict_sample(model_tokenizer,folder = './test_images'):
    no_files = len(os.listdir(folder))
    file = np.random.randint(1,no_files)
    file_path = os.path.join(folder,str(file))
    if len(os.listdir(file_path))==2:
        image_1 = os.path.join(file_path,os.listdir(file_path)[0])
        image_2 = os.path.join(file_path,os.listdir(file_path)[1])
        print(file_path)
    else:
        image_1 = os.path.join(file_path,os.listdir(file_path)[0])
        image_2 = image_1
    predict(image_1,image_2,model_tokenizer,True)
    

model_tokenizer = create_model()


if test_data:
    predict_sample(model_tokenizer)
else:
    predict(image_1,image_2,model_tokenizer)
