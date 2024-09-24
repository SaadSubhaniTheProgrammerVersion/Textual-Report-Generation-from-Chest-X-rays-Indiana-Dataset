import json
from flask import Flask, jsonify, request
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
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins='http://localhost:3000')
app.config['CORS_HEADERS'] = 'application/json'

def create_model():
    model_tokenizer = cm.create_model()
    return model_tokenizer

@app.route("/predict", methods=["POST"])
def predict():
    print("called")
    # print(request.files.keys)
    print(len(request.files))
    image1 = request.files['image1']
    if len(request.files) > 1:
        image2 = request.files['image2']
    else: 
        image2 = image1
    # print(image2)
    print(image1)
    if image1 == None:
        return "No file was uploaded", 400
    if image1 != None: 
        start = time.process_time()  
        image_1 = Image.open(image1).convert("RGB") #converting to 3 channels
        image_1 = np.array(image_1)/255
        image_2 = Image.open(image2).convert("RGB") #converting to 3 channels
        image_2 = np.array(image_2)/255
            # st.image([image_1,image_2],width=300)
        caption = cm.function1([image_1],[image_2],model_tokenizer)
# Load the language model and tokenizer
        model_directory = "C:/FYP_DATASET/results"

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
        decoded_output = json.dumps(decoded_output)
            # print("Generated output:", decoded_output)
        caption= decoded_output
        print(caption)


            # st.markdown(" ### *Generated Output:*")
            # impression = st.empty()
            # impression.write(caption)
        time_taken = "Time Taken for prediction: %i seconds"%(time.process_time()-start)
            
        response_data = {
                "caption":caption,
                "time": time_taken
            }

        return jsonify(response_data), 200
            # st.write(time_taken)
        # else:
        #     st.markdown("## Upload an Image")
model_tokenizer = create_model()
app.run(port=3500,debug=True)