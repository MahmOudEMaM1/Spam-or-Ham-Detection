from flask import Flask, request, render_template, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

# Load model
model_save_path = "bert_spam_classifier.pt"
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
model.eval()

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define route for index page
@app.route('/')
def index():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    # Tokenize message
    inputs = tokenizer.encode_plus(
        message,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()
    if predicted_label == 1:
        prediction = "spam"
    else:
        prediction = "ham"
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
