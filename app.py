from flask import Flask, request, render_template
from nn import bilstm_model, attention_model, predict_sentiment

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])

def predict():
    text = request.form['text']
    model_choice = request.form['model']
    style_encode = ""
    if model_choice == 'bilstm':
        model = bilstm_model
    else:
        model = attention_model
    
    sentiment = predict_sentiment(text, model)
    
    if sentiment == "negative":
        style_encode = "color: red;"
    elif sentiment == "positive":
        style_encode = "color: green;"
    else:
        style_encode = "color: blue;"
    
    return render_template('index.html', sentiment=sentiment, style_encode=style_encode, text=text)

if __name__ == '__main__':
    app.run(debug=True, port=3000)
