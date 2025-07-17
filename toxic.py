from flask import Flask, request, jsonify

from transformers import pipeline



app = Flask(__name__)
clf = pipeline(
    task = 'sentiment-analysis', 
    model = 'SkolkovoInstitute/russian_toxicity_classifier'
)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompts = data.get('prompts', [])
    outputs = list(clf(prompts))
    result = {
            'prompt': prompts,
            'generated_embeddings': None
        }
    if outputs:
        result = {
            'prompt': prompts,
            'generated_embeddings': outputs[0]['label']
        }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)