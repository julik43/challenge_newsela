from flask import Flask, request, jsonify
from predict_template import TopicPredictor

app = Flask(__name__)

# Load the predictor once
predictor = TopicPredictor()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    content_id = data.get('content_id')
    threshold = data.get('threshold', 0.5)

    if content_id is None:
        return jsonify({'error': 'content_id is required'}), 400

    topic_ids = predictor.predict(content_id, threshold)
    return jsonify({'topic_ids': topic_ids})

if __name__ == '__main__':
    app.run(debug=True)
