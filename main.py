from flask import Flask, request, jsonify
from model import RAG


app = Flask(__name__)





@app.route('/assist', methods=['POST'])
def assist():
    data = request.json
    a = RAG()
    query = "wadawdwad"
    res = get_hyeat(query, a.model, a.tokenizer)

    response = {
        "text": "Успешный ответ",
        "links": res #[link1, link2]
    }
    return jsonify(response), 200

if __name__ == '__main__':
    app.run(debug=False)