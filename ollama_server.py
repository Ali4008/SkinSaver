from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    messages = data.get('messages', [])

    # Ensure we are passing all the messages to the ollama process
    conversation = ""
    for msg in messages:
        role = "User" if msg['role'] == 'user' else "Assistant"
        conversation += f"{role}: {msg['content']}\n"

    if not conversation:
        return jsonify({"error": "No conversation provided"}), 400

    # Full path to the ollama executable
    ollama_path = r'C:\Users\Hp\AppData\Local\Programs\Ollama\ollama.exe'

    # Set the OLLAMA_RUNNERS_DIR environment variable
    os.environ['OLLAMA_RUNNERS_DIR'] = r'C:\Users\Hp\AppData\Local\Programs\Ollama\ollama_runners'

    # Run Ollama command
    try:
        result = subprocess.run([ollama_path, 'run', 'phi3'], input=conversation, capture_output=True, text=True, shell=True)
    except FileNotFoundError as e:
        return jsonify({"error": f"Command not found: {str(e)}"}), 500

    if result.returncode != 0:
        error_message = result.stderr.strip()
        return jsonify({"error": f"Failed to get response from Ollama: {error_message}"}), 500

    response_content = result.stdout.strip()

    return jsonify({
        "choices": [
            {
                "message": {
                    "content": response_content
                }
            }
        ]
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
