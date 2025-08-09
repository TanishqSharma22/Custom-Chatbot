from flask import Flask, request, jsonify
from flask_cors import CORS
from custom_chatbot import chatbot_bow
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Configure CORS properly for production
CORS(app, origins=['http://localhost:3000', 'http://localhost:5000'])

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    """Chatbot API endpoint with comprehensive error handling."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        if 'question' not in data:
            return jsonify({'error': 'Missing question parameter'}), 400
        
        question = str(data['question']).strip()
        if not question:
            return jsonify({'error': 'Question cannot be empty'}), 400
            
        if len(question) > 1000:  # Basic input validation
            return jsonify({'error': 'Question too long (max 1000 characters)'}), 400
            
        response = chatbot_bow(question)
        logger.info(f"Processed question: {question[:50]}...")
        return jsonify({'response': response})
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# Remove insecure dataset endpoint in production
# @app.route('/dataset/<path:path>')
# def serve_dataset(path):
#     return send_from_directory('Dataset', path)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'chatbot-api'})

if __name__ == '__main__':
    # Use environment variable for debug mode
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.getenv('PORT', 5000))
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
