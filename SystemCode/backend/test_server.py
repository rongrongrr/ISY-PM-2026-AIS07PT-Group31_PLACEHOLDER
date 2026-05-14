#!/usr/bin/env python3
"""
Simple test server to simulate the emotion analysis API for testing
"""
from flask import Flask, request, jsonify
import os
import numpy as np

app = Flask(__name__)

@app.route('/api/analyze-emotion', methods=['POST'])
def analyze_emotion():
    """Simulate emotion analysis API"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Simulate processing time
    import time
    time.sleep(1)

    # Return mock emotion results
    emotions = {
        'HAP': 0.85,  # Happy
        'SAD': 0.05,
        'ANG': 0.03,
        'NEU': 0.04,
        'FEA': 0.02,
        'DIS': 0.01
    }

    return jsonify({
        'emotions': emotions,
        'top_emotion': 'HAP',
        'confidence': 0.85
    })

@app.route('/')
def root():
    return jsonify({'message': 'Test API Server'})

if __name__ == '__main__':
    print("Starting test API server on http://localhost:8000")
    app.run(host='0.0.0.0', port=8000, debug=True)