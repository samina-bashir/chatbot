# Chatbot using Neural Networks and Natural Language Processing

This repository contains code for a simple console based chatbot implemented using Neural Networks and Natural Language Processing techniques. The chatbot is designed to engage in conversations with users, provide positive and supportive responses, and offer helpful responses on various well-being topics.

## Requirements

- Python (version x.x)
- Libraries: `nltk`, `numpy`, `tensorflow`, `matplotlib`

## Usage

1. Install the required libraries:
   ```sh
   pip install nltk numpy tensorflow matplotlib
   ```

2. Run the code:
   ```sh
   python chatbot.py
   ```

3. Interact with the chatbot:
   ```
   You: Hi there!
   Bot: Hello! Ready to add some positivity to your day?
   You: Tell me a joke
   Bot: Why don't scientists trust atoms? Because they make up everything!
   ...
   ```

## Description

This chatbot uses a simple neural network model to predict the intent of user input and respond accordingly. The `intents.json` file contains predefined intents, patterns, and responses that the chatbot uses to generate responses. The conversation flow is structured to cover various topics related to well-being, positivity, stress management, creativity, and personal growth.

The code is divided into the following main parts:

1. **Data Preprocessing:** The provided `intents.json` file is loaded, and data is processed to prepare it for training.

2. **Model Training:** A neural network model is created using the TensorFlow library. The model is trained on the preprocessed data to predict the intent of user input.

3. **Conversation Loop:** The chatbot enters an interactive loop, accepting user input and using the trained model to predict intents. Appropriate responses are selected based on the predicted intents.
