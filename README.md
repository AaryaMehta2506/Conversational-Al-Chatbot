AI/ML Advance Project
# Conversational Al Chatbot

## Overview
This project implements a conversational AI chatbot using natural language processing and a simple neural network classifier. The chatbot recognizes user intents from a predefined intents file and returns one of several canned responses per intent. This is an intent-classification-based chatbot, not a large language model.

## Important note
The chatbot can only respond to messages that match one of the trained intents. It is not a large language model like ChatGPT; it is a rule-based ML classifier trained on your provided intent patterns and responses. To expand its capabilities, add more patterns and responses to intents.json and retrain.

## Repository structure
Conversational-AI-Chatbot/
├── app.py                 (Streamlit web application for chat interface)
├── train_chatbot.py       (Training script that builds and saves the model)
├── intents.json           (Intent definitions: patterns and responses)
├── chatbot_model.h5       (Saved trained model, created by train_chatbot.py)
├── words.pkl              (Saved token list, created by train_chatbot.py)
├── classes.pkl            (Saved label list, created by train_chatbot.py)
└── README.md              (this file)

## Requirements
Install required Python packages:
```bash
pip install tensorflow==2.20.0 streamlit nltk numpy scikit-learn
```

If you use a different Python version, choose a compatible TensorFlow release. You may also need to download NLTK data:
``` bash
python -m nltk.downloader punkt punkt_tab wordnet
```

Training the model
1. Ensure intents.json exists and contains the intents you want the bot to recognize.
2. Run the training script from the project directory:
python train_chatbot.py
3. After successful training, the script will save:
   - chatbot_model.h5
   - words.pkl
   - classes.pkl

Running the chatbot (Streamlit)
1. From the project directory (where app.py and the saved artifacts are located), run:
streamlit run app.py
2. The Streamlit app opens in the browser. Enter messages and click Send to get responses.

How it works (high level)
- The training script tokenizes and lemmatizes patterns from intents.json, builds a bag-of-words representation, and trains a small feedforward neural network to classify intents.
- The Streamlit app loads the trained model and helper files, converts user input into the same bag-of-words representation, predicts the most likely intent, and returns a randomly chosen response for that intent.
- Expand the bot by editing intents.json (add patterns and responses) and retrain.

Extending the bot
- Add more intents and varied patterns to intents.json for broader coverage.
- Increase training data per intent to improve classification accuracy.
- Replace the bag-of-words model with contextual embeddings (for example, using sentence-transformers) for better generalization.
- Integrate an external LLM for fallback or open-ended responses if desired.

## Contributing
Contributions are welcome!
Feel free to fork the repository, improve the game, and open a pull request. Let's grow this classic game together!

## License
This project is licensed under the [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

## Author
**Aarya Mehta**  
🔗 [GitHub Profile](https://github.com/AaryaMehta2506)


