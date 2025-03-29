# Cyberbullying Text Classification

## Project Overview
This project focuses on detecting cyberbullying in text data using machine learning and fine-tuned language models. It involves data preprocessing, model training, evaluation, and deployment.

## Features
- **Data Processing:** Cleans and preprocesses textual data for analysis.
- **Machine Learning Models:** Implements traditional ML models and fine-tunes LLMs for classification.
- **Evaluation Metrics:** Uses accuracy, precision, recall, and F1-score.
- **Deployment Ready:** Can be integrated into applications for real-time moderation.

## Dataset
The project uses the `cyberbullying_tweets.csv` dataset, which contains labeled instances of text indicating whether a tweet contains cyberbullying content.

## Requirements
To run this project, install the required dependencies:
```bash
pip install pandas numpy scikit-learn transformers torch
```

## Usage
1. **Preprocess Data:**
   - Load dataset and clean text.
   - Tokenization using NLP techniques.

2. **Train Model:**
   - Train machine learning models (e.g., Logistic Regression, SVM, LSTMs).
   - Fine-tune a large language model (LLM) using `Cybertext_bullying_classifier_using_LLM_Tuning.ipynb`.

3. **Evaluate Model:**
   - Assess performance using accuracy, precision, recall, and F1-score.

4. **Run Inference:**
   - Use trained models to classify new text samples.

## Notebooks
- `Cybertext_bullying_classifier.ipynb`: Implements traditional ML classifiers.
- `Cybertext_bullying_classifier_using_LLM_Tuning.ipynb`: Fine-tunes LLMs for improved classification.

## Future Improvements
- Enhance dataset with diverse cyberbullying examples.
- Optimize models for real-time processing.
- Deploy as a web API or chatbot.

## Contributing
Feel free to fork this repository, make improvements, and submit a pull request.

## License
This project is open-source under the MIT License.


