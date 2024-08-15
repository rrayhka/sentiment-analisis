# Sentiment-Analysis-App

This repository contains a web-based sentiment analysis application that allows users to input text and choose between two different models for analysis: BiLSTM and Attention. The application provides a simple and interactive interface for sentiment analysis, enabling users to quickly determine the sentiment of a given sentence.

## Table of Contents

- [About](#about)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Contact](#contact)

## About

The Sentiment-Analysis-App is designed to analyze the sentiment of text data using two state-of-the-art models: BiLSTM (Bidirectional Long Short-Term Memory) and Attention. The user can input a sentence through a web interface and select which model to use for the sentiment analysis.

## Features

- **Two Models:** Choose between BiLSTM and Attention models for sentiment analysis.
- **User-Friendly Web Interface:** Simple input form for text and model selection.
- **Real-Time Sentiment Analysis:** Displays the sentiment result immediately after submission.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/rrayhka/sentiment-analisis-app.git
    cd sentiment-analisis-app
    ```

2. Run the web application:

    ```bash
    python app.py
    ```

## Usage

1. **Accessing the Web Interface:**
   - After running the application, open your web browser and navigate to `http://localhost:5000`.
   
2. **Performing Sentiment Analysis:**
   - Enter the text you wish to analyze in the input field.
   - Select either the BiLSTM or Attention model from the dropdown menu.
   - Click "Analyze Sentiment" to receive the sentiment result.

3. **Viewing Results:**
   - The sentiment result will be displayed on the page, indicating whether the input text is positive, negative, or neutral.

## Project Structure

- `app.py`: The main Flask application file that runs the web interface.
- `models/`: Directory containing pre-trained models (BiLSTM and Attention).
 - `nn.py`: Contains the neural network models and the logic for loading and predicting sentiment using BiLSTM and Attention models.
- `notebooks/`: Contains Jupyter notebooks used for model training and evaluation.
- `templates/`: HTML templates for the web interface.
- `dataset/`: Contains the dataset used for training and testing.

## Contributing

Contributions are welcome! If you have any suggestions, improvements, or bug fixes, feel free to submit a pull request or open an issue.

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.


## Contact

Akhyar - [khyar075@gmail.com](mailto:khyar075@gmail.com)
