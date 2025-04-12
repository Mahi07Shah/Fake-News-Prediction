# Fake News Detection System

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project aims to predict whether a news article is fake or real. A Logistic Regression model is trained and saved for predicting and deployed as a web application using Flask. By leveraging machine learning, this project provides insights that can help users quickly identify potentially fake news.

## Features

- **Predictive Modeling**: Utilizes a Logistic Regression model to predict whether a news article is fake.
- **User-Friendly Interface**: A web application built with Flask for easy interaction.
- **Data Preprocessing**: Handles stop word removal and stemming to prepare data for modeling.
- **Real-Time Predictions**: Users can input news article text and receive immediate predictions on its authenticity.

## Technologies Used

- **Python**: The primary programming language used for data analysis and model building.
- **Pandas**: A powerful data manipulation and analysis library.
- **Numpy**: A library for numerical computations and handling arrays.
- **Scikit-learn**: A machine learning library for model training, evaluation, and preprocessing.
- **Flask**: A micro web framework for building the interactive web application.
- **NLTK**: Natural Language Toolkit for text preprocessing (stop word removal, stemming).
- **Joblib**: A library for saving and loading Python objects, such as trained models.

## Dataset

The dataset used in this project is the Kaggle Fake News dataset, which contains information about news articles and their labels (real or fake). The dataset includes features such as:

- Article ID
- Title
- Author
- Text
- Label (target variable)

The dataset can be found on [Kaggle](https://www.kaggle.com/c/fake-news/data).

## Installation

To set up the project, follow these steps:

1.  Clone the repository:

git clone [repository_url]
cd [project_directory]

text

2.  Install the dependencies:

pip install -r requirements.txt

text

## Usage

1.  **Train the Model**:

*   Run the script to train the model.

2.  **Run the Flask Application**:

*   After training the model, start the Flask app to make predictions:

python app.py

text

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
