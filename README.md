# Sentiment Analysis on Hotel Reviews

This project is focused on performing **Sentiment Analysis** on a dataset of hotel reviews using various **Machine Learning** models. Sentiment analysis helps in determining the overall sentiment (positive, negative, or neutral) expressed in the hotel reviews. The goal is to build a model that accurately classifies the sentiment of user reviews, helping businesses gain insights into customer satisfaction.

## Features
- **Text Preprocessing:** Clean and prepare the text data by removing stop words, special characters, and performing tokenization.
- **Vectorization:** Use techniques like TF-IDF to convert text data into numerical features.
- **Modeling:** Implement different machine learning models such as:
  - Logistic Regression
  - Naive Bayes
  - Support Vector Machine (SVM)
  - Random Forest
- **Evaluation:** Compare the models based on accuracy, precision, recall, F1-score, and ROC-AUC curve.

## Dataset
The dataset contains thousands of hotel reviews, each labeled with a sentiment category. Sentiments are classified into three categories:
- **Positive**
- **Neutral**
- **Negative**

## Technologies and Tools
- **Python**
- **Scikit-learn**
- **Pandas**
- **Numpy**
- **Matplotlib/Seaborn** (for visualization)

## Results
The model with the highest accuracy was **SVM**, achieving an accuracy of **XX%** on the test set. Detailed performance metrics for each model are provided in the results section.

## Future Work
- Implement **Deep Learning** models like LSTM and CNN to improve accuracy.
- Perform **Aspect-based Sentiment Analysis** to capture sentiment on specific aspects like service, cleanliness, and location.
- Explore **BERT embeddings** for more sophisticated text representation.

## Contributing
Feel free to open issues or submit pull requests to improve the project!
