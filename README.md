Pokémon Image Classifier
========================

This project aims to classify Pokémon images using a variety of machine learning (ML) and deep learning (DL) models. The models utilize image preprocessing techniques to improve classification accuracy and computational efficiency. The project is implemented in Python using a Jupyter Notebook, and it explores various models including Convolutional Neural Networks (CNNs), Dense Neural Networks (DNNs), and traditional machine learning classifiers.

Project Overview
----------------

-   **Dataset**: 7,000 hand-cropped images of Pokémon characters (sourced from Kaggle).
-   **Image Preprocessing Techniques**: Blurring, sharpening, edge detection, and high-pass filtering.
-   **Models Used**:
    -   Convolutional Neural Networks (CNNs)
    -   Dense Neural Networks (DNNs)
    -   Random Forest
    -   Gradient Boosting
    -   Support Vector Machine (SVM)
    -   Logistic Regression

Methodology
-----------

The project includes:

1.  **Image Preprocessing**: Techniques like Gaussian blurring, unsharp masking, Sobel and Prewitt operators for edge detection, and high-pass filtering were applied to enhance image features for classification.
2.  **Model Training**: Various models were trained on the preprocessed images to evaluate their accuracy and performance.
3.  **Evaluation**: The models were evaluated based on classification accuracy and training time.

How to Run
----------

To run this project, follow these steps:

### Steps to Run

1.  **Clone the Repository**:

    `git clone https://github.com/your-username/Pokemon-image-classifier.git
    cd Pokemon-image-classifier`
    
2.  **Open the Notebook**: Open the provided `.ipynb` file (the Jupyter Notebook) and run the cells in sequence to preprocess the images, train the models, and evaluate their performance.

Image Preprocessing Techniques
------------------------------

The project applies the following preprocessing techniques:

-   **Blurring**: Using Gaussian filters to reduce noise.
-   **Sharpening**: Enhancing the edges in the images.
-   **Edge Detection**: Using Sobel and Prewitt operators to detect edges.
-   **High-Pass Filtering**: Retains high-frequency components for edge emphasis.

Model Performance
-----------------

The project compares different ML and DL models to find the best-performing model for Pokémon image classification. Key metrics include:

-   **Accuracy**: The percentage of correct predictions.
-   **Training Time**: The time taken to train each model.

### Best Model

-   **CNN**: Achieved the highest accuracy of 92.3% with Prewitt Vertical preprocessing.

Results
-------

The CNN models generally outperformed traditional ML models for image classification tasks. The preprocessing techniques, especially edge detection and high-pass filtering, helped improve classification accuracy for certain models.

Conclusion
----------

This project demonstrates the effectiveness of convolutional neural networks (CNNs) for Pokémon image classification, particularly when combined with targeted image preprocessing techniques.

**For more information, please read the pdf file in this repository. It contains a more in-depth analysis of this project, and will provide a lot more context for the same.**
