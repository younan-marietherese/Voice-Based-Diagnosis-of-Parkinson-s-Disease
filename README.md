# Voice-Based Diagnosis of Parkinson’s Disease

This project focuses on detecting Parkinson’s Disease using voice data. Parkinson’s Disease can cause changes in a person’s voice, and by analyzing these changes, we can develop models that assist in early diagnosis. The project involves feature extraction from voice recordings, building machine learning models, and evaluating their effectiveness in detecting Parkinson’s Disease.

---

## Project Overview
Parkinson’s Disease is a neurodegenerative disorder that affects motor functions, and changes in voice are among the symptoms. In this project, we use voice-based features such as frequency, amplitude, and jitter to build a machine learning model capable of diagnosing Parkinson’s Disease.

The project includes:
1. Preprocessing voice data to extract relevant features.
2. Training a machine learning model to classify whether a subject has Parkinson’s Disease based on their voice features.
3. Evaluating the model’s performance using standard metrics.

---

## Objectives
- Load and preprocess the voice dataset.
- Extract key voice features such as jitter, shimmer, harmonic-to-noise ratio (HNR), etc.
- Train machine learning models (e.g., Support Vector Machines, Random Forests) for diagnosis.
- Evaluate the models using accuracy, precision, recall, and F1-score.

---

## Technologies Used
- **Python**: For implementing the analysis and models.
- **Pandas and NumPy**: For data manipulation and analysis.
- **Scikit-learn**: For building and evaluating machine learning models.
- **Librosa**: For audio feature extraction (if used).
- **Matplotlib/Seaborn**: For data visualization.

---

## Dataset
The dataset used in this project consists of voice recordings of individuals, some of whom are diagnosed with Parkinson’s Disease. Each recording is associated with a set of extracted voice features such as:
- **MDVP (Jitter, Shimmer)**: Measures of variation in voice frequency and amplitude.
- **HNR (Harmonic-to-Noise Ratio)**: A measure of the noise levels in the voice.
- **Fundamental Frequency (F0)**: The basic pitch of the voice.

The features are extracted from the raw audio signals and used to train a classification model to predict the presence of Parkinson’s Disease.

---

## Key Steps

1. **Data Preprocessing**:
   - Load the dataset into a DataFrame.
   - Clean the data, handle missing values, and scale the features for model input.

2. **Feature Extraction**:
   - Extract voice features relevant to diagnosing Parkinson’s Disease, such as jitter, shimmer, and harmonic-to-noise ratio (HNR).

3. **Modeling**:
   - Train machine learning models like Support Vector Machines (SVM), Random Forests, or Neural Networks to classify the voice data as indicating Parkinson’s Disease or not.
   
4. **Evaluation**:
   - Evaluate the performance of the model using metrics such as:
     - **Accuracy**
     - **Precision**
     - **Recall**
     - **F1-score**
   - Visualize the results and analyze the model’s diagnostic accuracy.

---

## How to Use

### Prerequisites
Ensure you have the following libraries installed:
```bash
pip install pandas numpy scikit-learn librosa matplotlib seaborn
