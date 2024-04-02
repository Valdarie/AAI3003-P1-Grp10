# AAI3003-P1-10
Spam Detection using NLP

## Running on local host
1. Install all the required libraries using `pip install -r requirements.txt`
2. Type in `python -m streamlit run GUI/homepage.py`

## Streamlit
### Homepage Interface (`homepage.py`)
![image](https://github.com/Valdarie/AAI3003-P1-Grp10/assets/31137223/1f1f0aa1-ec0c-41ea-b763-aefa35356d44)

### Spam Detection Demo
- Type in a sentence to be used for determining whether or not it will be spam or ham
- Clear results button to clear results and reset session state
- Users are able to drag and drop or upload their own `.csv` files for entry with multiple input sentences
![image](https://github.com/Valdarie/AAI3003-P1-Grp10/assets/31137223/ac38f6ef-dd6c-42ab-9853-dd2c25cf58d9)

### Performance Evaluation
The best ML model for NLP will be highlighted in fluorescent green.


<p align="justify"> In our spam detection model, a `TF-IDF Vectorizer` is used to convert the preprocessed text data into numerical features. This vectorisation technique calculates the term frequency and inverse document frequency for each word in the dataset to represent the text data as a matrix of numerical values. This allows us to capture the importance of each word in relation to the document and the entire corpus. The `TF-IDF Vectorizer` not only helps in reducing the dimensionality of the text data but also enhances the model's ability to distinguish between spam and non-spam messages based on the significance of the words used. </p>

![image](https://github.com/Valdarie/AAI3003-P1-Grp10/assets/31137223/0197f367-8b75-483d-b4cb-3f3f24464236)
![image](https://github.com/Valdarie/AAI3003-P1-Grp10/assets/31137223/c8e398ed-77d7-43c6-b77e-991ace60dab8)

