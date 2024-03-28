import streamlit as st
import pickle
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from st_aggrid import AgGrid, GridUpdateMode, JsCode
import altair as alt

# Download necessary NLTK data only if not already downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize PorterStemmer
ps = PorterStemmer()

# Define stopwords set outside the function for efficiency
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [ps.stem(word) for word in text if word.isalnum() and word not in stop_words]
    return ' '.join(text)

# Load the TfidfVectorizer and BERT model only once
with open('GUI/extensive_training/models/tfid_vectorizer.pkl', 'rb') as file:
    tfid = pickle.load(file)

bert_tokenizer = BertTokenizer.from_pretrained('GUI/bert_model_files/')
bert_model = BertForSequenceClassification.from_pretrained('GUI/bert_model_files/')

def bert_predict(sentence):
    inputs = bert_tokenizer.encode_plus(
        sentence, 
        return_tensors='pt', 
        max_length=64,
        padding='max_length', 
        truncation=True
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()

    return 'Likely a Spam' if prediction == 1 else 'Likely Not a Spam'

model_paths = {
    'LogisticRegression': 'GUI/extensive_training/models/LR_model.pkl',
    'SupportVectorMachine': 'GUI/extensive_training/models/SVC_model.pkl',
    'MultinomialNB': 'GUI/extensive_training/models/NB_model.pkl',
    'DecisionTreeClassifier': 'GUI/extensive_training/models/DT_model.pkl',
    'AdaBoostClassifier': 'GUI/extensive_training/models/Adaboost_model.pkl',
    'BaggingClassifier': 'GUI/extensive_training/models/Bgc_model.pkl',
    'ExtraTreesClassifier': 'GUI/extensive_training/models/ETC_model.pkl',
    'GradientBoostingClassifier': 'GUI/extensive_training/models/GBDT_model.pkl',
    'XGBClassifier': 'GUI/extensive_training/models/xgb_model.pkl',
    'RandomForestClassifier': 'GUI/extensive_training/models/RF_model.pkl',
    'BERT': 'bert'
}

# Load other models only once
loaded_models = {name: pickle.load(open(path, 'rb')) for name, path in model_paths.items() if name != 'BERT'}

def predict_spam(sentence):
    preprocessed_sentence = transform_text(sentence)
    numerical_features = tfid.transform([preprocessed_sentence]).toarray()
    results = []
    for model_name, model_path in model_paths.items():
        if model_name == 'BERT':
            prediction_text = bert_predict(sentence)
        else:
            model = loaded_models[model_name]
            prediction = model.predict(numerical_features)
            prediction_text = 'Likely a Spam' if prediction == 1 else 'Likely Not a Spam'
        results.append({'Model': model_name, 'Prediction': prediction_text})
    return results

def plot_chart(data):
    if 'Model' in data.columns:
        spam_proportions = data.groupby('Model')['Prediction'].apply(lambda x: (x == 'Likely a Spam').mean()).reset_index(name='Spam Proportion')
        
        chart = alt.Chart(spam_proportions).mark_bar().encode(
            x='Model',
            y='Spam Proportion',
            color='Model',
            tooltip=['Model', 'Spam Proportion']
        ).properties(
            width=600,
            height=400
        )
        
        st.altair_chart(chart, use_container_width=True)
    else:
        st.write('No model predictions available to plot.')

# Streamlit app
st.title('Spam Detection Demo')
st.markdown('Enter a sentence or upload a CSV file with sentences for spam detection.')

sentence = st.text_input('Enter a sentence:')
uploaded_file = st.file_uploader('Or upload a CSV file with sentences:', type=['csv'])

if uploaded_file is not None:
    st.session_state['uploaded_file'] = uploaded_file

if st.button('Predict') or 'results_df' in st.session_state:
    uploaded_file = st.session_state.get('uploaded_file')
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'Sentence' in df.columns:
                st.write('### Prediction Results:')
                sentences = df['Sentence'].tolist()
            else:
                st.error('CSV file must have a column named "Sentence"')
                sentences = []
        except Exception as e:
            st.error(f'Error reading CSV file: {e}')
            sentences = []
    else:
        sentences = [sentence] if sentence else []

    if sentences:
        all_results = []
        for sent in sentences:
            results = predict_spam(sent)
            for result in results:
                result['Sentence'] = sent
            all_results.extend(results)

        results_df = pd.DataFrame(all_results)
        st.session_state['results_df'] = results_df

        cell_style_jscode = JsCode("""
        function(params) {
            if (params.value === 'Likely a Spam') {
                return {'color': 'white', 'backgroundColor': 'darkred'};
            } else {
                return {'color': 'white', 'backgroundColor': 'darkgreen'};
            }
        };
        """)

        grid_options = {
            'columnDefs': [
                {'field': 'Sentence', 'filter': 'agTextColumnFilter', 'sortable': True, 'resizable': True},
                {'field': 'Model', 'filter': 'agTextColumnFilter', 'sortable': True, 'resizable': True},
                {'field': 'Prediction', 'cellStyle': cell_style_jscode, 'filter': 'agTextColumnFilter', 'sortable': True, 'resizable': True}
            ],
            'defaultColDef': {
                'editable': False,
                'filter': True,
                'sortable': True,
                'resizable': True
            }
        }

        plot_chart(results_df)

        grid_response = AgGrid(
            results_df, 
            gridOptions=grid_options, 
            update_mode=GridUpdateMode.VALUE_CHANGED, 
            fit_columns_on_grid_load=True, 
            theme='streamlit', 
            allow_unsafe_jscode=True,
            data_return_mode='FILTERED',
            key='predictions_grid'
        )

        
    else:
        st.write('Please enter a sentence or upload a CSV file for prediction.')

if st.button('Clear Results'):
    if 'results_df' in st.session_state:
        del st.session_state['results_df']
    if 'uploaded_file' in st.session_state:
        del st.session_state['uploaded_file']
    st.rerun()
