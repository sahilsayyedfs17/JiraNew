from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import re
from collections import defaultdict
from transformers import pipeline, TapasTokenizer, TapasForQuestionAnswering, AutoTokenizer, AutoModelForSeq2SeqLM
import os
import json
import difflib
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize

# Ensure nltk resources are available
nltk.data.path.append('./nltk_data')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the Tapas model for table-question-answering
model_path = "./models/tapas"
model = TapasForQuestionAnswering.from_pretrained(model_path)
tokenizer = TapasTokenizer.from_pretrained(model_path)
tqa = pipeline("table-question-answering", model=model, tokenizer=tokenizer)

# Load CSV into a DataFrame
df = pd.read_csv('last Month VTVAS.csv')

# Load the SentenceTransformer model
sentence_model_path = "models/sentence-transformer"
sentence_model = SentenceTransformer(sentence_model_path)

# Load the summarization model
summarizer_model_path = "models/summarisation"
summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_model_path)
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_model_path)
summarizer = pipeline("summarization", model=summarizer_model, tokenizer=summarizer_tokenizer)

# Dictionary to store question-answer pairs
qa_pairs = {}

# Synonyms dictionary for column names
column_synonyms = {
    "priority": ["importance", "urgency", "Priority"],
    "summary": ["title", "heading", "Summary"],
    "description": ["details", "info", "Description"],
    "issue_type": ["type", "category", "Issue_type", "Issue Type", "issue type", "Issue type"],
    "assignee": ["assigned_to", "responsible", "assigned to", "assignee", "asign"],
    "created_date": ["date_created", "creation_date", "it created"],
    "labels": ["tags", "label"],
    "issue_key": ["issue key", "issue", "Issue Key", "Issue key", "ticket no", "Ticket no", "Ticket", "Ticked Id", "ticket Id", "ticket id"],
    "reporter": ["Reporter", "reported", "has raised"],
    "status": ["Status", "state"]
}

# Functions

def map_synonym_to_column(word):
    word_lower = word.lower()
    for column, synonyms in column_synonyms.items():
        if word_lower == column or word_lower in map(str.lower, synonyms):
            return column
    return None

def ask_question(query):
    condition_pattern = re.compile(r'(\w+)\s+as\s+(\w+)')
    keyword_pattern = re.compile(r'related\s+to\s+(\w+)')

    conditions = defaultdict(list)
    keyword = None

    for match in condition_pattern.finditer(query.lower()):
        column = match.group(1)
        value = match.group(2)
        mapped_column = map_synonym_to_column(column)
        if mapped_column:
            conditions[mapped_column].append(value)

    for match in keyword_pattern.finditer(query.lower()):
        keyword = match.group(1)

    filtered_df = df.copy()
    for column, values in conditions.items():
        if (column in filtered_df.columns):
            filtered_df = filtered_df[filtered_df[column].str.lower().isin(values)]

    if (keyword):
        filtered_df = filtered_df[filtered_df.apply(lambda row: keyword in row.astype(str).str.lower().values, axis=1)]

    return filtered_df.to_dict(orient='records')

def learn_question_answer_pairs_from_file(file_path):
    with open(file_path, 'r') as f:
        qa_data = json.load(f)
        for item in qa_data['question_answers']:
            question = item['question']
            answers = item['answers']
            learn_question_answer_pair(question, answers)

def learn_question_answer_pair(question, answers):
    qa_pairs[question.lower()] = answers

def calculate_similarity(query, text):
    matcher = difflib.SequenceMatcher(None, query.lower(), text.lower())
    return round(matcher.ratio() * 100, 2)

def get_response(question):
    if (question.lower() in qa_pairs):
        return qa_pairs[question.lower()]
    else:
        response = ask_question(question)
        return response

def handle_query(query):
    query_lower = query.lower()

    issue_key_pattern = re.compile(r'\b([A-Za-z0-9]+-[0-9]+)\b')
    list_keyword_pattern = re.compile(r'related\s+to\s+(\w+)')
    column_pattern = re.compile(r'(\w+)\s+as\s+(\w+)')
    complex_pattern = re.compile(r'what\s+are\s+the\s+(.+?)\s+(.+?)\s+issues\s+related\s+to\s+(.+)')

    match = issue_key_pattern.search(query)
    if (match):
        issue_key = match.group(1)
        filtered_df = df[df['issue_key'] == issue_key]

        if (filtered_df.empty):
            return f"No information found for issue key {issue_key}"

        for word in query.split():
            column = map_synonym_to_column(word)
            if (column):
                return filtered_df[column].iloc[0]

        return "Query does not specify a valid field (summary, description, issue_type, priority, assignee, created_date, labels)."

    match = complex_pattern.match(query_lower)
    if (match):
        column_content = match.group(1).strip()
        column_name = match.group(2).strip()
        keyword = match.group(3).strip()

        column_mapped = map_synonym_to_column(column_name)
        if (not column_mapped):
            return f"Column '{column_name}' not recognized."

        column_normalization = {}
        for column in df.columns:
            if (df[column].dtype == 'object'):
                unique_values = df[column].str.lower().unique()
                normalization_map = {value.lower(): value for value in unique_values}
                column_normalization[column.lower()] = normalization_map

        if (column_mapped in column_normalization):
            column_content_normalized = column_normalization[column_mapped].get(column_content.lower(), column_content)
        else:
            return "Query not recognized or conditions not met."

        if (column_mapped in df.columns):
            filtered_df = df[(df[column_mapped].str.lower() == column_content_normalized.lower()) & 
                             (df.apply(lambda row: keyword.lower() in row['summary'].lower(), axis=1))]
        else:
            return "Query not recognized or conditions not met."

        if (not filtered_df.empty):
            return filtered_df.to_dict(orient='records')
        else:
            return f"No matching issues found with {column_content_normalized} {column_mapped} related to {keyword}."

    match = list_keyword_pattern.search(query_lower)
    if (match):
        keyword = match.group(1)
        filtered_rows = df[df.apply(lambda row: keyword in row['summary'].lower(), axis=1)]

        if (not filtered_rows.empty):
            return filtered_rows.to_dict(orient='records')
        else:
            return "No matching issues found."

    match = column_pattern.search(query_lower)
    if (match):
        column_name = match.group(1)
        column_content = match.group(2)

        column_name_mapped = map_synonym_to_column(column_name)
        if (column_name_mapped):
            filtered_rows = df[df[column_name_mapped].str.lower() == column_content.lower()]
            if (not filtered_rows.empty):
                return filtered_rows.to_dict(orient='records')
            else:
                return f"No issues found with {column_name_mapped} as {column_content}."
        else:
            return f"Column '{column_name}' not recognized."

    return "Query not recognized or conditions not met."

def handle_similarity_check(user_input, dataframe):
    user_input_embedding = sentence_model.encode(user_input, convert_to_tensor=True)
    summary_embeddings = sentence_model.encode(dataframe['summary'].tolist(), convert_to_tensor=True)
    summary_similarities = util.pytorch_cos_sim(user_input_embedding, summary_embeddings)
    most_similar_index = summary_similarities.argmax().item()
    most_similar_row = dataframe.iloc[most_similar_index]
    similarity_score = summary_similarities[0][most_similar_index].item() * 100
    
    return most_similar_row, similarity_score

def simplify_text(text):
    sentences = sent_tokenize(text)
    simplified_sentences = []
    for sentence in sentences:
        sentence = re.sub(r'\b(complex|complicated|intricate|sophisticated)\b', 'simple', sentence, flags=re.IGNORECASE)
        sentence = re.sub(r'\b(approximately|around|about)\b', 'about', sentence, flags=re.IGNORECASE)
        simplified_sentences.append(sentence)
    return ' '.join(simplified_sentences)

def summarize_text(filepath):
    with open(filepath, 'r') as file:
        text = file.read()

    max_chunk_length = 1024
    chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]

    summaries = []
    for chunk in chunks:
        try:
            summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
            simplified_summary = simplify_text(summary)
            summaries.append(simplified_summary)
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
    
    full_summary = ' '.join(summaries)
    
    bullet_points = convert_to_bullet_points(full_summary)
    
    return bullet_points

def convert_to_bullet_points(text):
    sentences = sent_tokenize(text)
    bullet_points = ['• ' + sentence for sentence in sentences]
    return '\n'.join(bullet_points)

@app.route('/')
def index():
    return render_template('index2.3btn.html')

@app.route('/retrieve_query', methods=['POST'])
def retrieve_query():
    query = request.form.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    response = handle_query(query)
    return jsonify(response)

@app.route('/check_similarity', methods=['POST'])
def check_similarity():
    query = request.form.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    most_similar_row, similarity_score = handle_similarity_check(query, df)
    response = {
        "most_similar_row": most_similar_row.to_dict(),
        "similarity_score": similarity_score
    }
    return jsonify(response)

@app.route('/summarize', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        try:
            summary = summarize_text(filepath)
            return jsonify({"summary": summary})
        except Exception as e:
            flash(f'An error occurred: {e}')
            return redirect(request.url)
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
