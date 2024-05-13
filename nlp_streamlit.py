import streamlit as st
import torch
import PyPDF2
import io
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import time
from autocorrect import Speller
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


# Load RoBERTa model and tokenizer
model_name = "deepset/roberta-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

st.set_page_config(
    page_title="PDF GPT",
    page_icon="🤖",
    layout="wide"
)

st.title("PDF GPT")


def extract_text_from_pdf(uploaded_file):
    # Read PDF contents
    pdf_text = ""
    with io.BytesIO(uploaded_file.getvalue()) as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text().replace('\n', ' ')
    
    return pdf_text


# Load question-answering pipeline with custom preprocessing
def custom_preprocess_function(examples):
    
    for example in examples:
        sentence = example[extracted_text]  # Assuming "context" key for your passage
        inputs = tokenizer(
            sentence,
            truncation=True,
            return_overflowing_tokens=True,
            max_length=20,
            stride=2
        )
        example["inputs"] = inputs
    return examples


# Add custom CSS to set the width of the sidebar
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        width: 500px;
    }

    .stButton>button {
        background-color: #4CAF50; /* Green background */
        color: white;
        text-align: center;
        text-decoration: none;
        font-size: 16px;
        transition-duration: 0.4s;
        cursor: pointer;
    }
    
    .stButton>button:hover {
        background-color: #45a049; /* Darker green background on hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)


def reset_conversation():
    st.session_state.messages = None

def textToSpeech(summary):
    engine = pyttsx3.init()
    engine.say(summary)
    engine.runAndWait()


previous_uploaded_file = st.session_state.get("previous_uploaded_file")

# PDF Upload section
uploaded_file = st.file_uploader("Upload PDF file", type=['pdf'])

summary_text=''
pdf_text =''

if uploaded_file:
    # Check if it's the first time uploading a PDF file
    if previous_uploaded_file is None:
        # Perform any necessary initialization
        # For example, you might want to set previous_uploaded_file to the current uploaded_file\
        with st.spinner("Reading PDF....."):
            pdf_text = extract_text_from_pdf(uploaded_file)

            summarizer = pipeline("summarization", model="Falconsai/text_summarization")
            try:
                summary = summarizer(pdf_text, max_length=250, min_length=25, do_sample=False)
                summary_text = summary[0]["summary_text"]
            except Exception as e:
                print("An error occurred while summarizing the article:", e)
                summary = "Unable to summarize the article due to an error (file too large)."
                    
            st.success("Successfully Reading PDF")
            st.toast('Successful Read PDF', icon='✅')
            
            # Update the previous uploaded file in session state
            st.session_state.previous_uploaded_file = uploaded_file
            
    else:
        # Check if the current uploaded file is different from the previous one
        if uploaded_file != previous_uploaded_file:
            # Read PDF contents
            with st.spinner("Reading PDF....."):
                pdf_text = extract_text_from_pdf(uploaded_file)

                summarizer = pipeline("summarization", model="Falconsai/text_summarization")
                try:
                    summary = summarizer(pdf_text, max_length=250, min_length=25, do_sample=False)
                    summary_text = summary[0]["summary_text"]
                except Exception as e:
                    print("An error occurred while summarizing the article:", e)
                    summary = "Unable to summarize the article due to an error (file too large)."
                    
            st.success("Successfully Reading PDF")
            st.toast('Successful Read PDF', icon='✅')

            # Update the previous uploaded file in session state
            st.session_state.previous_uploaded_file = uploaded_file

        else:
            pdf_text = extract_text_from_pdf(uploaded_file)

            summarizer = pipeline("summarization", model="Falconsai/text_summarization")
            try:
                summary = summarizer(pdf_text, max_length=250, min_length=25, do_sample=False)
                summary_text = summary[0]["summary_text"]
            except Exception as e:
                print("An error occurred while summarizing the article:", e)
                summary = "Unable to summarize the article due to an error (file too large)."


#Display extracted PDF text in the sidebar
st.sidebar.subheader("Summary:")
st.sidebar.write(summary_text)

st.sidebar.subheader("Options:")
# Create two columns
col1, col2 = st.sidebar.columns([2, 3])

# Button to clear chat
if col1.button("Clear chat", on_click=reset_conversation):
    pass


# if col2.button("Text-to-Speech of Summarisation"):
#     with st.spinner("Reading summarisation"):
#         textToSpeech(summary_text)
#         st.toast('Summarisation Speech Finish', icon='🔊')

st.sidebar.write('')
st.sidebar.write('')
    
# Display extracted PDF text in the sidebar
st.sidebar.subheader("Extracted PDF Text:")
st.sidebar.write(pdf_text)


    
if "messages" not in st.session_state or st.session_state.messages is None:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    avatar = message.get("avatar")  # Get avatar URL if available
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)



# Load the question-answering pipeline
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer, preprocessing_function=custom_preprocess_function, is_fast=True)
        
# Function to get predictions for a given question
def get_answer(question, context):
    # Generate predictions using the pipeline
    res = nlp({
        'question': question,
        'context': context
    })
    return res


# Create a Speller object
spell = Speller(lang='en')


if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "https://raw.githubusercontent.com/z0z0z0z0z0/icon/main/icons8-avatar-64.png"})
    with st.chat_message("user",avatar='https://raw.githubusercontent.com/z0z0z0z0z0/icon/main/icons8-avatar-64.png'):
        st.markdown(prompt)
        prompt = spell(prompt)

    # Display assistant's responses
    with st.chat_message("assistant",avatar='https://raw.githubusercontent.com/z0z0z0z0z0/icon/main/icons8-bot-64.png'):
        message_placeholder = st.empty()
        full_response = ""

        # Check if the question is not empty
        if prompt:
            
            # Get the answer for the question
            ans = get_answer(prompt, pdf_text)

            messages = [
                {"role": m["role"], "content": m["content"], "avatar": m.get("avatar")}
                for m in st.session_state.messages
            ]
            
            # Iterate over each character in the answer
            for char in ans['answer']:
                # Accumulate characters progressively
                full_response += char

                # Display the accumulated response with "▌" separators
                message_placeholder.markdown(full_response + "▌")

                # Add a short delay to simulate typing
                time.sleep(0.05)

            
            
            message_placeholder.markdown(full_response)

    
    st.session_state.messages.append({"role": "assistant", "content": full_response, 'avatar':'https://raw.githubusercontent.com/z0z0z0z0z0/icon/main/icons8-bot-64.png'})
    
