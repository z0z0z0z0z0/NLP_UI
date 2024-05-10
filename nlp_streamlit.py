import streamlit as st
import torch
import PyPDF2
import io
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import time

# Load RoBERTa model and tokenizer
model_name = "deepset/roberta-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

st.title("PDF GPT")


def extract_text_from_pdf(uploaded_file):
    # Read PDF contents
    pdf_text = ""
    with io.BytesIO(uploaded_file.getvalue()) as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text().replace('\n', ' ')
    
    return pdf_text


# Add custom CSS to set the width of the sidebar
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        width: 500px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# PDF Upload section
uploaded_file = st.file_uploader("Upload PDF file", type=['pdf'])
if uploaded_file:
    # Read PDF contents
    pdf_text = extract_text_from_pdf(uploaded_file)

    # summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    # try:
    #     summary = summarizer(pdf_text, max_length=250, min_length=25, do_sample=False)
    # except Exception as e:
    #     print("An error occurred while summarizing the article:", e)
    #     summary = "Unable to summarize the article due to an error (file too large)."

    # Display extracted PDF text in the sidebar
    st.sidebar.subheader("Summary:")
    #st.sidebar.write(summary[0]["summary_text"])
    
    # Display extracted PDF text in the sidebar
    st.sidebar.subheader("Extracted PDF Text:")
    st.sidebar.write(pdf_text)

    
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    avatar = message.get("avatar")  # Get avatar URL if available
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)



# Load the question-answering pipeline
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
        
# Function to get predictions for a given question
def get_answer(question, context):
    # Generate predictions using the pipeline
    res = nlp({
        'question': question,
        'context': context
    })
    return res



if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "https://raw.githubusercontent.com/z0z0z0z0z0/icon/main/icons8-avatar-64.png"})
    with st.chat_message("user",avatar='https://raw.githubusercontent.com/z0z0z0z0z0/icon/main/icons8-avatar-64.png'):
        st.markdown(prompt)

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
    
