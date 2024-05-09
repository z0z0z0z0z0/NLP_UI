# Load necessary libraries
from streamlit import session_state as ss
from streamlit_pdf_viewer import pdf_viewer
import streamlit as st
import PyPDF2
import io
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Load model and tokenizer outside the tabs
model_name = "deepset/roberta-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize session state variables
if 'pdf_ref' not in ss:
    ss.pdf_ref = None
if 'question_input' not in ss:
    ss.question_input = ""
if 'answer' not in ss:
    ss.answer = ""
if 'score' not in ss:
    ss.score = ""

st.title("PDF Question Answering")

# Create tabs
tab1, tab2 = st.tabs(['Upload PDF', 'QA Model'])

# Tab 1 - PDF Viewer
with tab1:
    st.subheader("PDF Viewer")
    # Access the uploaded ref via a key.
    uploaded_file = st.file_uploader("Upload PDF file", type=('pdf'), key='pdf')
    
    # Checkbox to determine if extracted text should be displayed
    display_text = st.checkbox("Display extracted text", value=False)

    # Check if the file uploader component is cleared
    if uploaded_file is None:
        # Reset session state variables when PDF is not uploaded
        ss.pdf_ref = None
        ss.question_input = ""
        ss.answer = ""
        ss.score = ""
        st.write("Please upload a PDF file.")
    elif ss.pdf_ref != uploaded_file:
        # Reset session state variables when a new PDF is uploaded
        ss.pdf_ref = uploaded_file
        ss.question_input = ""
        ss.answer = ""
        ss.score = ""

    # Now you can access "pdf_ref" anywhere in your app.
    if ss.pdf_ref:
        binary_data = ss.pdf_ref.getvalue()

        # Try to extract text from the PDF
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(binary_data))
            has_extractable_text = any(page.extract_text() for page in pdf_reader.pages)
            if not has_extractable_text:
                st.warning("This PDF file does not contain extractable text. Please upload a copyable PDF.")
            elif display_text:
                # Display extracted text if checkbox is checked
                st.write("Extracted Text:")
                for page in pdf_reader.pages:
                    st.write(page.extract_text())
                
                # Display PDF viewer
                pdf_viewer(input=binary_data, width=700)
            else:
                # Display PDF viewer if the PDF can be read and contains extractable text
                pdf_viewer(input=binary_data, width=700)
        except PyPDF2.utils.PdfReadError:
            st.error("This PDF file cannot be read. Please upload a valid PDF.")
            # Reset session state variables when PDF cannot be read
            ss.pdf_ref = None
            ss.question_input = ""
            ss.answer = ""
            ss.score = ""

# Tab 2 - Extracted Text and ChatBot
with tab2:
    st.subheader("PDF GPT")
    
    # Check if the PDF file is uploaded and extracted
    if ss.pdf_ref:
        binary_data = ss.pdf_ref.getvalue()
        # Extract text from the PDF
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(binary_data))
        context_input = ""
        for page in pdf_reader.pages:
            context_input += page.extract_text()
        
        # Get the user question
        question_input = st.text_input("Question:", value=ss.question_input)
        
        # Store the question input in session state
        ss.question_input = question_input
        
        # Check if the question is not empty
        if question_input:
            # Load the question-answering pipeline
            nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

            # Get the answer for the question
            ans = nlp(question=question_input, context=context_input)

            # Store the answer and score in session state
            ss.answer = ans['answer']
            ss.score = ans['score']

            # Display the answer
            st.text_area("Answer:", value=ss.answer)
            st.write("Score:", ss.score)
    else:
        st.write("Please upload a PDF file in 'Upload PDF' tab")


css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:2rem;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)
