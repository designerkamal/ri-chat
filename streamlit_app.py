import streamlit as st
import os
# from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import tempfile
import math

# Ensure you've set your OpenAI API key in your environment variables
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Mortality Score Calculator Function
def calculate_mortality_score(inputs):
    # Convert inputs from strings to floats
    inputs = {k: float(v) for k, v in inputs.items()}
    
    # Define conversion factors
    conv = {
        'Albumin': 10,
        'Creatinine': 88.401,
        'Glucose': 0.0555,
        'C-reac Protein': 0.1
    }
    
    # Define weights
    wts = {
        'Albumin': -0.0336,
        'Creatinine': 0.0095,
        'Glucose': 0.1953,
        'C-reac Protein': 0.0954,
        'Lympocyte': -0.012,
        'Mean Cell Volume': 0.0268,
        'Red Cell Dist Width': 0.3306,
        'Alkaline Phosphatase': 0.0019,
        'White Blood Cells': 0.0554,
        'Age': 0.0804
    }
    
    # Constants
    gamma = 0.0076927
    b0 = -19.9067
    t = 120  # months
    
    # Calculate converted inputs
    cinputs = {k: inputs[k] * conv[k] if k in conv else inputs[k] for k in inputs}
    
    # Special case for C-reac Protein: natural log
    if 'C-reac Protein' in cinputs:
        cinputs['C-reac Protein'] = math.log(cinputs['C-reac Protein'])
    
    # Calculate terms (cinput * weight)
    terms = {k: cinputs[k] * wts[k] for k in cinputs}
    
    # Calculate linear combination, including b0
    lin_comb = sum(terms.values()) + b0
    
    # Calculate MortScore
    mort_score = 1 - math.exp(-math.exp(lin_comb) * (math.exp(gamma * t) - 1) / gamma)
    
    # Calculate Ptypic Age
    ptypic_age = 141.50225 + math.log(-0.00553 * math.log(1 - mort_score)) / 0.090165
    
    # Calculate est. DNAm Age
    est_dnam_age = ptypic_age / (1 + 1.28047 * math.exp(0.0344329 * (-182.344 + ptypic_age)))
    
    # Calculate est. D MScore
    est_d_mscore = 1 - math.exp(-0.000520363523 * math.exp(0.090165 * est_dnam_age))
    
    return lin_comb, mort_score, ptypic_age, est_dnam_age, est_d_mscore

# Document Processing Function
@st.cache_resource
def process_document(_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    
    os.unlink(tmp_file_path)
    return db

# Initialize LangChain agent with GPT-4
@st.cache_resource
def initialize_langchain_agent():
    document_tool = Tool(
        name="Document Processor",
        func=process_document,
        description="Processes and analyzes uploaded blood test reports in PDF format."
    )

    calculator_tool = Tool(
        name="Mortality Score Calculator",
        func=calculate_mortality_score,
        description="Calculates mortality scores based on input health metrics."
    )

    lifestyle_template = """
    Based on the user's health scores and lifestyle information, provide personalized advice:

    Health Scores: {scores}
    Lifestyle Info: {lifestyle}

    Advice:
    """
    lifestyle_prompt = PromptTemplate(
        input_variables=["scores", "lifestyle"],
        template=lifestyle_template
    )

    lifestyle_chain = LLMChain(llm=ChatOpenAI(model_name="gpt-4o", temperature=0.7), prompt=lifestyle_prompt)

    lifestyle_tool = Tool(
        name="Lifestyle Advisor",
        func=lifestyle_chain.run,
        description="Provides personalized lifestyle advice based on health scores and current habits."
    )

    tools = [document_tool, calculator_tool, lifestyle_tool]

    return initialize_agent(
        tools,
        ChatOpenAI(model_name="gpt-4o", temperature=0.5),
        agent="conversational-react-description",
        verbose=True
    )

# PDF Analysis Function
def analyze_pdf(pdf_doc, query):
    chain = load_qa_chain(ChatOpenAI(model_name="gpt-4o", temperature=0), chain_type="stuff")
    docs = pdf_doc.similarity_search(query)
    return chain.run(input_documents=docs, question=query)

# Streamlit UI
st.title("Health Analysis Chatbot (Powered by GPT-4o)")

# File upload
uploaded_file = st.file_uploader("Upload your blood test report (PDF)", type=["pdf"])

if uploaded_file is not None:
    st.write("File uploaded successfully!")
    
    # Process document
    with st.spinner("Processing document..."):
        pdf_doc = process_document(uploaded_file)
    st.success("Document processed!")

    # Initialize agent
    agent = initialize_langchain_agent()

    # Extract health metrics
    metrics_query = """
    Extract the following specific health metrics from the blood test report:
    1. Albumin (g/dL)
    2. Creatinine (mg/dL)
    3. Glucose (mg/dL)
    4. C-reactive Protein (mg/L)
    5. Lymphocyte (%)
    6. Mean Cell Volume (fL)
    7. Red Cell Distribution Width (%)
    8. Alkaline Phosphatase (U/L)
    9. White Blood Cell Count (10^3 cells/µL)

    If the age of the patient is mentioned, please include it as well.
    For each metric, provide the value and the unit of measurement.
    If any of these metrics are not present in the report, indicate that they are missing.
    Present the results in a JSON format with metric names as keys and their values (including units) as the corresponding values.
    """
    metrics_str = analyze_pdf(pdf_doc, metrics_query)
    st.write("Extracted Metrics:", metrics_str)

    # Parse the metrics string
    try:
        # Remove any leading/trailing whitespace and extract the JSON part
        metrics_str = metrics_str.strip()
        json_match = re.search(r'\{.*\}', metrics_str, re.DOTALL)
        if json_match:
            metrics_json = json_match.group()
            metrics = json.loads(metrics_json)
        else:
            raise ValueError("No valid JSON object found in the string")
    except json.JSONDecodeError as e:
        st.error(f"Error parsing metrics JSON: {e}")
        metrics = {}
    except ValueError as e:
        st.error(f"Error extracting JSON from string: {e}")
        metrics = {}
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        metrics = {}

    # Manual input for missing values
    st.write("Please fill in any missing values or correct extracted ones:")
    inputs = {}
    for metric in ['Albumin', 'Creatinine', 'Glucose', 'C-reac Protein', 'Lympocyte', 
                   'Mean Cell Volume', 'Red Cell Dist Width', 'Alkaline Phosphatase', 
                   'White Blood Cells', 'Age']:
        # Extract numeric value from metrics string, if present
        value = ""
        if metric in metrics:
            try:
                # Use regex to extract the numeric part
                match = re.search(r'(\d+\.?\d*)', metrics[metric])
                if match:
                    value = match.group(1)
            except:
                pass
        inputs[metric] = st.text_input(f"{metric} (numeric value only)", value=value)

    if st.button("Calculate Scores"):
        # Calculate scores
        lin_comb, mort_score, ptypic_age, est_dnam_age, est_d_mscore = calculate_mortality_score(inputs)
        
        # Display results
        st.write("Results:")
        st.write(f"LinComb: {lin_comb:.2f}")
        st.write(f"MortScore: {mort_score:.6f}")
        st.write(f"Ptypic Age: {ptypic_age:.2f}")
        st.write(f"est. DNAm Age: {est_dnam_age:.2f}")
        st.write(f"est. D MScore: {est_d_mscore:.6f}")

        # Interpret results
        interpretation_query = f"""
        Interpret these health scores in detail:
        LinComb={lin_comb:.2f}, MortScore={mort_score:.6f}, Ptypic Age={ptypic_age:.2f},
        est. DNAm Age={est_dnam_age:.2f}, est. D MScore={est_d_mscore:.6f}
        Explain what each score means and its implications for the person's health.
        """
        interpretation = analyze_pdf(pdf_doc, interpretation_query)
        st.write("Interpretation:", interpretation)

    # Lifestyle assessment
    lifestyle = st.text_area("Please describe your current lifestyle habits (diet, exercise, sleep, etc.):")
    
    if st.button("Get Lifestyle Advice"):
        # Provide recommendations
        scores = f"LinComb={lin_comb:.2f}, MortScore={mort_score:.6f}, Ptypic Age={ptypic_age:.2f}, est. DNAm Age={est_dnam_age:.2f}, est. D MScore={est_d_mscore:.6f}"
        advice_query = f"""
        Based on these health scores: {scores}
        And this lifestyle description: {lifestyle}
        Provide detailed, personalized lifestyle advice to improve health outcomes.
        Include specific recommendations for diet, exercise, sleep, and stress management.
        """
        advice = analyze_pdf(pdf_doc, advice_query)
        st.write("Lifestyle Advice:", advice)

    # Chat interface
    st.write("Ask any questions about your health analysis:")
    user_question = st.text_input("Your question")
    if user_question:
        response = analyze_pdf(pdf_doc, user_question)
        st.write("Response:", response)

st.write("Note: This chatbot provides general information and should not be considered as a substitute for professional medical advice.")