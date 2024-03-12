import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, pipeline
import textract
import configs

from tika import tika, parser
tika.TikaJarPath = os.path.expanduser("~")

# Get all supported files in the projects folder and its subfolders
def get_files(path):
    try:
        supported_formats = [".pdf", ".docx", ".ipynb", ".py", ".md", ".pptx", ".xls", ".xlsx", ".csv"]
        supported_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(tuple(supported_formats)):
                    supported_files.append(os.path.join(root, file))
        return supported_files
    except Exception as e:
        print(e)
        return []


# Parse the file based on the file extension   
def parse_single_file(file_name):
    
    textract_file_types = [".pptx", ".docx", ".xlsx", ".xls", ".csv"]
    tika_file_types = [".pdf"]
    python_standard_file_types = [".ipynb", ".py", ".md"]

    if file_name.endswith(tuple(textract_file_types)):
        try:
            # Parse using textract
            text = textract.process(file_name).decode("utf-8")
            return text
        except Exception as e:
            raise Exception("Failed to parse file using textract") from e
    if file_name.endswith(tuple(tika_file_types)):
        try:
            # Parse using tika
            parsed_pdf = parser.from_file(file_name)
            data = parsed_pdf['content']
            return data
        except Exception as e:
            raise Exception("Failed to parse file using tika") from e
    if file_name.endswith(tuple(python_standard_file_types)):
        try:
            # Parse using python standard library
            with open(file_name, "r") as f:
                data = f.read()
            return data
        except Exception as e:
            raise Exception("Failed to parse file using python standard library") from e
    else:
        raise Exception("File type not supported")


# Function to parse multiple files with different extensions 
def parse_files(files):
    texts = ""
    i = 0
    for file in files:
        try:
            parsed_text = parse_single_file(file)
            print("%s parsed successfully" % file)
            texts += "\n" + parsed_text
            i += 1
        except Exception as e:
            print("%s failed to parse" % file)
            print(e)
    return texts, i

# Function to create FAISS VectorDB
def create_vector_db(texts):
    try: 
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        split_texts = text_splitter.split_text(texts)

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                        model_kwargs={'device': 'cuda'})
        db = FAISS.from_texts(split_texts, embeddings)
        db.save_local(configs.DB_FAISS_PATH)
    except Exception as e:
        raise Exception("Failed to create VectorDB") from e 


# Load model and tokenizer
def load_model_and_tokenizer(model_name_or_path):
    try:
        model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True,
                                                trust_remote_code=False, safetensors=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
        pipe = pipeline("text-generation", model=model.model, tokenizer=tokenizer, max_new_tokens=512,
                    do_sample=True, temperature=0.7, top_p=0.95, top_k=40, repetition_penalty=1.1)
    except Exception as e:
        raise Exception("Failed to Load Model or Tokenizer") from e    
    else:
        return pipe


# Function to retrieve context from the database
def get_context_from_db(queries, db):
    try:
        retriever = db.as_retriever(search_kwargs={'k': 5},return_source_documents=True)
        contexts = [retriever.get_relevant_documents(query) for query in queries]
    except Exception as e:
        raise Exception("Failed to retrieve context from DB") from e    
    else:   
        return contexts


# Function to generate response using your LLM approach
def generate_response_with_llm(context, questions, pipeline):
    prompt_template=f'''
    [INST] <<SYS>>

    You are a highly knowledgeable assistant trained to analyze asset data and
    provide relevant metadata, short descriptions, and search tags.
    Your goal is to offer accurate information based on the provided data, 
    which can include words, code snippets, or any other relevant content.

    Tasks:
    1. Understand the context of the asset data.
    2. Generate concise metadata that describes the key attributes of the asset.
    3. Craft a short, informative description of the asset.
    4. Identify and assign appropriate search tags for efficient categorization.

    You will be asked one of the following questions only:
    1. What's the category of the project?
    2. Short description 
    3. What does this asset do
    4. How does this asset work
    5. Model types
    6. Search tags

    Remember:
    - Be precise and accurate in your responses.
    - If certain information is unclear or ambiguous, ask clarifying questions instead of guessing.
    - Prioritize relevance and conciseness in your outputs.

    Please make sure to answer them with the context provided.
    
    Here is the context for this conversation: \n
    {context}.\n According to this context, answer this question:
    <</SYS>>
    {questions}
    [/INST]
    '''

    generated_text = pipeline(prompt_template)[0]['generated_text']
    return generated_text.split('[/INST]')[-1].strip()


def multiple_question_answering(user_queries, pipeline, db):
    try:
        contexts = get_context_from_db(user_queries, db)

        responses = []
        for context, query in zip(contexts, user_queries):
            response = generate_response_with_llm(context, query, pipeline)
            responses.append(response)

    except Exception as e:
        raise Exception("Failed to answer questions") from e   
     
    else:   
        return responses
