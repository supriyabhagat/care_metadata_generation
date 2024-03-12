from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import torch
import configs
from utils import create_vector_db, get_files, load_model_and_tokenizer, parse_files, multiple_question_answering

# Run the main function
if __name__ == "__main__":
    
    # Code to Parse the Files 
    # Change it as per requirements

    print("\nParse project files...")

    supported_files = get_files(configs.PROJECT_FOLDER_PATH)
    if len(supported_files) == 0:
        print("No supported files found in the project folder")
        exit()
    else:
        print("Found %s files" % len(supported_files))

    texts, num_of_files_parsed = parse_files(supported_files)
    if num_of_files_parsed == 0:
        print("No files parsed")
        exit()
    else:
        print("Parsed %s / %s files" % (num_of_files_parsed, len(supported_files)))
    
    
    print("\nCreate vector db...")
    try:
        create_vector_db(texts)
    except Exception as e:
        print(e)
        exit()

    print("Vector db created successfully.")

    print("\nLoading model, tokenizer and db...")
    loaded_pipeline = load_model_and_tokenizer(configs.model_name_or_path)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cuda'})
    db = FAISS.load_local(configs.DB_FAISS_PATH, embeddings)
    print("Model, tokenizer, and db loaded successfully.")


    # For Predefined Questions
    questions = ["1. What's the category of the asset?",
        "2. Short description",
        "3. What does this asset do?",
        "4. How does this asset work?",
        "5. Model types",
        "6. Search tags"]
    
    llm_answers=multiple_question_answering(user_queries=questions, pipeline=loaded_pipeline, db=db)

    for answer in llm_answers:
        print("\n", answer)
             
    # clear cuda cache
    torch.cuda.empty_cache()