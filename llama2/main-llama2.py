from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from transformers import pipeline

model_name_or_path = "TheBloke/Llama-2-13B-chat-AWQ"

# Load model
print("Model loading...")
model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True,
                                          trust_remote_code=False, safetensors=True)
print("Model loaded.")

print("Tokenizer loading...")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
print("Tokenizer loaded.")

print("Reading context...")
context = open("llama2/context.txt", "r").read()
print("Context read.")

questions = ["1. What's the category of the project? (e.g., classification, regression, deep learning)",
        "2. Short description (limited to 150 characters)",
        "3. What does this asset do (up to 150 characters)",
        "4. How does this asset work (up to 150 characters)",
        "5. Model types (comma-separated list, limited to 50 characters)",
        "6. Search tags (comma-separated list)"]
questions = "\n".join(questions)

prompt_template=f'''[INST] <<SYS>>
You are a helpful, respectful and honest assistant. You have to answer the questions based on the context that you read. If a question does not make any sense, or is not mentioned in the context, 
explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Here is the context for this conversation: \n
{context}.\n According to this context, answer the following question:
<</SYS>>
{questions}[/INST]
'''

tokens = tokenizer(
    prompt_template,
    return_tensors='pt'
).input_ids.cuda()

while (True):
    input("Model Ready. Press Enter to generate...")
    pipe = pipeline(
        "text-generation",
        model=model.model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.1
    )
    print("\n\n********** GENERATED TEXT **********\n\n")
    print(pipe(prompt_template)[0]['generated_text'])
    print("\n\n********** END GENERATED TEXT **********\n\n")