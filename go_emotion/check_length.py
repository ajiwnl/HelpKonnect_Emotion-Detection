from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/bert-base-go-emotion")

# Print the tokenizer's max_length
print(f"Tokenizer max_length: {tokenizer.model_max_length}")