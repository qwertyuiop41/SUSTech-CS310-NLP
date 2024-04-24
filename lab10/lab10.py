from transformers import AutoTokenizer, GPT2LMHeadModel

gpt2_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="gpt2zh")
gpt2_model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path="gpt2zh")
# Evaluation mode
gpt2_model.eval()

print('vocab size:', gpt2_tokenizer.vocab_size)
print(f'special token {gpt2_tokenizer.sep_token}:', gpt2_tokenizer.sep_token_id)
print(f'special token {gpt2_tokenizer.cls_token}:', gpt2_tokenizer.cls_token_id)
print(f'special token {gpt2_tokenizer.pad_token}:', gpt2_tokenizer.pad_token_id)

# Use [SEP] as end-of-sentence token
gpt2_model.config.eos_token_id = gpt2_tokenizer.sep_token_id