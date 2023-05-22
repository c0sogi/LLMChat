from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained("/path/to/ggml-gpt-2.bin", model_type="gpt2")

print(llm("AI is going to"))


tokens = llm.tokenize("AI is going to")

for token in llm.generate(tokens):
    print(llm.detokenize([token]))
