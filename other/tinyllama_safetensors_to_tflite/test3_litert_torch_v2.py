import torch
import litert_torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Tiny Llama model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"Loading model: {model_name}")

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # Use float32 for TFLite conversion
    device_map="cpu"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set model to evaluation mode
model.eval()

# Prepare sample input
# The conversion needs valid shapes and dtypes, but this is not quantization
# calibration. Real tokens make the exported graph representative of runtime.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

sample_text = "Summarize the following news article:\n\nThis is a short news article for conversion."
inputs = tokenizer(
    sample_text,
    return_tensors="pt",
    max_length=128,
    truncation=True,
    padding="max_length",
)

sample_input_ids = inputs["input_ids"].to(dtype=torch.long)
sample_attention_mask = inputs["attention_mask"].to(dtype=torch.long)

# Convert using AI Edge Torch
print("Converting model to TFLite...")
try:
    # Wrap the model's forward method for conversion
    class TinyLlamaWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, input_ids, attention_mask):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False  # Disable cache for TFLite conversion
            )
            return outputs.logits
    
    # Create wrapped model
    wrapped_model = TinyLlamaWrapper(model)
    wrapped_model.eval()
    
    # Perform conversion
    edge_model = litert_torch.convert(
        wrapped_model, 
        (sample_input_ids, sample_attention_mask)
    )
    
    # Save as TFLite file
    edge_model.export("tinyllama.tflite")
    print("Model successfully converted and saved as 'tinyllama.tflite'")
    
except Exception as e:
    print(f"Conversion error: {e}")
    print("\nAlternative approach: Converting smaller components")