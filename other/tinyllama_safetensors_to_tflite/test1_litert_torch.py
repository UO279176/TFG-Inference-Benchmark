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
# Tiny Llama supports up to 2048 tokens, but using smaller size for memory efficiency
sample_text = "Hello, how are you?"
inputs = tokenizer(sample_text, return_tensors="pt", max_length=128, padding=True)

# Prepare input tensors (input_ids and attention_mask)
sample_input_ids = torch.randint(0, model.config.vocab_size, (1, 128))
sample_attention_mask = torch.ones(1, 128, dtype=torch.long)

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