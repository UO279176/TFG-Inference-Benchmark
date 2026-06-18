from inference.rkllm_lib.lib import RKLLMParam, RKLLMExtendParam, CALLBACK_FUNC, RKLLMResult, RKLLMInput, c_callback, RKLLMInferParam, RKLLM_INPUT_PROMPT, _RKLLMInputUnion, RKLLM_INFER_GENERATE, text_buffer
import ctypes
from ctypes import c_void_p, cast
from transformers import AutoTokenizer
import os

# Set the log level for RKLLM runtime to debug to see detailed logs
os.environ["RKLLM_RT_LOG_LEVEL"] = "1"
os.environ["RKLLM_LOG_LEVEL"] = "1"

model = ctypes.CDLL("/usr/lib/librkllmrt.so")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Aplica el formato de chat adecuado
messages = [{"role": "user", "content": "Hello, my dog is cute"}]
prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="np").flatten().tolist()
MODEL_PATH = "src/data/models/npu/tinyllama/tinyllama.rkllm"

model_handler = ctypes.c_void_p()

# Luego crea la estructura RKLLMTokenInput (necesitas definirla en lib.py)
class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t)
    ]
    
ids_array = (ctypes.c_int32 * len(prompt_ids))(*prompt_ids)
token_input = RKLLMTokenInput(input_ids=ids_array, n_tokens=len(prompt_ids))

model_params = RKLLMParam(
    model_path=MODEL_PATH.encode('utf-8'),
    max_context_len=4000,
    max_new_tokens=500,
    top_k=40,
    n_keep=0,
    top_p=0.9,
    temperature=0.7,
    repeat_penalty=1.1,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    mirostat=0,
    mirostat_tau=0.0,
    mirostat_eta=0.0,
    skip_special_token=False,
    is_async=False,
    img_start=b"",
    img_end=b"",
    img_content=b"",
    extend_param=RKLLMExtendParam(
        base_domain_id=1,
        embed_flash=0,
        enabled_cpus_num=4,
        enabled_cpus_mask=240,
        n_batch=1,
        use_cross_attn=0,
        reserved=(ctypes.c_uint8 * 104)()
    )
)

model.rkllm_init.argtypes = [
    ctypes.POINTER(ctypes.c_void_p), 
    ctypes.POINTER(RKLLMParam), 
    CALLBACK_FUNC
]
model.rkllm_init.restype = ctypes.c_int

ret = model.rkllm_init(ctypes.byref(model_handler), ctypes.byref(model_params), c_callback)
if ret != 0:
    raise RuntimeError(f"Error al inicializar TinyLlama en la NPU: código de error {ret}")

prompt = b"Hello, my dog is cute"

input_data = RKLLMInput(
    role=None,
    enable_thinking=False,
    input_type=RKLLM_INPUT_PROMPT,
    _input=_RKLLMInputUnion(
        prompt_input=prompt,
        embed_input=None,
        token_input=None,
        multimodal_input=None
    )
)

infer_params = RKLLMInferParam(
    mode=RKLLM_INFER_GENERATE,
    lora_params=None,
    prompt_cache_params=None,
    keep_history=0
)

text_buffer.clear()

model.rkllm_run.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(RKLLMInput),
    ctypes.POINTER(RKLLMInferParam),
    ctypes.c_void_p
]
model.rkllm_run.restype = ctypes.c_int
ret = model.rkllm_run(model_handler, ctypes.byref(input_data), ctypes.byref(infer_params), None)
if ret != 0:
    raise RuntimeError(f"Error al ejecutar la inferencia en TinyLlama: código de error {ret}")