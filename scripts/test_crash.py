import torch
from transformers import AutoModel, AutoTokenizer
import logging
import sys

# 1. SETUP LOGGING
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
print("--- SCRIPT STARTING ---", flush=True)

# 2. H200 STABILITY SETTINGS
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

MODEL_PATH = "OpenGVLab/InternVL2-8B"
DEVICE = "cuda"

print(f"--- LOADING MODEL ON {DEVICE} ---", flush=True)
# Load in BFloat16 (Native H200)
model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=False,
    attn_implementation="eager",
    trust_remote_code=True,
).to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# *** THE FIX ***
# Manually tell the model what the image token ID is
img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
model.img_context_token_id = img_context_token_id
print(f"DEBUG: Set img_context_token_id to {img_context_token_id}")

print("--- MODEL LOADED ---", flush=True)

# ---------------------------------------------------------
# DIAGNOSTIC 1: EMBEDDING SIZE MISMATCH
# ---------------------------------------------------------
embedding_size = model.get_input_embeddings().weight.shape[0]
vocab_size = len(tokenizer)
print(f"\nDEBUG: Model Embedding Matrix Size: {embedding_size}")
print(f"DEBUG: Tokenizer Vocab Size:       {vocab_size}")

if vocab_size > embedding_size:
    print("!!! WARNING: Tokenizer has more tokens than the model has embeddings!")
    print("!!! This acts as a 'Device-Side Assert' waiting to happen.")

img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
model.img_context_token_id = img_context_token_id

print("\n--- PREPARING INPUTS ---", flush=True)

# 1. Pixel Values (Simulate 3 patches)
# Shape: [3, 3, 448, 448]
raw_data = torch.randn(6, 3, 448, 448)
pixel_values = raw_data[::2].to(DEVICE).to(torch.bfloat16)

if not pixel_values.is_contiguous():
    pixel_values = pixel_values.contiguous()

# 2. Input IDs (THE LOGIC FIX)
# Calculate exactly how many tokens we need
num_patches = pixel_values.shape[0] 
tokens_per_patch = 256
total_img_tokens = num_patches * tokens_per_patch

print(f"DEBUG: Visual Patches: {num_patches}")
print(f"DEBUG: Required Tokens: {total_img_tokens}")

# Construct the ID sequence manually:
# [Start Token] + [768 Image Tokens] + [Text Tokens]
prefix = torch.tensor([1]).to(DEVICE)
img_fillers = torch.tensor([img_context_token_id] * total_img_tokens).to(DEVICE)
suffix = torch.tensor([200, 300]).to(DEVICE) # "Describe this"

# Concatenate them into one long tensor
input_ids = torch.cat([prefix, img_fillers, suffix]).unsqueeze(0) # Add batch dimension
attention_mask = torch.ones_like(input_ids).to(DEVICE)

print(f"DEBUG: Final Input ID Shape: {input_ids.shape}")

print(f"Pixel values contiguous? {pixel_values.is_contiguous()}")
print(f"Max Token ID in input: {input_ids.max().item()}")

# ---------------------------------------------------------
# THE FIXES
# ---------------------------------------------------------

# FIX A: Force Contiguous Memory
if not pixel_values.is_contiguous():
    print("Applying Fix A: Making tensor contiguous...", flush=True)
    pixel_values = pixel_values.contiguous()

# FIX B: Resize Embeddings (Uncomment if Diagnostic 1 showed a mismatch)
if input_ids.max().item() >= embedding_size:
    print("Applying Fix B: Resizing model embeddings...", flush=True)
    model.resize_token_embeddings(vocab_size)

# ---------------------------------------------------------
# GENERATE
# ---------------------------------------------------------
print("\n--- STARTING GENERATION ---", flush=True)
from torch.backends.cuda import sdp_kernel

try:
    with sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
        response = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=10,
            do_sample=False
        )
    print(response)
    print("\n!!! SUCCESS !!! Generation completed.")
    
except Exception as e:
    print(f"\nCRITICAL FAILURE TYPE: {type(e).__name__}")
    print(f"CRITICAL FAILURE MSG:  {e}")
    # If message is empty, it's likely an AssertionError in C++
    import traceback
    traceback.print_exc()

print("--- SCRIPT FINISHED ---")