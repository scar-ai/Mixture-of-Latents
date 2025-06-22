import streamlit as st
import torch
from MoE import TheTransformer
from torch.nn import functional as F
from transformers import AutoTokenizer

st.set_page_config(
    page_title="MoE Transformer Inference",
    page_icon="🤖",
    layout="wide",
)

if 'generation_in_progress' not in st.session_state:
    st.session_state.generation_in_progress = False

@st.cache_resource
def load_model_and_tokenizer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    N_HEADS = 20
    D_MODEL = 1600
    DROPOUT = 0.1
    N_BLOCKS = 12
    LATENT_DIM = 256

    model = TheTransformer(
        vocab_size=vocab_size,
        num_heads=N_HEADS,
        n_layers=N_BLOCKS,
        d_model=D_MODEL,
        latent_dim=LATENT_DIM,
        ignore_index=tokenizer.pad_token_id,
        dropout=DROPOUT,
        num_experts=12,
        topk_experts=2
    ).to(device)

    try:
        checkpoint = torch.load(r"weights/MLA+MoE_Finetuned.pth", map_location=device, weights_only=False)
        new_state_dict = {key.replace("module.", ""): value for key, value in checkpoint.items()}
        model.load_state_dict(new_state_dict)
    except FileNotFoundError:
        st.error("Weight file not found. Make sure 'weights/MLA+MoE_Finetuned.pth' exists.")
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None, None, None

    model.eval()
    return model, tokenizer, device

def generate_text_stream(model, tokenizer, device, prompt, max_length, temp, top_k, top_p, repetition_penalty):
    context = tokenizer(prompt, padding=False, truncation=False, return_tensors="pt")["input_ids"]
    context = torch.cat([torch.tensor([[tokenizer.bos_token_id]]).to(device), context.to(device)], dim=1)
    log_tokens = context.clone()

    with torch.no_grad():
        for _ in range(max_length):
            if not st.session_state.generation_in_progress:
                break

            logits = model(log_tokens)[:, -1, :]

            unique_tokens = torch.unique(log_tokens)
            for token_id in unique_tokens:
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

            if temp > 0:
                logits /= temp

            if top_k > 0:
                top_k_values, _ = torch.topk(logits, top_k)
                kth_value = top_k_values[:, -1]
                indices_to_remove = logits < kth_value
                logits[indices_to_remove] = float('-inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

            log_tokens = torch.cat((log_tokens, next_token), dim=1)
            decoded_token = tokenizer.decode(next_token.item())
            yield decoded_token

st.title("📝 Mixture-of-latents")
st.markdown("""
This app uses a custom Mixture-of-Experts (MoE) Transformer model to generate text based on your instructions.
Provide an instruction and an optional input, then adjust the generation parameters in the sidebar to control the output.
""")

if 'generation_in_progress' not in st.session_state:
    st.session_state.generation_in_progress = False
if 'last_response' not in st.session_state:
    st.session_state.last_response = None

@st.cache_resource
def get_model():
    with st.spinner("Loading model... This may take a moment."):
        model, tokenizer, device = load_model_and_tokenizer()
    return model, tokenizer, device

model, tokenizer, device = get_model()

if model is None:
    st.error("Failed to load the model.")
    st.stop()

model_params = sum(p.numel() for p in model.parameters()) / 1e6
st.success(f"Model loaded successfully on **{device.upper()}** ({model_params:.2f}M parameters).")

st.sidebar.header("Generation Parameters")
max_len = st.sidebar.slider("Max Length", 50, 500, 250, 10)
temp = st.sidebar.slider("Temperature", 0.1, 1.5, 0.8, 0.05)
top_k = st.sidebar.slider("Top-K", 0, 100, 40, 1)
top_p = st.sidebar.slider("Top-P (Nucleus)", 0.1, 1.0, 0.9, 0.05)
rep_penalty = st.sidebar.slider("Repetition Penalty", 1.0, 2.0, 1.2, 0.05)

st.header("Provide Input")
col1, col2 = st.columns(2)
with col1:
    instruction = st.text_area("Instruction:", "What is the capital of France?", height=150)
with col2:
    user_input = st.text_area("Input:", "", height=150)

button_col1, button_col2 = st.columns([1, 6]) 

with button_col1:
    if st.button("Generate", type="primary", disabled=st.session_state.generation_in_progress):
        if not instruction:
            st.warning("Please provide an instruction.")
        else:
            st.session_state.last_response = None
            st.session_state.generation_in_progress = True
            st.rerun()

with button_col2:
    if st.session_state.generation_in_progress:
        if st.button("Stop Generation"):
            st.session_state.generation_in_progress = False

if st.session_state.generation_in_progress:
    prompt = f"Instruction: {instruction}\nInput: {user_input}{tokenizer.eos_token}"
    st.markdown("---")
    st.subheader("Model Response")
    
    output_container = st.empty()
    full_response = ""
    
    try:
        generator = generate_text_stream(
            model, tokenizer, device, prompt, max_len, temp, top_k, top_p, rep_penalty
        )
        for token in generator:
            if not st.session_state.generation_in_progress:
                break
            full_response += token
            output_container.markdown(full_response + " ▌")
        
        st.session_state.last_response = full_response
        output_container.markdown(full_response)

    except Exception as e:
        st.error(f"An error occurred during generation: {e}")
    finally:
        st.session_state.generation_in_progress = False
        st.rerun()

elif st.session_state.last_response:
    st.markdown("---")
    st.subheader("Last Model Response")
    st.markdown(st.session_state.last_response)
