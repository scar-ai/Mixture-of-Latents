import torch
from model import TheTransformer
from torch.nn import functional as F
from transformers import AutoTokenizer

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token

vocab_size = len(tokenizer)

N_HEADS = 16
D_MODEL = 1280
DROPOUT = 0.1
N_BLOCKS = 12
LATENT_DIM = 512

model = TheTransformer(vocab_size=vocab_size, num_heads=N_HEADS, n_layers=N_BLOCKS, d_model=D_MODEL, latent_dim=LATENT_DIM,
                    ignore_index=tokenizer.pad_token_id, dropout=DROPOUT).to(device)

print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
checkpoint = torch.load(r"weights/fast_distributed1.pth", map_location=device, weights_only=False)
new_state_dict = {key.replace("module.", ""): value for key, value in checkpoint.items()}
model.load_state_dict(new_state_dict)


model.eval()
def generate_text(prompt, max_length=100, temp=0.7, top_k=50, top_p=0.9, repetition_penalty=1.2):
    context = tokenizer(prompt, padding=False, truncation=False, return_tensors="pt")["input_ids"]
    context = torch.cat([torch.tensor([[tokenizer.bos_token_id]]).to(device), context.to(device)], dim=1)
    log_tokens = context.to(device)

    with torch.no_grad():
        for _ in range(max_length):
            data_pred = model(log_tokens)

            logits = data_pred[:, -1, :].clone()

            for token_id in set(log_tokens[0].tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

            logits /= temp

            if top_k > 0:
                top_k_indices = torch.topk(logits, top_k, dim=-1).indices
                mask = torch.ones_like(logits, dtype=torch.bool)
                mask.scatter_(1, top_k_indices, False)
                logits[mask] = float('-inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                if sorted_indices_to_remove[..., 1:].shape[0] > 0:
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = float('-inf')


            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            next_token = dist.sample().unsqueeze(0)

            log_tokens = torch.cat((log_tokens, next_token), dim=1)


            decoded_token = tokenizer.decode(next_token.item())
            print(decoded_token, end="", flush=True)

            if next_token.item() == tokenizer.eos_token_id:
                break

    print("\n") 
    return tokenizer.decode(log_tokens[0])


while True:
    print('------------------ User: ------------------')
    user_input = input('> ')
    print('------------------ Model: ------------------')
    generated_text = generate_text(
        prompt=user_input,
        max_length=150,
        temp=0.50,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.2
    )

