import math
import time

import torch
from torch import nn

import torch.nn.init as init
from torch.nn import functional as F

from torch.utils.data import DataLoader

from model import TheTransformer
from OpenWebText import OpenWebTextDataset

from transformers import DataCollatorForLanguageModeling, AutoTokenizer

from torch.optim.lr_scheduler import _LRScheduler

#--------------------------------------------------
# Data Preparation
#--------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
vocab_size = len(tokenizer)

scaler = torch.amp.GradScaler(device)


class ShiftedDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __call__(self, features):
        batch = super().__call__(features)

        shifted_input_ids = []
        shifted_labels = []
        shifted_attention_masks = []

        input_ids, labels, attention_mask = batch["input_ids"], batch["labels"], batch["attention_mask"]

        shifted_input = input_ids[:, 0:-1].squeeze(0)
        shifted_input_ids.append(shifted_input)

        shifted_label = labels[:, 1:].squeeze(0)
        shifted_label[shifted_label == -100] = tokenizer.pad_token_id
        shifted_labels.append(shifted_label)

        shifted_attention = attention_mask[:, 0:-1]
        shifted_attention_masks.append(shifted_attention)
        
        batch["input_ids"] = torch.nn.utils.rnn.pad_sequence(
            shifted_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        batch["labels"] = torch.nn.utils.rnn.pad_sequence(
            shifted_labels, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        batch["attention_mask"] = torch.nn.utils.rnn.pad_sequence(
            shifted_attention_masks, batch_first=True, padding_value=0
        )

        return batch


print( tokenizer.pad_token_id)


print('Processing Dataset...')
dataset = OpenWebTextDataset(tokenizer=tokenizer, device=device, split='train', max_length=512, load_path =r"../Transformerv2/data/openweb/train")

data_collator = ShiftedDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm = False)

train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn = data_collator)
test_dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn = data_collator)
print(next(iter(train_dataloader)))

#--------------------------------------------------
# Model Loading
#--------------------------------------------------


def initialize_weights(module):
    if isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        init.normal_(module.weight, mean=0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        init.normal_(module.weight, mean=1.0, std=0.01)
        init.zeros_(module.bias)


N_HEADS = 16
D_MODEL = 1280
DROPOUT = 0.1
N_BLOCKS = 12
LATENT_DIM = 512

model = TheTransformer(vocab_size=vocab_size, num_heads=N_HEADS, n_layers=N_BLOCKS, d_model=D_MODEL, latent_dim=LATENT_DIM,
                    ignore_index=tokenizer.pad_token_id, dropout=DROPOUT).to(device)

for param in model.parameters():
    param.requires_grad = True


print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')


#--------------------------------------------------
# Training Hyperparameters
#--------------------------------------------------

n_epochs = 5
base_lr = 1e-4
weight_decay = 0.01

optimizer = torch.optim.AdamW(model.parameters(), weight_decay=weight_decay, lr=base_lr)
T_0 = len(train_dataloader)
T_mult = 1


class WarmupAndStep(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, plateaus1, plateaus2, decay_factor, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.plateau1 = plateaus1
        self.plateau2 = plateaus2
        self.decay_factor = decay_factor
        self.current_lrs = [group['lr'] for group in optimizer.param_groups]
        super(WarmupAndStep, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            progress = self.last_epoch / self.warmup_steps
            self.current_lrs = [base_lr * progress for base_lr in self.base_lrs]
        elif self.last_epoch == self.plateau1 or self.last_epoch == self.plateau2:
            self.current_lrs = [lr * self.decay_factor for lr in self.current_lrs]
        
        return self.current_lrs

scheduler = WarmupAndStep(optimizer=optimizer, warmup_steps= 2000, plateaus1= 12000, plateaus2=18000, decay_factor=0.316)

#--------------------------------------------------
# Training Function
#--------------------------------------------------


def train(model: nn.Module, n_epochs: int, optimizer, scheduler, dataloader, save_path=None, load_path=None):
    time_b = time.time()
    if load_path:
        checkpoint = torch.load(load_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint["params"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr
        if scheduler:
            scheduler.load_state_dict(checkpoint["scheduler"])
    else:
        model.apply(initialize_weights)

    model.train()

    val_losses=[]
    for epoch in range(n_epochs):
        running_loss = 0
        total_tokens = 0
        optimizer.zero_grad()

        for index, element in enumerate(dataloader):
            text = (element["input_ids"].squeeze(0)).to(device)
            label = element["labels"].squeeze(0).to(device)
            attention_mask = element["attention_mask"].to(device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                _, loss = model(x=text, targets=label, mask=attention_mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            if scheduler:
                scheduler.step()


            running_loss += loss.item() * text.numel()
            total_tokens += text.numel()
            
            optimizer.zero_grad()


            if (index) % 100 == 0:
                time_curr = time.time()
                avg_loss = running_loss / total_tokens
                perplexity = math.exp(avg_loss)
                if scheduler:
                    clr = scheduler.get_last_lr()[0]
                
                if save_path:
                    if scheduler:
                        checkpoint = {'params': model.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    'scheduler': scheduler.state_dict()}
                    else:
                        checkpoint = {'params': model.state_dict(),
                            'optimizer': optimizer.state_dict()}
                    torch.save(checkpoint, save_path)
                if scheduler:
                    print(f"Step {index}/{len(dataloader)} - Loss: {avg_loss:.4f} - Perplexity: {perplexity} - LR: {clr} - Time: {(time_curr - time_b):4f}s", flush=True)
                else:
                    print(f"Step {index}/{len(dataloader)} - Loss: {avg_loss:.4f} - Perplexity: {perplexity} - Time: {(time_curr - time_b):4f}s", flush=True)

        validation_loss = validate(model, n_epochs=1, dataloader=test_dataloader, load_path=save_path)
        val_losses.append(validation_loss)

        

        print('==========================================')
        print(f"Epoch {epoch}/{n_epochs} - Validation Loss: {validation_loss:.4f}")
        print(f'All validation losses: {val_losses}')
        print('==========================================')
        

#train(model=model, n_epochs=n_epochs, optimizer=optimizer , dataloader=train_dataloader, scheduler=None,
#      save_path=r"weights/normal1.pth")

        
        
#--------------------------------------------------
# Validation Function
#--------------------------------------------------

def validate(model: nn.Module, n_epochs: int, dataloader, device, load_path=None):
    model.eval()

    with torch.no_grad():
        if load_path:
            checkpoint = torch.load(load_path, weights_only=False, map_location=torch.device(device))
            new_state_dict = {}
            for key, value in checkpoint.items():
                new_key = key.replace("module.", "") 
                new_state_dict[new_key] = value
            model.load_state_dict(new_state_dict)

        for epoch in range(n_epochs):
            running_loss = 0
            running_correct = 0
            total_samples = 0

            for index, element in enumerate(dataloader):
                text = element["input_ids"].squeeze(0).to(device)
                label = element["labels"].squeeze(0).to(device)

                logits = model(x=text)

                loss = F.cross_entropy(logits.transpose(1, 2), label, ignore_index=tokenizer.pad_token_id)

                pred_tokens = torch.argmax(F.softmax(logits, dim=-1), dim=-1)

                correct = (pred_tokens == label).sum().item()
                running_correct += correct
                total_samples += label.numel() 

                running_loss += loss.item()


                if index % 2 == 0 and index > 0:
                    avg_loss = running_loss / index
                    avg_accu = (running_correct / total_samples) * 100
                    avg_perplexity = math.exp(avg_loss)
                    print(f'Step {index}/{len(dataloader)} - Loss: {avg_loss:.4f} - Accuracy: {avg_accu:.2f}% - Perplexity: {avg_perplexity:.2f}')

validate(model = model, n_epochs=1, dataloader=test_dataloader, load_path=r"weights/fast_distributed1.pth", device=device)
