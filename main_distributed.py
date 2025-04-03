import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import _LRScheduler

import math
import time

from model import TheTransformer
from OpenWebText import OpenWebTextDataset

from transformers import DataCollatorForLanguageModeling, AutoTokenizer

import torch.distributed as dist
import os

#--------------------------------------------------
# Data Preparation
#--------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()
print(f"Using device: {device}, Number of GPUs: {n_gpus}")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

vocab_size = len(tokenizer)


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


def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank

def cleanup_ddp():
    dist.destroy_grocess_group()

#--------------------------------------------------
# Main Training Function
#--------------------------------------------------


def main():
    rank, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    vocab_size = len(tokenizer)

    dataset = OpenWebTextDataset(tokenizer=tokenizer, device=device, split='train', max_length=512, load_path=r"../Transformerv2/data/openweb/train")

    data_collator = ShiftedDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_sampler = DistributedSampler(dataset)

    train_dataloader = DataLoader(dataset, batch_size=64, sampler=train_sampler, collate_fn=data_collator)
    
    N_HEADS = 16
    D_MODEL = 1280
    DROPOUT = 0.1
    N_BLOCKS = 12
    LATENT_DIM = 512

    model = TheTransformer(vocab_size=vocab_size, num_heads=N_HEADS, n_layers=N_BLOCKS, d_model=D_MODEL, latent_dim=LATENT_DIM,
                        ignore_index=tokenizer.pad_token_id, dropout=DROPOUT).to(device)

    scaler = torch.amp.GradScaler(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
     
    checkpoint = torch.load(r"weights/fast_distributed1.pth", weights_only=True, map_location=torch.device(device))
    model.load_state_dict(checkpoint)

    for param in model.parameters():
        param.requires_grad = True

    print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

    base_lr = 2.4e-4

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01, betas=(0.9, 0.95))

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

    n_epochs = 5
    
    t1=time.time()
    
    for epoch in range(n_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()
        
        running_loss = 0
        total_tokens = 0

        for index, element in enumerate(train_dataloader):
            text = (element["input_ids"].squeeze(0)).to(device)
            label = element["labels"].squeeze(0).to(device)
            attention_mask = element["attention_mask"].squeeze(0).to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                _, loss = model(x=text, targets=label, mask=attention_mask)
            
            running_loss += loss.item() * text.numel()
            total_tokens += text.numel()
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()


            if rank == 0 and index % 100 == 0:
                avg_loss = running_loss / total_tokens
                perplexity = math.exp(avg_loss)
                t2 = time.time()
                clr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch}, Step {index}/{len(train_dataloader)}, LR: {clr}, Loss: {avg_loss}, Perplexity: {perplexity} - time elapsed: {(t2-t1)/60}")
                torch.save(model.state_dict(), "weights/fast_distributed1.pth")

    cleanup_ddp()

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
