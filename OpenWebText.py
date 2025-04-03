from torch.utils.data import Dataset
from datasets import load_dataset
from datasets import load_from_disk
import re

class OpenWebTextDataset(Dataset):
    def __init__(self, tokenizer, device, split, max_length, save_path=None, load_path=None):
        assert split in ["train", "validation", "test"], "Invalid split provided. Choose from 'train', 'validation', 'test'."
        assert max_length > 2, "max_length must be greater than 2."

        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

        if load_path == None:
           self.data = load_dataset("openwebtext", split=split, trust_remote_code=True)


           def tokenize_function(examples):
               return tokenizer(examples["text"], truncation=True)

           self.data = self.data.map(self.clean_data)
           self.tokenized_train = self.data.map(tokenize_function, batched=True, remove_columns=["text"])
           self.tokenized_train = self.group_texts(self.tokenized_train, self.max_length)
           if save_path:
               self.save_tokenized_dataset(self.tokenized_train, save_path)

        elif load_path:
           self.tokenized_train = self.load_tokenized_dataset(load_path)

    def __len__(self):
        return len(self.tokenized_train)

    def __getitem__(self, index):
        return self.tokenized_train[index]

    def group_texts(self, tokenized_dataset, block_size):
        block_size -= 2
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id
        def group_function(examples):
            concatenated = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated["input_ids"])
            
            total_length = (total_length // block_size) * block_size
            
            result = {}
            for k, t in concatenated.items():
                if k == "input_ids":
                    result[k] = [
                        [bos_token_id] + t[i : i + block_size] + [eos_token_id]
                        for i in range(0, total_length, block_size)
                    ]
                elif k == "attention_mask":
                    result[k] = [
                        [1] + t[i : i + block_size] + [1]
                        for i in range(0, total_length, block_size)
                    ]
                else:
                    result[k] = [
                        t[i : i + block_size]
                        for i in range(0, total_length, block_size)
                    ]
            return result

        return tokenized_dataset.map(group_function, batched=True)


    def clean_data(self, data):
        data = data["text"]
        data = re.sub(r'\(|\)', '', data)
        data = re.sub(r'<unk>', '', data)
        data = re.sub(r'=.*?=', '', data)
        data = re.sub(r'[^\w\s.-]', '', data)
        data = re.sub(r'-', '', data)
        return {"text": data.strip()}

    def save_tokenized_dataset(self, tokenized_train, path):
       print(f"Saving tokenized dataset to {path}...")
       tokenized_train.save_to_disk(f"{path}")
       print("Dataset saved successfully.")

    def load_tokenized_dataset(self, path):
       print(f"Loading tokenized dataset from {path}...")
       tokenized_train = load_from_disk(f"{path}")
       print("Dataset loaded successfully.")
       return tokenized_train