import jax
import jax.numpy as jnp

import grain.python as grain

import tiktoken
from pathlib import Path
from helper import load_stories_from_file

file_path = Path("TinyStories-1000.txt")

with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
    data = f.read()
    stories = data.split('<|endoftext|>')

    print("First story (300 chars):\n")
    story = stories[0]
    print(story.strip()[:300], "...")

    print(f"\nTotal number of stories: {len(stories) - 1:,}")

tokenizer = tiktoken.get_encoding("gpt2")

print(f"Vocabulary size: {tokenizer.n_vocab:,}")
print(f"Special tokens: {tokenizer.special_tokens_set}")

class StoryDataset:

    def __init__(self, stories, maxlen, tokenizer):
        self.stories = stories
        self.maxlen = maxlen
        self.tokenizer = tokenizer
        self.end_token = tokenizer.encode('<|endoftext|>', \
                        allowed_special={'<|endoftext|>'})[0]

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        story = self.stories[idx]
        tokens = self.tokenizer.encode(story,
                                       allowed_special={'<|endoftext|>'})

        if len(tokens) > self.maxlen:
            tokens = tokens[:self.maxlen]

        tokens.extend([0] * (self.maxlen - len(tokens)))
        return tokens

shuffled_sampler = grain.IndexSampler(
    num_records=10,
    shuffle=True,
    seed=42,
    shard_options=grain.NoSharding(),
    num_epochs=1
)

def print_sampler_example(sampler, name):
    print(f"\n{name}")
    for i, idx in enumerate(sampler):
        print(f"Record {i}: {idx}")

print_sampler_example(shuffled_sampler, "Shuffled sampler")

batch_op_keep = grain.Batch(
    batch_size=32,
    drop_remainder=False
)

def create_dataloader(
    stories,
    tokenizer,
    maxlen,
    batch_size,
    shuffle = False,
    num_epochs = 1,
    seed = 42,
    worker_count = 0
):
    dataset = StoryDataset(stories, maxlen, tokenizer)
    estimated_batches = len(dataset) // batch_size

    sampler = grain.IndexSampler(
        num_records=len(dataset), # 1,000 stories for this dataset
        shuffle=shuffle,
        seed=seed,
        shard_options=grain.NoSharding(),
        num_epochs=num_epochs
    )
    dataloader = grain.DataLoader(
        data_source=dataset,
        sampler=sampler,
        operations=[
            grain.Batch(batch_size=batch_size, drop_remainder=True)
        ],
        worker_count=worker_count
    )

    return dataloader, estimated_batches

stories = load_stories_from_file(
    "TinyStories-1000.txt",
    max_stories=100
)

stories[0]

dataloader, batches_per_epoch = create_dataloader(
    stories=stories,
    tokenizer=tokenizer,
    maxlen=128,
    batch_size=32,
    shuffle=False,
    num_epochs=1,
    seed=42,
    worker_count=0  # Single process for experimentation
)

print(f"\nDataLoader created successfully:")
print(f"Will produce {batches_per_epoch} batches per epoch")

next(iter(dataloader))