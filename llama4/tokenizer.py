import re 
from pathlib import Path 
from collections import Counter, defaultdict
from tqdm.auto import tqdm 

from datasets import load_dataset #using a pre-existing one instead of making my own data corpus 

DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
SPLIT = "train"
NUM_MERGES = 10000  # this is for number of BPE merge operations
VOCAB_SIZE = 30000  # stopping early when the vocab reaches this much size 

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def load_wikitext(split: str):

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)
    for sample in tqdm(dataset, desc="Streaming lines"):
        text = sample.get('text', '').strip() 
        if not text:
            continue 

        text = re.sub(r"\s+", " ", text).lower() 
        for word in text.split():
            #appending the end of word token
            chars = list(word) + ['</w>']
            yield tuple(chars)

# word freq counter 

print("Loading and tokenizing WikiText-2.....")
word_freq = Counter()
for word in load_wikitext(SPLIT):
    word_freq[word] += 1 

print(f"Total unique word types: {len(word_freq)}")

# initialization of vocabulary 
vocab = set(sym for word in word_freq for sym in word)
print(f"Initial vocab size (symbols inclduing </w>): {len(vocab)}")

# BPE training utils 

def get_pair_counts(word_freq: Counter) -> Counter: 
    # count the adjacent symbol pair frqs in word frq map 
    pairs = Counter()
    for word, freq in word_freq.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    
    return pairs 

def compile_merge_pattern(pair): 
    a, b = map(re.escape, pair)
    return re.compile(fr"(?<!\S){a}\s+{b}(?!\S)")


def merge_pair_in_vocab(word_freq: Counter, pair: tuple, merged_symbol: str) -> Counter:
    pattern = compile_merge_pattern(pair)
    new_freq = Counter()
    for word, freq in word_freq.items():
        token_str = " ".join(word)
        merged_str = pattern.sub(merged_symbol, token_str)
        new_word = tuple(merged_str.split())
        new_freq[new_word] += freq
    return new_freq

#execute the bpe merges 

merges = []
for i in tqdm(range(NUM_MERGES), desc="BPE merges"):
    pair_counts = get_pair_counts(word_freq)
    if not pair_counts:
        break
    best_pair, best_count = pair_counts.most_common(1)[0]
    merged_symbol = ''.join(best_pair)
    merges.append((best_pair, merged_symbol))

    # Update frequencies & vocab
    word_freq = merge_pair_in_vocab(word_freq, best_pair, merged_symbol)
    vocab.add(merged_symbol)

    # Early stop
    if len(vocab) >= VOCAB_SIZE:
        print(f"Reached target vocab size of {VOCAB_SIZE}. Stopping.")
        break

print(f"Total merges performed: {len(merges)}")
print(f"Final vocab size: {len(vocab)}")


with open(RESULTS_DIR / "vocab.txt", "w", encoding="utf-8") as vf:
    for token in sorted(vocab):
        vf.write(token + "\n")

with open(RESULTS_DIR / "merges.txt", "w", encoding="utf-8") as mf:
    for pair, merged in merges:
        mf.write(f"{pair[0]} {pair[1]} -> {merged}\n")

print("Saved vocab.txt and merges.txt in results/")
