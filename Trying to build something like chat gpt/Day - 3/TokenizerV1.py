import re
with open('Trying to build something like chat gpt/Day - 3/the-verdict.txt', 'r' , encoding='utf-8') as f:
    raw_text = f.read()
# print ("total number of characters in the book: ", len(raw_text))
# print (raw_text[:99])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
# print (preprocessed[:30])
# print (len(preprocessed))


all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
# print (vocab_size)

vocab = {token:integer for integer, token in enumerate(all_words)}
# for i , item in enumerate(vocab.items()):
#     print (item)
#     if i >= 50:
#         break

class TokenizerV1:
    def __init__(self, vocab) :
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self , text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)',text)

        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

tokenizer = TokenizerV1(vocab)

text = """"It's the last he painted, you know,"Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
# print(ids)
text = tokenizer.decode(ids)
# print(text)

text1 = "Hello, do you like tea?"
print(tokenizer.encode(text1))