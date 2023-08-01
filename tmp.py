from transformers import BertTokenizer, AutoTokenizer

# import os
# print(os.listdir("./cache/xlnet-large-cased"))

tokenizer = AutoTokenizer.from_pretrained("./cache/xlnet-large-cased")
st = ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']
res1 = []
for word in st:
    res1.append(tokenizer(st,add_special_tokens=False))
cls_token = tokenizer.convert_tokens_to_ids([tokenizer.cls_token])[0]
sep_token = tokenizer.convert_tokens_to_ids([tokenizer.sep_token])[0]
res1 = [cls_token] + res1 + [sep_token]

all_tokens = []
for word in st:
    tokens = tokenizer.tokenize(word)
    all_tokens.extend(tokens)
all_tokens.insert(0,'[CLS]')
all_tokens.append('[SEP]')
res2 = tokenizer.convert_tokens_to_ids(all_tokens)

print(res1)
print(res2)
