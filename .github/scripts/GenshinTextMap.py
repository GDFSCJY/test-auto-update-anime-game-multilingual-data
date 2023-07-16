# kaggle environment
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
username = user_secrets.get_secret("GITHUB_USERNAME")
token = user_secrets.get_secret("GITHUB_TOKEN")


import os
import subprocess


def callsh(command):
    status = subprocess.run(command, shell=True)
    status.check_returncode()
    print(status.stdout)


# git config --global user.name 'github-actions[bot]'
callsh('git config --global user.name \'github-actions[bot]\'')
# git config --global user.email '114514+github-actions[bot]@noreply.github.com'
callsh('git config --global user.email \'114514+github-actions[bot]@noreply.github.com\'')
# git clone https://github.com/GDFSCJY/test-auto-update-anime-game-multilingual-data.git
callsh('git clone https://github.com/GDFSCJY/test-auto-update-anime-game-multilingual-data.git')
# cd update-anime-game-multilingual-data
os.chdir('test-auto-update-anime-game-multilingual-data')
# # git checkout -b github-action
# callsh('git checkout -b github-action')
# git submodule init GAMEDATA/GenshinData
callsh('git submodule init GAMEDATA/GenshinData')
# git submodule update GAMEDATA/GenshinData
callsh('git submodule update GAMEDATA/GenshinData')

# pip install sentence_transformers
callsh('pip install sentence_transformers')

from functools import partialmethod
from glob import glob
import json
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import torch
from transformers import MT5TokenizerFast
from sentence_transformers import SentenceTransformer

# disable tqdm output, enable it when debug
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

# load all en ja zh json file paths
en_text_map_path = 'GAMEDATA/GenshinData/TextMap/TextMapEN.json'
ja_text_map_path = 'GAMEDATA/GenshinData/TextMap/TextMapJP.json'
zh_text_map_path = 'GAMEDATA/GenshinData/TextMap/TextMapCHS.json'

# read json file and append to df
# json format: {index: text}
with open(en_text_map_path, 'r', encoding='utf-8') as f:
    en_text_map = pd.DataFrame(json.load(f).items(), columns=['index', 'en'])
with open(ja_text_map_path, 'r', encoding='utf-8') as f:
    ja_text_map = pd.DataFrame(json.load(f).items(), columns=['index', 'ja'])
with open(zh_text_map_path, 'r', encoding='utf-8') as f:
    zh_text_map = pd.DataFrame(json.load(f).items(), columns=['index', 'zh'])
# merge all df
df = pd.merge(en_text_map, ja_text_map, on='index', how='outer')
df = pd.merge(df, zh_text_map, on='index', how='outer')
# drop index column
df = df.drop(columns=['index'])

# drop duplicate, remove empty lines and nan
df = df.drop_duplicates(subset=['en', 'ja', 'zh'], keep='first')
df = df[df['en'] != '']
df = df[df['ja'] != '']
df = df[df['zh'] != '']
df = df.dropna()
# remove html tag
df['en'] = df['en'].apply(lambda x: re.sub(r'<[^>]*?>', '', str(x)))
df['ja'] = df['ja'].apply(lambda x: re.sub(r'<[^>]*?>', '', str(x)))
df['zh'] = df['zh'].apply(lambda x: re.sub(r'<[^>]*?>', '', str(x)))

# split text by \n
# check if number \n is the same for all languages
# if not, just keep it
drop_count = 0
for i, row in tqdm(enumerate(df.itertuples()), total=df.shape[0]):
    en, ja, zh = row.en, row.ja, row.zh
    if not r'\n' in en or not r'\n' in ja or not r'\n' in zh:
        continue
    en_split = en.split(r'\n')
    ja_split = ja.split(r'\n')
    zh_split = zh.split(r'\n')
    if len(en_split) == len(ja_split) == len(zh_split):
        # append to df
        df = pd.concat([df, pd.DataFrame({'en': en_split, 'ja': ja_split, 'zh': zh_split})], ignore_index=True)
        # drop
        df = df.drop(index=i)
        drop_count += 1
# remove \n
df['en'] = df['en'].apply(lambda x: re.sub(r'\\n', '', str(x)))
df['ja'] = df['ja'].apply(lambda x: re.sub(r'\\n', '', str(x)))
df['zh'] = df['zh'].apply(lambda x: re.sub(r'\\n', '', str(x)))
print(f'drop {drop_count} rows')

# remove row with only punctuation
df = df[~df['en'].str.match(r'^[^\w\s]+$')]
df = df[~df['ja'].str.match(r'^[^\w\s]+$')]
df = df[~df['zh'].str.match(r'^[^\w\s]+$')]
# remove lines with only numbers
df = df[~df['en'].str.match(r'^\d+$')]
df = df[~df['ja'].str.match(r'^\d+$')]
df = df[~df['zh'].str.match(r'^\d+$')]

# remove lines that tokens is more than 256 or less than 1
tokenizer = MT5TokenizerFast.from_pretrained('google/mt5-small')

df['en_len'] = df['en'].apply(lambda x: len(tokenizer.tokenize(x)))
df['ja_len'] = df['ja'].apply(lambda x: len(tokenizer.tokenize(x)))
df['zh_len'] = df['zh'].apply(lambda x: len(tokenizer.tokenize(x)))
df = df[df['en_len'] <= 256]
df = df[df['ja_len'] <= 256]
df = df[df['zh_len'] <= 256]
df = df[df['en_len'] >= 1]
df = df[df['ja_len'] >= 1]
df = df[df['zh_len'] >= 1]

# remove lines that LaBSE score is less than 0.6 or more than 0.99
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentenceTransformer('sentence-transformers/LaBSE').cuda()


_batch, _scores = [], []
_bs = 4
for i, row in tqdm(enumerate(df.itertuples()), total=df.shape[0]):
    inputs = [row.en, row.ja, row.zh]
    _batch.extend(inputs)
    if (i+1) % _bs == 0 or i == df.shape[0]-1:
        embeddings = model.encode(_batch)
        # calculate score between each pair
        for j in range(embeddings.shape[0]//3):
            _scores.append(np.average([
                np.matmul(embeddings[j*3], embeddings[j*3+1].T),
                np.matmul(embeddings[j*3], embeddings[j*3+2].T),
                np.matmul(embeddings[j*3+1], embeddings[j*3+2].T)
            ]))
        _batch = []
df = df.assign(score=_scores)
df = df[df['score'] >= 0.6]
df = df[df['score'] <= 0.99]

# replace「」to “”, 『』to ‘’ in zh
df['zh'] = df['zh'].apply(lambda x: x.replace('「', '“'))
df['zh'] = df['zh'].apply(lambda x: x.replace('」', '”'))
df['zh'] = df['zh'].apply(lambda x: x.replace('『', '‘'))
df['zh'] = df['zh'].apply(lambda x: x.replace('』', '’'))

# drop len and score column
df = df.drop(columns=['en_len', 'ja_len', 'zh_len', 'score'])

# save to parquet
df.to_parquet('parquet/GenshinTextMap.parquet', index=False)


# commit
import datetime
time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
callsh('git add parquet/GenshinTextMap.parquet')
callsh(f'git commit -m \"updated at {time_str}\"')

# push
callsh(f'git push https://{username}:{token}@github.com/GDFSCJY/test-auto-update-anime-game-multilingual-data.git')
