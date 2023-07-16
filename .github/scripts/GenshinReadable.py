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
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import torch
from transformers import MT5TokenizerFast
from sentence_transformers import SentenceTransformer

# disable tqdm output, enable it when debug
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

en_path = 'GAMEDATA/GenshinData/Readable/EN'
ja_path = 'GAMEDATA/GenshinData/Readable/JP'
zh_path = 'GAMEDATA/GenshinData/Readable/CHS'

skip_count = 0
en_files, ja_files, zh_files = [], [], []
for zh_file in tqdm(os.listdir(zh_path), total=len(os.listdir(zh_path))):
    en_file = os.path.join(en_path, zh_file.split('.')[0] + '_EN.txt')
    ja_file = os.path.join(ja_path, zh_file.split('.')[0] + '_JP.txt')
    zh_file = os.path.join(zh_path, zh_file)
    if not os.path.exists(en_file) or not os.path.exists(ja_file):
        skip_count += 1
        continue
    en_files.append(en_file)
    ja_files.append(ja_file)
    zh_files.append(zh_file)

print(f'Skip {skip_count} files, {len(en_files)} files left.')

df = pd.DataFrame(columns=['en', 'ja', 'zh'])


def read_file(file):
    text = []
    for line in open(file, 'r', encoding='utf-8'):
        line = line.strip()
        if line == '':
            continue
        text.append(line)
    return text


pbar = tqdm(zip(en_files, ja_files, zh_files), total=len(en_files))
for en_file, ja_file, zh_file in pbar:
    pbar.set_description(f'Processing {en_file}')

    en_text = read_file(en_file)
    ja_text = read_file(ja_file)
    zh_text = read_file(zh_file)

    if len(en_text) != len(ja_text) or len(en_text) != len(zh_text):
        print(f'Warning: length of {en_file} is not equal to {ja_file} or {zh_file}')
        print(f'length of en: {len(en_text)}, ja: {len(ja_text)}, zh: {len(zh_text)}')
        continue

    df = pd.concat([df, pd.DataFrame({'en': en_text, 'ja': ja_text, 'zh': zh_text})], ignore_index=True)

# drop duplicate
df = df.drop_duplicates(subset=['en', 'ja', 'zh'], keep='first')
# remove html tag
df['en'] = df['en'].apply(lambda x: re.sub(r'<[^>]*?>', '', x))
df['ja'] = df['ja'].apply(lambda x: re.sub(r'<[^>]*?>', '', x))
df['zh'] = df['zh'].apply(lambda x: re.sub(r'<[^>]*?>', '', x))
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
model = SentenceTransformer('sentence-transformers/LaBSE').to(device)

_batch, _scores = [], []
_bs = 8
for i, row in tqdm(enumerate(df.itertuples()), total=df.shape[0]):
    inputs = [row.en, row.ja, row.zh]
    _batch.extend(inputs)
    if (i + 1) % _bs == 0 or i == df.shape[0] - 1:
        embeddings = model.encode(_batch)
        # calculate score between each pair
        for j in range(embeddings.shape[0] // 3):
            _scores.append(np.average([
                np.matmul(embeddings[j * 3], embeddings[j * 3 + 1].T),
                np.matmul(embeddings[j * 3], embeddings[j * 3 + 2].T),
                np.matmul(embeddings[j * 3 + 1], embeddings[j * 3 + 2].T)
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
df.to_parquet('parquet/GenshinReadable.parquet', index=False)


# commit
import datetime
time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
callsh('git add parquet/GenshinReadable.parquet')
callsh(f'git commit -m \"updated at {time_str}\"')

# push
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
username = user_secrets.get_secret("GITHUB_USERNAME")
token = user_secrets.get_secret("GITHUB_TOKEN")
callsh(f'git push https://{username}:{token}@github.com/GDFSCJY/test-auto-update-anime-game-multilingual-data.git')
