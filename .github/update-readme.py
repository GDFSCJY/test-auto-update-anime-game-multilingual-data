import os
import pandas as pd
import subprocess
import datetime


def callsh(command):
    status = subprocess.run(command, shell=True)
    status.check_returncode()
    print(status.stdout)


# game names: parquets_file_name: len
game_names = {
    'Genshin Impact': {
        'GenshinReadable.parquet': 0,
        'GenshinSubtitle.parquet': 0,
        'GenshinTextMap.parquet': 0,
    },
    'Arknights': {
        'ArknightsStory.parquet': 0,
    }
}

readme_header = '''
# Anime Game Multilingual Data

Collection of multilingual language data from anime games stored in parquet format.

'''

readme_footer = '''
# How to load parquet file

```python
import pandas as pd
df = pd.read_parquet('*****.parquet')
```
you might need to install `pyarrow` first.

# Announcement

> Data is for personal use only. Please DO NOT use it for commercial purposes. 

> 小孩子不懂事，上传着玩的 :)
'''

readme_menu = ''
readme_sample = ''

for game_name in game_names:
    readme_sample += f'## {game_name}\n\n'
    samples = ''
    lang_set = set()
    for parquet_name in game_names[game_name]:
        parquet_path = os.path.join('parquet', parquet_name)
        df = pd.read_parquet(parquet_path)

        df_len = len(df)
        game_names[game_name][parquet_name] = df_len
        lang_set.update(df.columns)

        sample_df = pd.concat([df.head(5), df.tail(5)])

        samples += f'''
        ### {parquet_name}: {df_len} rows
        
        {sample_df.to_markdown()}

        '''

    readme_sample += f'### Languages: {", ".join(list(lang_set))}\n\n'
    readme_sample += samples

    readme_menu += f'- [{game_name}](#{game_name}): {sum([v for v in game_names[game_name].values()])}\n'

readme = readme_header + readme_menu + readme_sample + readme_footer

with open('README.md', 'r', encoding='utf-8') as f:
    f.write(readme)

# git add README.md
callsh('git add README.md')
# git commit -m 'update README.md'
callsh('git commit -m \'update README.md\'')
# git push
username = os.environ['MY_GITHUB_USERNAME']
token = os.environ['MY_GITHUB_TOKEN']
callsh(f'git push https://{username}:{token}@github.com/GDFSCJY/test-auto-update-anime-game-multilingual-data.git')
