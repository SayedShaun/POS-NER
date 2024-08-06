import pandas as pd

def load_clean_process(data_path:str, test_size:float=0.2):
    words, pos_tag , ner_tag = [], [], []
    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            parts = line.strip().split('\t')
            if len(parts)==3:
                s, p, n = line.split('\t')
                words.append(s), pos_tag.append(p), ner_tag.append(n)
    # Create DataFrame
    df = pd.DataFrame({
        'Word': words,
        'POS': pos_tag,
        'NER': ner_tag
    }).dropna().drop_duplicates().reset_index(drop=True)
    df = df.apply(lambda x: x.str.strip() if x.dtype=="object" else x)

    # Remove Punctuation
    punctuation = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
    df = df.apply(lambda x: x.str.strip(punctuation) if x.dtype=="object" else x)
    df = df.dropna().reset_index(drop=True)
    # Save DataFrame
    df.to_csv("Data/train_test_df.csv", index=False)

load_clean_process("Data/data.tsv", test_size=0.2)