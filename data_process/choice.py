
import pandas as pd
import json
from typing import List, Dict

def format_search_results(contexts: List[Dict], max_results: int) -> str:
    return "Retrieved information:\n"+"\n".join(
        f"[webpage {ctx['idx']+1} begin]\n{ctx['context']}\n[webpage {ctx['idx']+1} end]"
        for ctx in contexts[:max_results]
    ) +"\nQusetion:\n"

def choice(data_path: str, max_results: int = 5) -> List[Dict]:
    df = pd.read_excel(data_path)
    QAs = []
    
    for _, row in df.iterrows():
        question = row['query']
        options = eval(row['option']) if isinstance(row['option'], str) else row['option']
        option_text = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))

        contexts = []
        document= []
        for i in range(1, 6):
            context = row.get(f'top{i}', '')
            if pd.notna(context):
                contexts.append({
                    "idx": i-1,
                    "title": f"Result {i}",
                    "context": str(context)
                })
                document.append(str(context))
        QAs.append({
            "problem": f"{question}\n{option_text} Your answer should only be A or B or C or D or E.",
            "passage": format_search_results(contexts, max_results),
            "answer": row['answer'],
            "document":document
        })
    
    return QAs

def save_to_json(data: List[Dict], output_path: str):

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
