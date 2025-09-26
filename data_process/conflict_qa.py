

import json


def conflict(data_path, max_results=10):

    with open(data_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    
    QAs = []
    for item in dataset:
        
        
        QAs.append({
            "problem": item['question'],
            "passage": "Retrieved information:\n"+item['context']+"\nQusetion:\n",
            'answer': item['answer'],
            "document":item['context'],
            'is_cf' : item['is_cf']
        })
    return QAs
def conflict_mix(data_path, max_results=5):

    with open(data_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    
    QAs = []
    for item in dataset:
        
        
        QAs.append({
            "problem": item['question'],
            "passage": "Retrieved information:\n"+item['mixcontext']+"\nQusetion:\n",
            'answer': item['answer'],
            "document":item['mixcontext'],
            'is_cf' : item['is_cf']
        })
    return QAs
