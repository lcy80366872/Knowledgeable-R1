import json
def format_search_results(paragraphs, max_results=5):
    formatted = []
    for idx, p in enumerate(paragraphs, 1):
        if idx >=max_results+1: 
            break
        content = f"[webpage {idx} begin]\n"
        content += f"Title: {p.get('title', '')}\n"
        content += f"Content: {p.get('paragraph_text', '')}\n"
        content += f"[webpage {idx} end]"
        formatted.append(content)
    return "Retrieved information:\n" +"\n".join(formatted)+"\nQusetion:\n"

def musique(data_path,max_results=5): 

    QAs = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if 'question' in data and 'answer' in data and 'paragraphs' in data:
                QAs.append({
                    "problem": data['question'],
                    "passage": format_search_results(data['paragraphs'],max_results),  # paragraphs是list类型
                    "answer": data['answer'],
                    "document": data['paragraphs'][:max_results]
                })
    return QAs