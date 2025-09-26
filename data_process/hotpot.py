import json
def format_search_results(paragraphs, max_results=5):

    formatted = []
    for idx, p in enumerate(paragraphs[:max_results], 1):

        title = p.get("title", "").replace("\n", " ").strip()
        context = p.get("context", "").replace("\n", " ").strip()

        entry = f"[webpage {idx} begin]\nTitle: {title}\nContent: {context}\n[webpage {idx} end]"
        formatted.append(entry)
    return "Retrieved information:\n" +"\n".join(formatted)+"\nQusetion:\n"

def hotpotqa(data_path,max_results=5): 
    with open(data_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    QAs = []
    for item in dataset:
        processed_context = []
        for idx, entry in enumerate(item['context']):
            if isinstance(entry, list) and len(entry) >= 2:
                title = str(entry[0])
                paragraphs = entry[1] if isinstance(entry[1], list) else []
            else:
                title, paragraphs = "Unknown", []
            
            # 合并段落
            context_text = " ".join(paragraphs).strip()
            
            processed_context.append({
                "idx": idx,
                "title": title,
                "context": context_text
            })
    
        document_list = []
        for doc in processed_context[:max_results]:
            document_list.append({
                "Title:"+doc["title"]+"\nContent:"+doc["context"]
            })
        
        QAs.append({
            "problem": item['question'],
            "passage": format_search_results(processed_context, max_results),
            "answer": item['answer'],
            "document": document_list  
        })
    return QAs
