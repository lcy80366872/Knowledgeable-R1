import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict

def process_results_folder(root_folder):

    subfolders = [f for f in os.listdir(root_folder) 
                 if os.path.isdir(os.path.join(root_folder, f)) and not f.startswith('.')]

    dataset_results = defaultdict(lambda: defaultdict(list))

    dataset_is_cf = defaultdict(list)
    
    # 遍历每个子文件夹
    for method in subfolders:
        method_path = os.path.join(root_folder, method)
        
        if not os.path.exists(method_path):
            print(f"warning: file {method_path} not exists, skipped.")
            continue

        for filename in os.listdir(method_path):
            if filename.endswith('.json'):
                dataset_name = os.path.splitext(filename)[0]
                file_path = os.path.join(method_path, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    correct_values = []
                    is_cf_values = []
                    
                    for result in data.get('results', []):
                        correct_values.append(result.get('correct', False))

                        if 'is_cf' in result:
                            is_cf_values.append(result.get('is_cf', False))
                    
                    dataset_results[dataset_name][method] = correct_values

                    if not dataset_is_cf[dataset_name] and is_cf_values:
                        dataset_is_cf[dataset_name] = is_cf_values
                    
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {str(e)}")
    
    return dataset_results, dataset_is_cf, subfolders 

def create_summary_tables(dataset_results, dataset_is_cf):

    tables = {}
    
    for dataset_name, methods_data in dataset_results.items():

        max_samples = max(len(correct_list) for correct_list in methods_data.values())

        df_data = {}
        for method, correct_list in methods_data.items():

            padded_list = correct_list + [None] * (max_samples - len(correct_list))
            df_data[method] = padded_list

        if dataset_name in dataset_is_cf and dataset_is_cf[dataset_name]:
            is_cf_list = dataset_is_cf[dataset_name]
            padded_is_cf = is_cf_list + [None] * (max_samples - len(is_cf_list))
            df_data['is_cf'] = padded_is_cf
        
        df = pd.DataFrame(df_data)
        tables[dataset_name] = df
    
    return tables

def save_tables_to_excel(tables, output_file):

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for dataset_name, df in tables.items():

            sheet_name = dataset_name[:31] if len(dataset_name) > 31 else dataset_name
            df.to_excel(writer, sheet_name=sheet_name, index_label='样本索引')

def calculate_metrics_for_dataset(df, dataset_name, all_methods):

    has_is_cf = 'is_cf' in df.columns
    
\
    if not has_is_cf:
        if 'rag_prompting' in df.columns:
            df['is_cf'] = df['rag_prompting'].apply(lambda x: not bool(x))
            print(f"dataset {dataset_name}: using rag_prompting as the replace of is_cf")
        else:
            print(f"dataset {dataset_name}: no is_cf ")
            return None
    

    target_columns = [method for method in all_methods if method in df.columns]
    
\
    results = {
        'column': [],
        # based on  query-only
        'SCTI': [],  # TI子集
        'SCFI': [],  # FI子集
        
        # based on is_cf
        'ACC_Te': [],  # Te子集
        'ACC_Fe': [],  # Fe子集
        

        'ACC_TiTe': [],  # query-only=true & is_cf=true
        'ACC_TiFe': [],  # query-only=true & is_cf=false
        'ACC_FiFe': [],  # query-only=false & is_cf=false
        'ACC_FiTe': [],  # query-only=false & is_cf=true
        

        'ACC_TIuTe': [],
        
 
        'n_total': [],
        'n_TI': [],
        'n_FI': [],
        'n_Te': [],
        'n_Fe': [],
        'n_TiTe': [],
        'n_TiFe': [],
        'n_FiFe': [],
        'n_FiTe': [],
        'n_TIuTe': [],  
        

        'p_TI': [],
        'p_FI': [],
        'p_Te': [],
        'p_Fe': [],
        'p_TiTe': [],
        'p_TiFe': [],
        'p_FiFe': [],
        'p_FiTe': [],
        'p_TIuTe': []  
    }
    
    if 'query_only' not in df.columns:
        print(f"数据集 {dataset_name}: do not have query_only")
        return None

    n_total = len(df)
    

    subset_TI = df[df['query_only'] == True]    # TI子集
    subset_FI = df[df['query_only'] == False]   # FI子集

    subset_Te = df[df['is_cf'] == False]         # Te子集
    subset_Fe = df[df['is_cf'] == True]          # Fe子集

    subset_TiTe = df[(df['query_only'] == True) & (df['is_cf'] == False)]
    subset_TiFe = df[(df['query_only'] == True) & (df['is_cf'] == True)]
    subset_FiFe = df[(df['query_only'] == False) & (df['is_cf'] == True)]
    subset_FiTe = df[(df['query_only'] == False) & (df['is_cf'] == False)]
    

    subset_TIuTe = df[(df['query_only'] == True) | (df['is_cf'] == False)]
    
 
    n_TI = len(subset_TI)
    n_FI = len(subset_FI)
    n_Te = len(subset_Te)
    n_Fe = len(subset_Fe)
    n_TiTe = len(subset_TiTe)
    n_TiFe = len(subset_TiFe)
    n_FiFe = len(subset_FiFe)
    n_FiTe = len(subset_FiTe)
    n_TIuTe = len(subset_TIuTe)  
    

    p_TI = n_TI / n_total if n_total > 0 else 0
    p_FI = n_FI / n_total if n_total > 0 else 0
    p_Te = n_Te / n_total if n_total > 0 else 0
    p_Fe = n_Fe / n_total if n_total > 0 else 0
    p_TiTe = n_TiTe / n_total if n_total > 0 else 0
    p_TiFe = n_TiFe / n_total if n_total > 0 else 0
    p_FiFe = n_FiFe / n_total if n_total > 0 else 0
    p_FiTe = n_FiTe / n_total if n_total > 0 else 0
    p_TIuTe = n_TIuTe / n_total if n_total > 0 else 0  

    for col in target_columns:
        if col not in df.columns:
            print(f"dataset {dataset_name}: don't have col {col}, skip ")
            continue
            

        scti = subset_TI[col].mean() if len(subset_TI) > 0 else np.nan
        scfi = subset_FI[col].mean() if len(subset_FI) > 0 else np.nan
        

        acc_te = subset_Te[col].mean() if len(subset_Te) > 0 else np.nan
        acc_fe = subset_Fe[col].mean() if len(subset_Fe) > 0 else np.nan
        
  
        acc_tite = subset_TiTe[col].mean() if len(subset_TiTe) > 0 else np.nan
        acc_tife = subset_TiFe[col].mean() if len(subset_TiFe) > 0 else np.nan
        acc_fife = subset_FiFe[col].mean() if len(subset_FiFe) > 0 else np.nan
        acc_fite = subset_FiTe[col].mean() if len(subset_FiTe) > 0 else np.nan
        
   
        acc_tiute = subset_TIuTe[col].mean() if len(subset_TIuTe) > 0 else np.nan
        
 
        results['column'].append(col)
        results['SCTI'].append(scti)
        results['SCFI'].append(scfi)
        results['ACC_Te'].append(acc_te)
        results['ACC_Fe'].append(acc_fe)
        results['ACC_TiTe'].append(acc_tite)
        results['ACC_TiFe'].append(acc_tife)
        results['ACC_FiFe'].append(acc_fife)
        results['ACC_FiTe'].append(acc_fite)
        results['ACC_TIuTe'].append(acc_tiute) 
        
   
        results['n_total'].append(n_total)
        results['n_TI'].append(n_TI)
        results['n_FI'].append(n_FI)
        results['n_Te'].append(n_Te)
        results['n_Fe'].append(n_Fe)
        results['n_TiTe'].append(n_TiTe)
        results['n_TiFe'].append(n_TiFe)
        results['n_FiFe'].append(n_FiFe)
        results['n_FiTe'].append(n_FiTe)
        results['n_TIuTe'].append(n_TIuTe)  
        
        results['p_TI'].append(p_TI)
        results['p_FI'].append(p_FI)
        results['p_Te'].append(p_Te)
        results['p_Fe'].append(p_Fe)
        results['p_TiTe'].append(p_TiTe)
        results['p_TiFe'].append(p_TiFe)
        results['p_FiFe'].append(p_FiFe)
        results['p_FiTe'].append(p_FiTe)
        results['p_TIuTe'].append(p_TIuTe)  

    results_df = pd.DataFrame(results)
    results_df['dataset'] = dataset_name
    
    return results_df

def main():
    root_folder = 'results'  
    output_file = 'results_summary.xlsx'
    metrics_output_file = 'all_accuracy_metrics.csv'
    

    dataset_results, dataset_is_cf, all_methods = process_results_folder(root_folder)
    
    if not dataset_results:
        
        return
    

    tables = create_summary_tables(dataset_results, dataset_is_cf)
    

    save_tables_to_excel(tables, output_file)
    print(f"all results saved at {output_file}")
    

    all_metrics = []
    for dataset_name, df in tables.items():
        metrics_df = calculate_metrics_for_dataset(df, dataset_name, all_methods)
        if metrics_df is not None:
            all_metrics.append(metrics_df)
    
   
    if all_metrics:
        combined_metrics = pd.concat(all_metrics, ignore_index=True)
        combined_metrics.to_csv(metrics_output_file, index=False)
        print(f"saved at {metrics_output_file}")
        

        for dataset_name in tables.keys():
            dataset_metrics = combined_metrics[combined_metrics['dataset'] == dataset_name]
            if not dataset_metrics.empty:
                print(f"\ndataset: {dataset_name}")
                print(f"  TiTe: {dataset_metrics['n_TiTe'].iloc[0]} ({dataset_metrics['p_TiTe'].iloc[0]*100:.1f}%)")
                print(f"  TiFe: {dataset_metrics['n_TiFe'].iloc[0]} ({dataset_metrics['p_TiFe'].iloc[0]*100:.1f}%)")
                print(f"  FiFe: {dataset_metrics['n_FiFe'].iloc[0]} ({dataset_metrics['p_FiFe'].iloc[0]*100:.1f}%)")
                print(f"  FiTe: {dataset_metrics['n_FiTe'].iloc[0]} ({dataset_metrics['p_FiTe'].iloc[0]*100:.1f}%)")
                print(f"  TIuTe: {dataset_metrics['n_TIuTe'].iloc[0]} ({dataset_metrics['p_TIuTe'].iloc[0]*100:.1f}%)")
                
                for _, row in dataset_metrics.iterrows():
                    print(f"\n  method: {row['column']}")
                    print(f"    SCTI: {row['SCTI']:.4f}, SCFI: {row['SCFI']:.4f}")
                    print(f"    ACC_Te: {row['ACC_Te']:.4f}, ACC_Fe: {row['ACC_Fe']:.4f}")
                    print(f"    TiTe: {row['ACC_TiTe']:.4f}, TiFe: {row['ACC_TiFe']:.4f}")
                    print(f"    FiFe: {row['ACC_FiFe']:.4f}, FiTe: {row['ACC_FiTe']:.4f}")
                    print(f"    TIuTe: {row['ACC_TIuTe']:.4f}")  # 新增: 打印TIuTe指标
    else:
        print("no metric")

if __name__ == "__main__":
    main()