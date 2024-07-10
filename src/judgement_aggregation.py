import pandas as pd
import numpy as np
from collections import defaultdict

def load_data(file_path):
    return pd.read_csv(file_path, delimiter='\t')

def aggregate_judgments(data):
    aggregated = defaultdict(list)

    grouped = data.groupby(['queryId', 'documentId'])
    
    for (queryId, documentId), group in grouped:
        weighted_sum = 0
        total_weight = 0
        
        for _, row in group.iterrows():
            relevance = int(row['relevanceLevel'].split('_')[0])
            weight = row['durationUsedToJudgeMs']
            weighted_sum += relevance * weight
            total_weight += weight
        
        aggregated_relevance = weighted_sum / total_weight if total_weight > 0 else 0
        aggregated_relevance = round(aggregated_relevance)
        
        aggregated[(queryId, documentId)] = aggregated_relevance
    
    aggregated_df = pd.DataFrame([
        {'queryid': k[0], 'zero' : 0, 'documentid': k[1], 'relevance-grade': v}\
        for k, v in aggregated.items()
    ])
    
    return aggregated_df

def save_aggregated_data(aggregated_data, output_path):
    aggregated_data.to_csv(output_path, sep='\t', index=False, header=False)

if __name__ == "__main__":
    input_path = './data/Part-1/fira-22.judgements-anonymized.tsv'
    output_path = './data/Part-1/aggregated_qrels.tsv'
    
    raw_data = load_data(input_path)
    aggregated_data = aggregate_judgments(raw_data)
    save_aggregated_data(aggregated_data, output_path)