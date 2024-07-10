import pandas as pd
from transformers import pipeline

from core_metrics import compute_exact, compute_f1, compute_jaccard

class ExtractiveQA:
    def __init__(self, model_name):
        self.model_name = model_name
        self.qa_pipeline = pipeline('question-answering', model=model_name, tokenizer=model_name)
        self.missing_query_ids = []
        self.missing_doc_ids = []

    def load_top1_results(self, file_path):
        return pd.read_csv(file_path)

    def preprocess_tsv(self, file_path, num_fixed_fields):
        """
        Preprocess TSV file with variable-length text selection fields.
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('\t')
                fixed_fields = parts[:num_fixed_fields]
                text_selection = '\t'.join(parts[num_fixed_fields:]).lstrip('\t')
                data.append(fixed_fields + [text_selection])
        return data

    def load_fira_gold_labels(self, answers_path, tuples_path):
        answers_data = self.preprocess_tsv(answers_path, 3)
        tuples_data = self.preprocess_tsv(tuples_path, 5)
        
        answers_df = pd.DataFrame(answers_data, columns=['query_id', 'doc_id', 'relevance', 'answer'])
        tuples_df = pd.DataFrame(tuples_data, columns=['query_id', 'doc_id', 'relevance', 'query_text', 'document_text', 'answer'])
        
        return answers_df, tuples_df

    def run_extractive_qa(self, top1_results, fira_tuples):
        results = []
        total_rows = top1_results.shape[0]
        for i, row in top1_results.iterrows():
            if i % 100 == 0:
                print(f"Processing row {i} of {total_rows}")
            query_id = str(row['query_id'])
            doc_id = str(row['doc_id'])

            # check if the query_id is in the fira_tuples
            if query_id not in fira_tuples['query_id'].values:
                self.missing_query_ids.append(query_id)
                continue
            query_text = fira_tuples[fira_tuples["query_id"] == query_id]['query_text'].values[0]

            # check if the doc_id is in the fira_tuples
            if doc_id not in fira_tuples['doc_id'].values:
                self.missing_doc_ids.append(doc_id)
                continue
            document_text = fira_tuples[fira_tuples["doc_id"] == doc_id]['document_text'].values[0]

            qa_input = {
                'question': query_text,
                'context': document_text
            }
            qa_output = self.qa_pipeline(qa_input)

            answer = qa_output['answer']
            results.append((query_id, doc_id, answer))

        print(f"Missing query_ids: {len(self.missing_query_ids)}")
        print(f"Missing doc_ids: {len(self.missing_doc_ids)}")
        return pd.DataFrame(results, columns=['query_id', 'doc_id', 'answer'])
    
    def run_extractive_qa2(self, top2_results, fira_tuples):
        results = []
        total_rows = top2_results.shape[0]
        for i, row in top2_results.iterrows():
            if i % 100 == 0:
                print(f"Processing row {i} of {total_rows}")
            query_id = str(row['query_id'])
            doc1_id = str(row['doc1_id'])
            doc2_id = str(row['doc2_id'])

            # check if the query_id is in the fira_tuples
            if query_id not in fira_tuples['query_id'].values:
                self.missing_query_ids.append(query_id)
                continue
            query_text = fira_tuples[fira_tuples["query_id"] == query_id]['query_text'].values[0]

            # Process doc1
            if doc1_id in fira_tuples['doc_id'].values:
                document_text1 = fira_tuples[fira_tuples["doc_id"] == doc1_id]['document_text'].values[0]
                qa_input1 = {
                    'question': query_text,
                    'context': document_text1
                }
                qa_output1 = self.qa_pipeline(qa_input1)
                answer1 = qa_output1['answer']
                results.append((query_id, doc1_id, answer1))
            else:
                self.missing_doc_ids.append(doc1_id)

            # Process doc2
            if doc2_id in fira_tuples['doc_id'].values:
                document_text2 = fira_tuples[fira_tuples["doc_id"] == doc2_id]['document_text'].values[0]
                qa_input2 = {
                    'question': query_text,
                    'context': document_text2
                }
                qa_output2 = self.qa_pipeline(qa_input2)
                answer2 = qa_output2['answer']
                results.append((query_id, doc2_id, answer2))
            else:
                self.missing_doc_ids.append(doc2_id)

        print(f"Missing query_ids: {len(self.missing_query_ids)}")
        print(f"Missing doc_ids: {len(self.missing_doc_ids)}")
        return pd.DataFrame(results, columns=['query_id', 'doc_id', 'answer'])


    def save_results(self, results, output_path):
        results.to_csv(output_path, index=False)
        
    def evaluate_results(self, results, gold_answers_df):
        # Convert results to a dictionary
        results_dict = results.groupby('query_id').apply(lambda x: dict(zip(x['doc_id'], x['answer']))).to_dict()
        
        metrics = {'exact_match': [], 'f1': [], 'jacard': []}
        
        for _, row in gold_answers_df.iterrows():
            query_id = row['query_id']
            doc_id = row['doc_id']
            gold_answers = row['answer'].split('\t')

            if query_id in results_dict and doc_id in results_dict[query_id]:
                pred_answer = results_dict[query_id][doc_id]
                exact_match_scores = [compute_exact(gold_answer, pred_answer) for gold_answer in gold_answers]
                f1_scores = [compute_f1(gold_answer, pred_answer) for gold_answer in gold_answers]
                jaccard_scores = [compute_jaccard(gold_answer, pred_answer) for gold_answer in gold_answers]

                # print("Gold answers:", gold_answers)
                # print("Predicted answer:", pred_answer)
                # print("Max exact match score:", max(exact_match_scores))

                if max(jaccard_scores) < 0.15:
                    print("Gold answers:", gold_answers)
                    print("Predicted answer:", pred_answer)
                    print("Jaccard score:", max(jaccard_scores))
                    print("Exact match scores:", exact_match_scores)
                    print("F1 scores:", f1_scores)
                    print()

                metrics['exact_match'].append(max(exact_match_scores))
                metrics['f1'].append(max(f1_scores))
                metrics['jacard'].append(max(jaccard_scores))
        
        avg_exact_match = sum(metrics['exact_match']) / len(metrics['exact_match'])
        avg_f1 = sum(metrics['f1']) / len(metrics['f1'])
        avg_jacard = sum(metrics['jacard']) / len(metrics['jacard'])
        
        return {'avg_exact_match': avg_exact_match, 'avg_f1': avg_f1, 'avg_jacard': avg_jacard}

    def unrolled_to_ranked_result(self, unrolled_results):
        ranked_result = {}
        for query_id, group in unrolled_results.groupby('query_id'):
            ranked_result[query_id] = group.sort_values(by='doc_id')['doc_id'].tolist()
        return ranked_result