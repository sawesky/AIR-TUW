# Team information

- Nemanja Saveski, 12333452
- Lucija Aleksić, 12202117

# Report
## Data aggregation
- The data is firstly loaded from the tsv file. 
- Then it is grouped by queryId and documentId.
- For each group, the weighted sum of relevance levels is calculated, where the weight is the duration used to judge the relevance level.
```python
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
```
- The data is then stored in a DataFrame with columns queryid, zero, documentid, and relevance-grade and saved.

- We have also made an alternative qrels weighting which incorporates log1p function (basically log(1 + x)). This way we wanted to penalize more the relevance of query-document pairs with lower values of "durationUsedToJudge", which comes intuitive: someone who graded some query-document pair very fast, probably hasn't even read it thoroughly, and in the worst case graded randomly.

```python
weight = np.log1p(row['durationUsedToJudgeMs'])
```

## Problems & Solutions

### Provided implementation of `forward` function in KNRM and TK classes

One big problem we've spent much time on (especially because we had to setup the environment on colab everytime we wanted to train something, ~40mins on average) was adapting our training loop to the implementation of KNRM and TK classes, specifically `forward` function. In starter pack, for both classes, it was given:

```python
    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        # shape: (batch, query_max)
        query_pad_oov_mask = (query["tokens"]["tokens"] > 0).float() # > 1 to also mask oov terms
        # shape: (batch, doc_max)
        document_pad_oov_mask = (document["tokens"]["tokens"] > 0).float()
        # shape: (batch, query_max,emb_dim)
        query_embeddings = self.word_embeddings(query)
        # shape: (batch, document_max,emb_dim)
        document_embeddings = self.word_embeddings(document)
```
When we tried to get data as: 

```python
        query = batch['query_tokens'].to(device)
        doc_pos = batch['doc_pos_tokens'].to(device)
        doc_neg = batch['doc_neg_tokens'].to(device)
```
We got plenty of errors in when running using these instances in training loop. Therefore, we changed definiton of `forward` function a little bit:

```python
    def forward(self, query: torch.Tensor, document: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        query_pad_oov_mask = (query > 0).float() # > 1 to also mask oov terms
        document_pad_oov_mask = (document > 0).float()

        query_embeddings = self.word_embeddings({'tokens': {'tokens': query}})
        document_embeddings = self.word_embeddings({'tokens': {'tokens': document}})
```


And we succeeded by getting data like this in our training loop:

```python
        query = batch['query_tokens']['tokens']['tokens'].to(device)
        doc_pos = batch['doc_pos_tokens']['tokens']['tokens'].to(device)
        doc_neg = batch['doc_neg_tokens']['tokens']['tokens'].to(device)
```

### TK class implementation
The TK class was not implemented correctly, so the evaluation results were not as expected. We didn't implement positional encoding and also we didn't branch tensors after kernel pooling and summing by document dimension into log and length normalization (we used same code as it is in KNRM for the part after calculating similarty matrix). 
The training loss for the first TK model version was:
```
loss
0.800668880221049
0.42502457576433816
0.30850102209568026
0.25835871711095176
```
And the validation results on MSMarco were:
``` python
results = {
    "MRR@10": 0.04,
    "Recall@10": 0.13,
    "QueriesWithNoRelevant@10": 1728.00,
    "QueriesWithRelevant@10": 272.00,
    "AverageRankGoldLabel@10": 5.55,
    "MedianRankGoldLabel@10": 5.00,
    "MRR@20": 0.05,
    "Recall@20": 0.27,
    "QueriesWithNoRelevant@20": 1440.00,
    "QueriesWithRelevant@20": 560.00,
    "AverageRankGoldLabel@20": 10.60,
    "MedianRankGoldLabel@20": 11.00,
    "MRR@1000": 0.06,
    "Recall@1000": 0.57,
    "QueriesWithNoRelevant@1000": 830.00,
    "QueriesWithRelevant@1000": 1170.00,
    "AverageRankGoldLabel@1000": 21.09,
    "MedianRankGoldLabel@1000": 21.00,
    "nDCG@3": 0.03,
    "nDCG@5": 0.04,
    "nDCG@10": 0.06,
    "nDCG@20": 0.09,
    "nDCG@1000": 0.16,
    "QueriesRanked": 2000.00,
    "MAP@1000": 0.06
}
```
### Extractive QA evaluation metrics issues 

We haven't read properly text of the assignment at the start and we got very bad results. When computing metrics, we had taken pairs that are not both in the result and the qrels.

- The evaluation results for top1 results were:
```python
Evaluation Metrics:
avg_exact_match: 0.002813367296506102
avg_f1: 0.006710919354558345
avg_jacard: 0.005610564971671238
```
- The evaluation results for top2 results were:
```python
Evaluation Metrics:
avg_exact_match: 0.00452419876059765
avg_f1: 0.01197783075883628
avg_jacard: 0.00994104023859013
```

### Google Colab issues
The training of the models was done on Google Colab.
Because of big data size there were some issues with the memory. Using only the free version of Google Colab the training was almost impossible.
Issues were lack of GPU and the RAM would run out.
At the start of doing exercise, we tried to somehow pack the environment so we don't need to wait ~40mins every time, but we didn't succeed in that.

## Evaluation results

### Early stopping implementation

- Early stopping was implemented based on MRR@10 metric (if, when validating, we get worse MRR@10 than best MRR@10 in two epochs in row, we stop training; in code it's defined as ```patience = 2``` right before starting training loop).
 - Just not to make any confusion, if we make improvement on best MMR@10, we save that model weights, and when we want to evaluate on test sets, we load weights from epoch when we achieved best performance, but not the weights from the latest epoch.

```python
    #early stopping based on mrr@10 values
    curr_epoch_mrr = val_metrics.get("MRR@10", 0)
    if curr_epoch_mrr > best_mrr:
        best_mrr = curr_epoch_mrr
        no_improvement_cnt = 0
        torch.save(model.state_dict(), config["model_path"])
    else:
        no_improvement_cnt = no_improvement_cnt + 1
        if no_improvement_cnt >= patience:
            print(f"Stopped early at epoch {epoch + 1}")
            break
```

- For both training KNRM and TK models, we used plain margin loss function ```nn.MarginRankingLoss(margin=1.0)```, Adam optimizer with learning rate 1e-3 when training KNRM and 4e-5 when training TK.
- Validation metrics through epochs are given in *results/val_metrics_knrm.csv* and *results/val_metrics_tk.csv*.

### KNRM class
The KNRM is implemented and trained on Google Colab.

The training loss was (there was early stopping at epoch 4/10):
``` 
loss
0.536101473
0.136926917
0.095192244
0.063868043
```

Evaluation metrics are:

| Evaluation      | MRR@10      | Recall@10     | AverageRankGoldLabel@10 | nDCG@3 | nDCG@10 |
|  ---            |     ----    |          ---  |    ---                  | ---    | ---     |
| MSMarco         | 0.216       | 0.447         | 3.649                   |  0.200 | 0.269   |
| FIRA base       | 0.926       | 0.951         | 1.162                   |  0.848 | 0.878   |
| FIRA custom     | 0.931       | 0.951         | 1.155                   |  0.848 | 0.883   |
| FIRA custom log | 0.936       | 0.950         | 1.141                   |  0.854 | 0.888   |

In the FIRA custom log, instead of regular weights we used ```weight = np.log1p(row[‘durationUsedToJudgeMs’])```.

As we can see, our log(1 + x) preformed best.


### TK class
In the new version of the TK class, we have implemented positional encoding and also added length normalization later in the model. 
``` python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
```

The training loss was (there was early stopping at epoch 6/10):

```
loss
0.190456537
0.059029584
0.046768919
0.039491684
0.033951011
0.029256552
```

The evaluation metrics are the following:
| Evaluation               | MRR@10    | Recall@10 | AverageRankGoldLabel@10 | nDCG@3   | nDCG@10 |
| ------------------------ | --------- | --------- | ----------------------- | -------- | ------- |
| MSMarco                  | 0.279     | 0.495     | 3.129                   | 0.270    | 0.329   |
| FIRA base                | 0.931     | 0.941     | 1.165                   | 0.843    | 0.875   |
| FIRA custom              | 0.936     | 0.940     | 1.148                   | 0.839    | 0.877   |
| FIRA custom log          | 0.940     | 0.941     | 1.143                   | 0.848    | 0.884   |

Again, weighting with log(1 + x) produced the best results.

### Extractive QA
- TK model was used to rank the documents for each query.
- We used both top1 and top2 results.
- The class is initialized with a pre-trained model from the HuggingFace model hub.
- A QA pipeline using the specified model and tokenizer is created.
- The model which we used was 'bert-large-uncased-whole-word-masking-finetuned-squad'.
- For each query-document pair the QA pipeline is used to extract the answer from the document that best matches the query.
- The answer is then stored in a DataFrame with columns queryid, zero, documentid, and answer and saved.
- An example of query, document and answer:
```
Query: access parallels cost
Document: Parallels Access makes using desktop programs touch-friendly (pictures) It's important to note that Parallels Access is a subscription-based service that costs $79.99 per year, which you will pay for in the App Store. But the problem is that it's $79.99 for each computer you access, so if I were a paying customer, the ability to access both my Mac laptop and my Windows desktop for this review would cost $159.98 per year.
Answer: $79.99 per year
```
- For metrics we used F1, exact match (from `core_metrics.py`) and implemented our own Jaccard similarity metric:
``` python
def compute_jaccard(gold_answer, pred_answer):
    gold_tokens = set(normalize_answer(gold_answer).split())
    pred_tokens = set(normalize_answer(pred_answer).split())
    intersection = gold_tokens.intersection(pred_tokens)
    union = gold_tokens.union(pred_tokens)
    if not union:
        return 0.0
    return len(intersection) / len(union)
```
- The evaluation results for top1 results were:
```
Evaluation Metrics:
avg_exact_match: 0.18129218900675023
avg_f1: 0.47209173808411203
avg_jacard: 0.39704471251310514
```
- The evaluation results for top2 results were:
```
Evaluation Metrics:
avg_exact_match: 0.17659574468085107
avg_f1: 0.4654543457738629
avg_jacard: 0.39192366083144536
```
As expected, we have better results for top1 than for top2 (there are more passages in top 2, but also more nonrelevant)
