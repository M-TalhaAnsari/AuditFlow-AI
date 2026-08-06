
import json
with open('Evaluation/eval_set_draft.json') as f: 
    q = json.load(f)
for item in (q if isinstance(q, list) else sum(q.values(), [])):
    item.pop('ground_truth_document_id', None)
with open('Evaluation/eval_set_draft.json', 'w') as f: 
    json.dump(q, f, indent=2)
print('done')
