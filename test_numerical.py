import json
import jsonlines
from tqdm import tqdm
from NumericalAspects import NumericalAspects
import evaluator


query_map = {}
with jsonlines.open('datasets/scifact/queries.jsonl') as f:
    for line in f.iter():
        query_map[line["_id"]]=line["text"]

num_aspects = NumericalAspects()

numerical_queries = []

for key in tqdm(query_map.keys(),total = len(query_map.keys())):
    if(num_aspects.is_numerical_claim(query_map[key])):
        numerical_queries.append(key)


base_path = "results/fail-fast-tests/scifact/"

out_file = open(base_path+"qrel_mod.json", "r") 

qrel = json.load(out_file)

out_file = open(base_path+"zero_shot_all.json", "r") 

result_qrels = json.load(out_file)

numerical_qrels = dict(filter(lambda item: item[0] in numerical_queries, result_qrels.items()))

res = evaluator.evaluate(qrel,numerical_qrels,k_values=[1,3,5,10])
print(res)

