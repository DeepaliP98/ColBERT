import json
import os
import tqdm


from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
import NumericalAspects
import evaluator


import jsonlines
import pandas as pd




if __name__ == '__main__':
    # experiments = {
                # 'scifact-all':'experiments/train_triple/none/2023-12/26/02.05.31/checkpoints/colbert',
    #             # 'scifact-quant':'experiments/quant_train_triple/none/2023-12/26/02.05.31/checkpoints/colbert',
    #             # 'scifact-non-quant-all':'experiments/non_quant_train_triple/none/2023-12/26/02.05.31/checkpoints/colbert',
    #             # 'scifact-balanced-non_quant_train_triples_0':'experiments/balanced/non_quant_train_triples_0/none/2023-12/25/23.59.12/checkpoints/colbert',
    #             # 'scifact-balanced-non_quant_train_triples_1':'experiments/balanced/non_quant_train_triples_1/none/2023-12/25/23.59.12/checkpoints/colbert',
    #             # 'scifact-balanced-non_quant_train_triples_2':'experiments/balanced/non_quant_train_triples_2/none/2023-12/25/23.59.12/checkpoints/colbert',
    #             # 'scifact-balanced-non_quant_train_triples_3':'experiments/balanced/non_quant_train_triples_3/none/2023-12/25/23.59.12/checkpoints/colbert',
    #             # 'scifact-balanced-non_quant_train_triples_4':'experiments/balanced/non_quant_train_triples_4/none/2023-12/25/23.59.12/checkpoints/colbert'
    #             # 'scifact-all-balanced-0':'experiments/balanced-all/train_triples_0/none/2023-12/27/16.45.13/checkpoints/colbert',
    #             # 'scifact-all-balanced-1':'experiments/balanced-all/train_triples_1/none/2023-12/27/16.45.13/checkpoints/colbert',
    #             # 'scifact-all-balanced-2':'experiments/balanced-all/train_triples_2/none/2023-12/27/16.45.13/checkpoints/colbert',
    #             # 'scifact-all-balanced-3':'experiments/balanced-all/train_triples_3/none/2023-12/27/16.45.13/checkpoints/colbert',
    #             # 'scifact-all-balanced-4':'experiments/balanced-all/train_triples_4/none/2023-12/27/16.45.13/checkpoints/colbert'
        
    #             # 'scifact-masked-all':'experiments/colbert-masked/train_triple/none/2023-12/27/17.47.37/checkpoints/colbert',
    #             # 'scifact-masked-quant':'experiments/colbert-masked/quant_train_triple/none/2023-12/27/17.47.37/checkpoints/colbert',
    #             # 'scifact-masked-non-quant-all':'experiments/colbert-masked/non_quant_train_triple/none/2023-12/27/17.47.37/checkpoints/colbert',
    #             # 'scifact-masked-balanced-non_quant_train_triples_0':'experiments/colbert-masked/balanced/non_quant_train_triples_0/none/2023-12/27/17.47.37/checkpoints/colbert',
    #             # 'scifact-masked-balanced-non_quant_train_triples_1':'experiments/colbert-masked/balanced/non_quant_train_triples_1/none/2023-12/27/17.47.37/checkpoints/colbert',
    #             # 'scifact-masked-balanced-non_quant_train_triples_2':'experiments/colbert-masked/balanced/non_quant_train_triples_2/none/2023-12/27/17.47.37/checkpoints/colbert',
    #             # 'scifact-masked-balanced-non_quant_train_triples_3':'experiments/colbert-masked/balanced/non_quant_train_triples_3/none/2023-12/27/17.47.37/checkpoints/colbert',
    #             # 'scifact-masked-balanced-non_quant_train_triples_4':'experiments/colbert-masked/balanced/non_quant_train_triples_4/none/2023-12/27/17.47.37/checkpoints/colbert'
    #             }
    # for experiment in experiments.keys():

    queries_df = pd.read_csv('datasets/scifact/colbert/queries_quant.tsv', sep='\t')
    queries = {}
    for index, row in queries_df.iterrows():
        queries[str(row[0])] = str(row[1])

    query_ids = list(queries.keys())
    query_texts = [queries[key] for key in query_ids]


    corpus_df = pd.read_csv('datasets/scifact/colbert/corpus.tsv', sep='\t')
    corpus = {}
    for index, row in corpus_df.iterrows():
        corpus[str(row[0])] = str(row[1])

    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[key] for key in corpus_ids]


    with open('datasets/scifact/colbert/test_qrels.json') as fp:
        test_qrels = json.load(fp)

    test_qrels = {key: value for key, value in test_qrels.items() if key in query_ids}  

    test_qrels_mod = {}
    for key,value in test_qrels.items():
        q_id = query_ids.index(key)
        test_qrels_mod[str(q_id)] = {}

        for c_id in value.keys():
            c_id_new = corpus_ids.index(c_id)
            test_qrels_mod[str(q_id)][str(c_id_new)]=1




    dataset = "scifact"
    datasplit = "test"
    # experiment = experiment

    nbits = 2
    doc_maxlen = 128
    experiment = "scifact_bert"

    index_name = f'{dataset}.{datasplit}.{nbits}bits'

    print(f'Filtered down to {len(test_qrels)} queries')

    # checkpoint = 'experiments/train_triple/none/2023-12/26/02.05.31/checkpoints/colbert'

    checkpoint = 'colbert-ir/colbertv2.0'
    index_name = 'scifact'



    with Run().context(RunConfig(nranks=1,rank=1, experiment=experiment)):
        config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits, kmeans_niters=4,bsize=4)

        indexer = Indexer(config=config,checkpoint=checkpoint)
        indexer.index(name=index_name, collection=corpus_texts, overwrite=True)

        indexer.get_index()


    with Run().context(RunConfig(experiment=experiment)):
        searcher = Searcher(index=index_name, collection=corpus_texts)


        result_qrels = {} 
        for idx,query in tqdm.tqdm(enumerate(query_texts)):
            result_qrels[str(idx)] = {}
            results = searcher.search(query, k=10)
            for passage_id, passage_rank, passage_score in zip(*results):
                result_qrels[str(idx)][str(passage_id)] = passage_score
        
        base_path = "ColBERT/results/scifact/"+'erroranal'

        import pandas as pd

        # Assuming test_qrels_mod, result_qrels, query_texts, and corpus_texts are already defined
        data = []

        for i in list(test_qrels_mod.keys())[:30]:
            entry = {"Query ID": i, "Query Text": query_texts[int(i)]}
            actual = list(test_qrels_mod[i].keys())[0]
            fetched = list(result_qrels[i].keys())[:10]
            
            if actual in fetched:                
                if actual == fetched[0]:
                    entry["Status"] = "Perfectly fetched"
                else:
                    entry["Status"] = "Correctly fetched"
                entry["Fetched Text"] = corpus_texts[int(fetched[0])]
            else:
                entry["Status"] = "Wrong fetched"
                entry["Fetched Text"] = corpus_texts[int(fetched[0])]
                entry["Actual Text"] = corpus_texts[int(actual)]
            
            data.append(entry)

        # Convert the data to a DataFrame
        df = pd.DataFrame(data)
        fp = open("./fetched_results.csv",'w+')
        fp.close()

        # Save to Excel
        df.to_csv("./fetched_results.csv", index=False)
                

        if not os.path.exists(base_path):
        # Create the directory
            os.makedirs(base_path)
            print(f"Directory '{base_path}' created")
        else:   
            print(f"Directory '{base_path}' already exists")
        
        out_file = open(base_path+"/test_qrels.json", "w+") 

        json.dump(test_qrels_mod, out_file)

        out_file = open(base_path+"/result_qrels.json", "w+") 
        
        json.dump(result_qrels, out_file)

        res = evaluator.evaluate(test_qrels_mod,result_qrels,k_values=[1,3,5,10])
        out_file = open(base_path+"/evaluation_results.json", "w+")
        json.dump(res,out_file)

