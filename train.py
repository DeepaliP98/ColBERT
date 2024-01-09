import torch
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer

if __name__=='__main__':
    experiments = [
                   'train_triples.jsonl',
                #    'quant_train_triples.jsonl',
                #    'non_quant_train_triples.jsonl',
                #    'balanced/non_quant_train_triples_0.jsonl',
                #    'balanced/non_quant_train_triples_1.jsonl',
                #    'balanced/non_quant_train_triples_2.jsonl',
                #    'balanced/non_quant_train_triples_3.jsonl',
                #    'balanced/non_quant_train_triples_4.jsonl'
                # 'balanced-all/train_triples_0.jsonl',
                # 'balanced-all/train_triples_1.jsonl',
                # 'balanced-all/train_triples_2.jsonl',
                # 'balanced-all/train_triples_3.jsonl',
                # 'balanced-all/train_triples_4.jsonl'
                   ]
    
    for experiment in experiments:
            experiment_name = experiment.rstrip('.jsonl')
            with Run().context(RunConfig(nranks=1, experiment='colbert/bert-base-uncased/'+experiment_name)):

                config = ColBERTConfig(
                    bsize=4,
                    root="/home/venky/numerical-llm/experiments/",
                    checkpoint='roberta-large',
                    epochs=11
                )
                trainer = Trainer(
                    triples="datasets/scifact/colbert/"+experiment,
                    queries="datasets/scifact/colbert/queries.tsv",
                    collection="datasets/scifact/colbert/corpus.tsv",
                    config=config,
                )

                checkpoint_path = trainer.train(checkpoint='colbert-ir/colbertv2.0')

                print(f"Saved checkpoint to {checkpoint_path}...")