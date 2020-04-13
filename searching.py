import os
from pprint import pprint

from bert_serving.client import BertClient
from elasticsearch import Elasticsearch
from flask import Flask, jsonify, render_template, request

class Searcher:
    
    def __init__(self,search_size,index_name):
        self.search_size = search_size
        os.environ['INDEX_NAME'] = index_name #'jobsearch'
        #print(os.environ['INDEX_NAME'])
        self.index_name = os.environ['INDEX_NAME']

        self.bc = BertClient(ip='localhost', output_fmt='list')
        self.client = Elasticsearch('localhost:9200')
    
    def search(query):
        #query = input(">>> ")
        query_vector = bc.encode([query])[0]

        script_query = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, doc['text_vector']) + 1.0",
                    "params": {"query_vector": query_vector}
                }
            }
        }

        response = client.search(
            index=index_name,
            body={
                "size": search_size,
                "query": script_query,
                "_source": {"includes": ["sentenceId", "personaId", "text"]}
            }
        )
        #print(query)
        score = response['hits']['hits'][00]['_score']
        persona_block = response['hits']['hits'][00]['_source']
        persona_id =persona_block['personaId']
        script_query2 = {
            "term": {
                "personaId": str(persona_id)
            }
        }


        response2 = client.search(
            index=index_name,
            body={
                "query": script_query2,
                "_source": {"includes": ["sentenceId", "personaId", "text"]}
            }
        )
        #print(score) 
        return persona_block['text']
