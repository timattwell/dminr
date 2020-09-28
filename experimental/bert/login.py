import requests
import json
import query
import nltk
from tqdm import tqdm 
from inferring import EntityClassifier

class SearchTask():
    def __init__(self, args, model, embeddings, tokenizer):
        nltk.download('punkt')
        self.ent_clas = EntityClassifier(args, model, embeddings, tokenizer)

        req = requests.post(url="http://localhost:8000/api/token/",
                    data=json.dumps({
                        "username": "tim",
                        "password": "qwertim"
                    }),
                    headers={"Content-Type": "application/json"})

        print(req)
        self.refresh = req.json()["refresh"]
        self.access = req.json()["access"]

#print(access)
    def recurrant_search(self):
        search_ents = {"token": [], "label": []}
        while True:
            srch_trm = input("Enter search term(with +'s): ")   
            if srch_trm == "q":
                break
            search_ents = self.search_funct(srch_trm)

            print(search_ents)



    def search_funct(self, srch_trm):
        srch = requests.get(url="http://localhost:8000/api/search/news?query=" +
                            srch_trm,
                            headers={"Authorization": "Bearer " + self.access})

        srch = requests.get(url="http://localhost:8000/api/search/news?query=" +
                            srch_trm,
                            headers={"Authorization": "Bearer " + self.access})
        entities = {"token": [], "label": []}

        for article in srch.json()['results']:
            print("Title: " + article["title"])
            print("  URL: " + article["url"])
            print("   ID: " + article["id"])
            res = requests.get(url="http://localhost:9200/_search?q=" +
                            article["id"])
            try:
                sent = str(res.json()['hits']['hits'][0]['_source']['body'])
                art_list = nltk.tokenize.sent_tokenize(sent)
                art_ents = {"token": [], "label": []}
                for sentence in tqdm(art_list):
                    new_ents = self.ent_clas.infer_entities(sentence)
                    art_ents["token"].append(new_ents["token"])
                    art_ents["label"].append(new_ents["label"])
                entities["token"].append(art_ents["token"])
                entities["label"].append(art_ents["label"])
                print(" Ents: " + list(set(art_ents["token"])))
            except:
                print(res)
            
        return entities

                