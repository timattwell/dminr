import requests
import json
import nltk
from tqdm import tqdm 
from inferring import EntityClassifier
#from extract import json_extract

class SearchTask():
    def __init__(self, args, model, embeddings, tokenizer):
        nltk.download('punkt')
        self.ent_clas = EntityClassifier(args, model, embeddings, tokenizer)
        self.nyt_key = 'iFzGeWsfQAExVFhBG5ZtcckhVP0CAjmO'
        
    def recurrant_search(self):
        search_ents = {"token": [], "label": []}
        while True:
            srch_trm = input("Enter search term(with +'s): ")   
            if srch_trm == "q":
                break
            search_ents = self.search_funct(srch_trm)

            print(search_ents)

    def search_funct(self, srch_trm):
        srch = requests.get(
                    url="https://api.nytimes.com/svc/search/v2/articlesearch.json?q=" +  
                    srch_trm + 
                    "&api-key="+self.nyt_key
        )

        entities = {"token": [], "label": []}

        for article in srch.json()['response']['docs']:
            #print("Title: " + article["title"])
            #print("  URL: " + article["url"])
            #print("   ID: " + article["id"])

            sent = article
            art_list = nltk.tokenize.sent_tokenize(sent['snippet'])
            art_ents = {"token": [], "label": []}
            for sentence in tqdm(art_list):
                new_ents = self.ent_clas.infer_entities(sentence)
                art_ents["token"].append(new_ents["token"])
                art_ents["label"].append(new_ents["label"])
            entities["token"].append(art_ents["token"])
            entities["label"].append(art_ents["label"])
            print(" Ents: " + set(art_ents["token"]))

        return entities
'''
    cont = True
    while cont == True:
        q = input("What do you want to search for? ")
        if q == "q":
            cont = False
            print("Thank you for using Ner_bert.")
        else:
            c=0
            art_ents = []
            art_label = []
            for p in range(10):
                def get_url(q, page):
                    url = "https://api.nytimes.com/svc/search/v2/articlesearch.json?q="+q+"&api-key="+nyt_key
                    return url
                
                r = requests.get(get_url(q, str(p)))

                json_data = r.json()#['response']['docs']
                #print(json_data)

                with open('nyt.json','w') as outfile:
                    json.dump(json_data, outfile, indent=4)
                
                try:
                    for article in r.json()['response']['docs']:
                        art = ent_clas.infer_entities(article['snippet'][:511])
                        art_ents.extend(art["token"])
                        art_label.extend(art["label"])
                        c=c+1
                except:
                    print("Could not get data.")
                    print(r.json())

            wordfreq = []
            for w in art_ents:
                wordfreq.append(art_ents.count(w))

            def take_second(elem):
                return elem[1]
            pairs = sorted(list(set(zip(art_ents, wordfreq))),key=take_second,reverse=False)
            for pair in pairs:
                print(pair)

            print("Found over "+str(c)+" articles.")

'''
