import requests
import json
import inferring
#from extract import json_extract
def query(args, model, embeddings, tokenizer):
    cont = True
    while cont == True:
        q = input("What do you want to search for? ")
        if q == "q":
            cont = False
            print("Thank you for using Ner_bert.")
        else:
            nyt_key = 'iFzGeWsfQAExVFhBG5ZtcckhVP0CAjmO'#+'Y4eEsEg01aVjGURF'
            nyt_key_ = '9qVEPvGsY2GT0IIrndQp8LfCmOIZWvYW'
            #
            c=0
            art_ents = []
            art_label = []
            for p in range(10):
                def get_url(q, begin_date, end_date, page):
                    url = "https://api.nytimes.com/svc/search/v2/articlesearch.json?q="+q+"&begin_date="+begin_date+"&end_date="+end_date+"&page="+page+"&api-key="+nyt_key
                    return url
                
                r = requests.get(get_url(q, '20000101', '20200918',str(p)))

                json_data = r.json()#['response']['docs']
                #print(json_data)

                with open('nyt.json','w') as outfile:
                    json.dump(json_data, outfile, indent=4)
                
                try:
                    for article in r.json()['response']['docs']:
                        art = inferring.entity_classification(args, model, embeddings, tokenizer,article['snippet'][:511])
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