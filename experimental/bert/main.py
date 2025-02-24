from argparse import ArgumentParser
#import .tools
import training
import query
import torch
from flask import Flask, request, jsonify


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train", type=bool, default=False, help="True = Train, *False = Don't Train")
    parser.add_argument("--classify", type=bool, default=True, help="True = do classification task, *False = Don't do classification task")
    parser.add_argument("--model_size", type=str, default='large', help="""*large = BERT Large
                                                                          base = BERT base""")
    parser.add_argument("--epochs", type=int, default=4, help="Define number of training epochs (*4)")
    parser.add_argument("--bs", type=int, default=32, help="Define batch size (*32)")
    parser.add_argument("--training_data", type=str, default="./data/ner_dataset.csv", help="""Path to training data.
                                                                                             default=./.data/ner_dataset.csv""")
    parser.add_argument("--save_model", type=bool, default=True, help="True = Save model, *False = Save model")
    parser.add_argument("--nyt", type=bool, default=True, help="Use the New York Times API")
    
    args = parser.parse_args()
    args.max_len = 128
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train == True:
        model, embeddings, tokenizer = training.build_model(args)

    else:
        model, embeddings, tokenizer = training.load_model(args)

    if args.classify == True:
        
        
        if args.nyt == True:
            from query import SearchTask
        else:
            from login import SearchTask
        
        task = SearchTask(args, model, embeddings, tokenizer)
        
        app = Flask(__name__)
        @app.route("/predict",methods=['GET'])
        def predict():
            text = request.args.get('q')
            #print(type(text))
            try:
                task.search_funct(text)
                out = task.print_entities()

                return jsonify({"result":out})
            except Exception as e:
                print(e)
                return jsonify({"result":"Model Failed"})
        
        app.run('0.0.0.0',port=6969)
        #task.recurrant_search()
        
        #query.query(args, model, embeddings, tokenizer)

























