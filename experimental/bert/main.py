from argparse import ArgumentParser
#import .tools
import training
import torch

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train", type=bool, default=False, help="True = Train, *False = Don't Train")
    parser.add_argument("--classify", type=bool, default=True, help="True = do classification task, *False = Don't do classification task")
    parser.add_argument("--model_size", type=str, default='=large', help="""*large = BERT Large
                                                                          base = BERT base""")
    parser.add_argument("--epochs", type=int, default=4, help="Define number of training epochs (*4)")
    parser.add_argument("--batch", type=int, default=32, help="Define batch size (*32)")
    parser.add_argument("--training_data", type=str, default="./.data/ner_dataset.csv", help="""Path to training data.
                                                                                             default=./.data/ner_dataset.csv""")
    parser.add_argument("--save_model", type=bool, default=True, help="True = Save model, *False = Save model")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train == True:
        model, embeddings, tokenizer = training.build_model(args)

    else:
        model, embeddings, tokenizer = training.load_model(args)

    if args.classify == True:
        query.query(args, model, embeddings, tokenizer)

























