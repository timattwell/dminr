from argparse import ArgumentParser
import .tools

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train", type=bool, default=False, help="True = Train, *False = Don't Train")
    parser.add_argument("--model_size", type=str, default='base', help="""large = BERT Large
                                                                          *base = BERT base""")
    parser.add_argument("--epochs", type=int, default=4, help="Define number of training epochs (*4)")
    parser.add_argument("--batch", type=int, defalt=32, help="Define batch size (*32)")

    args = parser.parse_args()

























