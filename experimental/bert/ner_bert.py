
# Imports
import pandas as pd
import numpy as np
from tqdm import tqdm, trange

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import transformers
from transformers import BertForTokenClassification, AdamW

from transformers import get_linear_schedule_with_warmup

from seqeval.metrics import f1_score, accuracy_score

from tools import save_pkl, load_pkl

if True:
    model_size = 'base'
else:
    model_size = 'large'

should_train = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
print("Loading BERT tokeniser.")
tokenizer = BertTokenizer.from_pretrained('bert-'+model_size+'-cased', do_lower_case=False)
if should_train == True:
    # Load in previous dataset
    print("Loading in dataset. Please wait...")
    data = pd.read_csv(".data/ner_dataset.csv", encoding="latin1").fillna(method="ffill")
    data.tail(10)

    class SentenceGetter(object):
        def __init__(self, data):
            self.n_sent = 1
            self.data = data
            self.empty = False
            #print(data)
            # groups each sentence, and within that sentence, each word with it's POS and Tag)
            agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                            s["POS"].values.tolist(),
                                                            s["Tag"].values.tolist())]
            self.grouped = self.data.groupby("Sentence #").apply(agg_func)
            self.sentences = [s for s in self.grouped]
            #print(self.grouped)
        def get_next(self):
            try:
                s = self.grouped["Sentence: {}".format(self.n_sent)]
                self.n_sent += 1
                return s
            except:
                return None

    getter = SentenceGetter(data)

    sentences = [[word[0] for word in sentence] for sentence in getter.sentences]
    #print(sentences[0])

    labels = [[label[2] for label in sentence] for sentence in getter.sentences]
    #print(labels[0])

    print("Finding and sorting [TAG] values...")
    tag_values = sorted(list(set(data["Tag"].values)))
    tag_values.append("PAD")
    tag2idx = {t: i for i, t in enumerate(tag_values)}

    save_pkl(tag_values, './.data/tag_values.pkl')
    save_pkl(tag2idx, './.data/tag2idx.pkl')

    #print(tag_values)
    #print(tag_values)
    # Apply Bert
    # Prepare sentences and labels

    #print(torch.__version__)
#if should_train == True:
    MAX_LEN = 75
    bs = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    print("Loading BERT tokeniser.")
    tokenizer = BertTokenizer.from_pretrained('bert-'+model_size+'-cased', do_lower_case=False)

    # Now we tokenize all sentences. Since the BERT tokenizer is based a Wordpiece 
    # tokenizer it will split tokens in subword tokens. For example ‘gunships’ will 
    # be split in the two tokens ‘guns’ and ‘##hips’. We have to deal with the issue 
    # of splitting our token-level labels to related subtokens. In practice you would 
    # solve this by a specialized data structure based on label spans, but for 
    # simplicity I do it explicitly here.

    def tokenize_and_preserve_labels(sentence, text_labels):
        tokenized_sentence = []
        labels = []

        for word, label in zip(sentence, text_labels):

            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)

            # Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(tokenized_word)

            # Add the same label to the new list of labels `n_subwords` times
            labels.extend([label] * n_subwords)

        return tokenized_sentence, labels


    tokenized_texts_and_labels = [
        tokenize_and_preserve_labels(sent, labs)
        for sent, labs in tqdm(zip(sentences, labels),desc="Tokenising")
    ]

    # Splits things back up again - this time with byte piece
    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

    #print(tokenized_texts[9994])
    #print(labels[9994])
    #print(labels_1[9994])

    # Now pad - from keras
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                            maxlen=MAX_LEN, dtype="long", value=0.0,
                            truncating="post", padding="post")

    tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                        maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                        dtype="long", truncating="post")

    #print(tokenized_texts[9994])
    #print(tokenizer.convert_tokens_to_ids(tokenized_texts[9994]))
    #print(input_ids[9994])

#if should_train == True:
    ## attention mask stuff -  masks padding
    attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

    # Now split the dataset - sklearn
    # dont need to save masks for
    tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                                random_state=2018, test_size=0.1)
    tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                random_state=2018, test_size=0.1)


    tr_inputs = torch.tensor(tr_inputs)
    val_inputs = torch.tensor(val_inputs)
    tr_tags = torch.tensor(tr_tags)
    val_tags = torch.tensor(val_tags)
    tr_masks = torch.tensor(tr_masks)
    val_masks = torch.tensor(val_masks)


    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

    ################

    print("Loading BERT from transformers library version {}.".format(transformers.__version__))

    model = BertForTokenClassification.from_pretrained(
        'bert-'+model_size+'-cased',
        num_labels=len(tag2idx),
        output_attentions = False,
        output_hidden_states = False
    )

    model.to(device)

    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=3e-5,
        eps=1e-8
    )

    epochs = 4
    max_grad_norm = 1.0

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    print("Beginning training...")

    ## Store the average loss after each epoch so we can plot them.
    loss_values, validation_loss_values = [], []

    for _ in trange(epochs, desc="Epoch"):
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.

        # Put the model into training mode.
        model.train()
        # Reset the total loss for this epoch.
        total_loss = 0

        # Training loop
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training Step")):
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            # Always clear any previously calculated gradients before performing a backward pass.
            model.zero_grad()
            # forward pass
            # This will return the loss (rather than the model output)
            # because we have provided the `labels`.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
            # get the loss
            loss = outputs[0]
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # track train loss
            total_loss += loss.item()
            # Clip the norm of the gradient
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        print("Average train loss: {}".format(avg_train_loss))

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)


        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        # Put the model into evaluation mode
        model.eval()
        # Reset the validation loss for this epoch.
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels = [], []
        for batch in tqdm(valid_dataloader, desc="Validation Step"):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have not provided labels.
                outputs = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels)
            # Move logits and labels to CPU
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            eval_loss += outputs[0].mean().item()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)

        eval_loss = eval_loss / len(valid_dataloader)
        validation_loss_values.append(eval_loss)
        print("Validation loss: {}".format(eval_loss))
        pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                    for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
        valid_tags = [tag_values[l_i] for l in true_labels
                                    for l_i in l if tag_values[l_i] != "PAD"]
        print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
        print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
        print()

    model.save_pretrained('./data'+model_size+'/')
else:
    model = BertForTokenClassification.from_pretrained('./data'+model_size+'/')
    model.to(device)
    tag_values = load_pkl('./.data/tag_values.pkl')
    tag2idx = load_pkl('./.data/tag2idx.pkl')

test_sentence = """ Mr. Trump’s tweets began just moments after a Fox News report by Mike Tobin, a reporter for the network, about protests in Minnesota and elsewhere. """


def infer_entities(test_sentence):
    tokenized_sentence = tokenizer.encode(test_sentence)
    input_ids = torch.tensor([tokenized_sentence]).cuda()

    with torch.no_grad():
        output = model(input_ids)
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)

    # join bpe split tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices[0]):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(tag_values[label_idx])
            new_tokens.append(token)
    savetoken=' '
    savelabel=' '
    mydict = {"token": [], "label": []}
    for token, label in zip(new_tokens, new_labels):
        #print("{}\t{}".format(label, token))
        
        if (label[0] != 'I') & (savelabel != ' '):
            mydict["token"].append(savetoken)
            mydict["label"].append(savelabel[2:])
            savetoken=' '
            savelabel=' '
        if label[0] == 'B':
            savetoken = token
            savelabel = label    
        elif label[0] == 'I':
            savetoken = savetoken+' '+token
            savelabel = label
        
    #print(mydict)
    return(mydict)

    
import requests
import json
#from extract import json_extract
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
                    art = infer_entities(article['snippet'][:511])
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
        print("Pairs\n" + str(sorted(list(set(zip(art_ents, art_label, wordfreq))),key=take_second,reverse=True)))
        print("Found over "+str(c)+" articles.")



