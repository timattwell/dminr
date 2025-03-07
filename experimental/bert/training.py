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

class DataImporter():
    def __init__(self, args):
        self.data_path = args.training_data
        self.data = pd.read_csv(self.data_path, encoding="latin1").fillna(method="ffill")
        self.data.tail(10)
        self.args = args
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
        
        getter = SentenceGetter(self.data)

        self.sentences = [[word[0] for word in sentence] for sentence in getter.sentences]
        #print(sentences[0])

        self.labels = [[label[2] for label in sentence] for sentence in getter.sentences]
        #print(labels[0])

        print("Finding and sorting [TAG] values...")
        self.tag_values = sorted(list(set(self.data["Tag"].values)))
        self.tag_values.append("PAD")
        self.tag2idx = {t: i for i, t in enumerate(self.tag_values)}

        if args.save_model == True:
            self.save_data()

    def save_data(self):
        save_pkl(self.tag_values, './data/tag_values.pkl')
        save_pkl(self.tag2idx, './data/tag2idx.pkl')

    def tokenise(self, tokenizer):
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
            for sent, labs in tqdm(zip(self.sentences, self.labels),desc="Tokenising")
        ]

        # Splits things back up again - this time with byte piece
        tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
        labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]
        MAX_LEN = self.args.max_len
        # Now pad - from keras
        self.input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                                maxlen=MAX_LEN, dtype="long", value=0.0,
                                truncating="post", padding="post")

        self.tags = pad_sequences([[self.tag2idx.get(l) for l in lab] for lab in labels],
                            maxlen=MAX_LEN, value=self.tag2idx["PAD"], padding="post",
                            dtype="long", truncating="post")

class ModelConfig():
    def __init__(self, args, data):
        ## attention mask stuff -  masks padding
        self.attention_masks = [[float(i != 0.0) for i in ii] for ii in data.input_ids]
        self.args = args
        # Now split the dataset - sklearn
        # dont need to save masks for
        tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(data.input_ids, data.tags,
                                                                    random_state=2018, test_size=0.1)
        tr_masks, val_masks, _, _ = train_test_split(self.attention_masks, data.input_ids,
                                                    random_state=2018, test_size=0.1)

        tr_inputs = torch.tensor(tr_inputs)
        val_inputs = torch.tensor(val_inputs)
        tr_tags = torch.tensor(tr_tags)
        val_tags = torch.tensor(val_tags)
        tr_masks = torch.tensor(tr_masks)
        val_masks = torch.tensor(val_masks)


        train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
        train_sampler = RandomSampler(train_data)
        self.train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.args.bs)

        valid_data = TensorDataset(val_inputs, val_masks, val_tags)
        valid_sampler = SequentialSampler(valid_data)
        self.valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=self.args.bs)

        print("Loading BERT from transformers library version {}.".format(transformers.__version__))

        self.model = BertForTokenClassification.from_pretrained(
            'bert-'+args.model_size+'-cased',
            num_labels=len(data.tag2idx),
            output_attentions = False,
            output_hidden_states = False
        )
        self.model.to(args.device)

        FULL_FINETUNING = True
        if FULL_FINETUNING:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer = list(self.model.classifier.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=3e-5,
            eps=1e-8
        )

        self.epochs = 4
        self.max_grad_norm = 1.0

        # Total number of training steps is number of batches * number of epochs.
        self.total_steps = len(self.train_dataloader) * self.epochs

        # Create the learning rate scheduler.
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.total_steps
        )

def train(args, data_, model_):
    print("Beginning training...")

    ## Store the average loss after each epoch so we can plot them.
    loss_values, validation_loss_values = [], []

    for _ in trange(model_.epochs, desc="Epoch"):
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.

        # Put the model into training mode.
        model_.model.train()
        # Reset the total loss for this epoch.
        total_loss = 0

        # Training loop
        for step, batch in enumerate(tqdm(model_.train_dataloader, desc="Training Step")):
            # add batch to gpu
            batch = tuple(t.to(args.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            # Always clear any previously calculated gradients before performing a backward pass.
            model_.model.zero_grad()
            # forward pass
            # This will return the loss (rather than the model output)
            # because we have provided the `labels`.
            outputs = model_.model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
            # get the loss
            loss = outputs[0]
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # track train loss
            total_loss += loss.item()
            # Clip the norm of the gradient
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(parameters=model_.model.parameters(), max_norm=model_.max_grad_norm)
            # update parameters
            model_.optimizer.step()
            # Update the learning rate.
            model_.scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(model_.train_dataloader)
        print("Average train loss: {}".format(avg_train_loss))

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)


        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        # Put the model into evaluation mode
        model_.model.eval()
        # Reset the validation loss for this epoch.
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels = [], []
        for batch in tqdm(model_.valid_dataloader, desc="Validation Step"):
            batch = tuple(t.to(args.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have not provided labels.
                outputs = model_.model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels)
            # Move logits and labels to CPU
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            eval_loss += outputs[0].mean().item()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)

        eval_loss = eval_loss / len(model_.valid_dataloader)
        validation_loss_values.append(eval_loss)
        print("Validation loss: {}".format(eval_loss))
        pred_tags = [data_.tag_values[p_i] for p, l in zip(predictions, true_labels)
                                    for p_i, l_i in zip(p, l) if data_.tag_values[l_i] != "PAD"]
        valid_tags = [data_.tag_values[l_i] for l in true_labels
                                    for l_i in l if data_.tag_values[l_i] != "PAD"]
        print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
        print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
        print()

def build_model(args):
    print("Loading BERT tokeniser.")
    tokenizer = BertTokenizer.from_pretrained('bert-'+args.model_size+'-cased', do_lower_case=False)
    data_ = DataImporter(args)
    data_.tokenise(tokenizer)

    model_ = ModelConfig(args, data_) 

    train(args, data_, model_)

    model_.model.save_pretrained('./data'+args.model_size+'/')

    return model_.model, (data_.tag_values, data_.tag2idx), tokenizer

def load_model(args):
    tokenizer = BertTokenizer.from_pretrained('bert-'+args.model_size+'-cased', do_lower_case=False)
    model = BertForTokenClassification.from_pretrained('./data'+args.model_size+'/')
    
    tag_values = load_pkl('./data/tag_values.pkl')
    tag2idx = load_pkl('./data/tag2idx.pkl')
    
    model.to(args.device)

    return model, (tag_values, tag2idx), tokenizer


'''
if __name__ == "__main__":
    model, tag_embeddings, tokenizer = build_model()
'''