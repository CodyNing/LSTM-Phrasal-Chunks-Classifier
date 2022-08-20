# Code adapted from original code by Robert Guthrie

import os, sys, optparse, gzip, re, logging, string, random, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

PRINTABLEDICT = {v:i for i, v in enumerate(string.printable)}

def read_conll(handle, input_idx=0, label_idx=2):
    conll_data = []
    contents = re.sub(r'\n\s*\n', r'\n\n', handle.read())
    contents = contents.rstrip()
    for sent_string in contents.split('\n\n'):
        annotations = list(zip(*[ word_string.split() for word_string in sent_string.split('\n') ]))
        assert(input_idx < len(annotations))
        if label_idx < 0:
            conll_data.append( annotations[input_idx] )
            logging.info("CoNLL: {}".format( " ".join(annotations[input_idx])))
        else:
            assert(label_idx < len(annotations))
            conll_data.append(( annotations[input_idx], annotations[label_idx] ))
            logging.info("CoNLL: {} ||| {}".format( " ".join(annotations[input_idx]), " ".join(annotations[label_idx])))
    return conll_data

def read_vocab(handle):
    vocab = {line.split()[1] for line in handle}
    return vocab

def prepare_sequence(seq, to_ix, unk):
    idxs = []
    if unk not in to_ix:
        idxs = [to_ix[w] for w in seq]
    else:
        idxs = [to_ix[w] for w in map(lambda w: unk if w not in to_ix else w, seq)]
    return torch.tensor(idxs, dtype=torch.long)

def semiChara(sentence, unk):
    vec = torch.empty((len(sentence), 300))
    for i, word in enumerate(sentence):
        vs = [ torch.zeros(100) for i in range(3)]
        if(word != unk):
            vs[0][ PRINTABLEDICT[ word[0] ] ] = 1
            for j in range(1, len(word) - 1):
                vs[1][ PRINTABLEDICT[ word[j] ] ] += 1
            vs[2][ PRINTABLEDICT[ word[len(word) - 1] ] ] = 1
        vec[i] = torch.cat(vs, 0)
    return vec

def noisy_word_map(ori_data):
    data = {}
    for word in ori_data:
        if len(word) < 4:
            continue

        data[word] = []
        wlen = len(word)
        #swap
        i = random.randint(1, wlen - 3)
        swap = ''.join([word[:i], word[i+1], word[i], word[i+2:]])

        #drop
        i = random.randint(1, wlen - 2)
        drop = ''.join([word[:i], word[i+1:]])

        #insert
        i = random.randint(1, wlen - 2)
        j = random.randint(10, 35) #'a' - 'z'
        insert = ''.join([word[:i], string.printable[j], word[i:]])

        data[word].append(swap)
        data[word].append(drop)
        data[word].append(insert)

    return data

def get_noisy_word(word, unk):
    if len(word) < 4 or word == unk:
        return word

    i = random.randint(0, 2)
    
    if (i == 0):
        return swap_word(word)
    elif (i == 1):
        return drop_word(word)
    else:
        return inert_word(word)
    

def swap_word(word):
    wlen = len(word)
    i = random.randint(1, wlen - 3)
    return ''.join([word[:i], word[i+1], word[i], word[i+2:]])

def drop_word(word):
    wlen = len(word)
    i = random.randint(1, wlen - 2)
    return ''.join([word[:i], word[i+1:]])

def inert_word(word):
    wlen = len(word)
    i = random.randint(1, wlen - 2)
    j = random.randint(10, 35) #'a' - 'z'
    return ''.join([word[:i], string.printable[j], word[i:]])

def prepare_noisy_sequence(sentence, noisy_ratio, unk):
    noisy = [w for w in sentence]
    slen = len(sentence)
    noisy_num = int(noisy_ratio * slen)
    for i in random.sample(range(0, slen), noisy_num):
        noisy[i] = get_noisy_word(sentence[i], unk)

    return semiChara(noisy, unk)

class LSTMTaggerModel(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, idx_to_word):
        torch.manual_seed(1)
        super(LSTMTaggerModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim + 300, hidden_dim, bidirectional=False)
        # self.lstm = nn.LSTM(embedding_dim * 2, hidden_dim, bidirectional=False)
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=False)


        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        self.idx_to_word = idx_to_word

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        semivec = self.semiChara(sentence)
        ebcat = torch.cat((embeds, semivec), dim=1)
        # semi_lstm_out, _ = self.semi_lstm(semivec.view(len(sentence), 1, -1))
        # ebcat = torch.cat((embeds, semi_lstm_out.view(len(sentence), -1)), dim=1)
        lstm_out, _ = self.lstm(ebcat.view(len(sentence), 1, -1))
        # lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

    def semiChara(self, sentence):
        vec = torch.empty((len(sentence), 300))
        for i, word_idx in enumerate(sentence):
            word = self.idx_to_word[word_idx.item()]
            vs = [ torch.zeros(100) for i in range(3)]
            vs[0][ PRINTABLEDICT[ word[0] ] ] = 1
            for j in range(1, len(word) - 1):
                vs[1][ PRINTABLEDICT[ word[j] ] ] += 1
            vs[2][ PRINTABLEDICT[ word[len(word) - 1] ] ] = 1
            vec[i] = torch.cat(vs, 0)
        return vec

class LSTMCLFModel(nn.Module):

    def __init__(self, hidden_dim, vocab_size):
        torch.manual_seed(1)
        super(LSTMCLFModel, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(300, hidden_dim, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, sentence):
        lstm_out, _ = self.lstm(sentence.view(len(sentence), 1, -1))
        vocab_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        vocab_score = F.log_softmax(vocab_space, dim=1)
        return vocab_score

class LSTMTagger:

    def __init__(self, trainfile, vocabfile, modelfile, modelsuffix, unk="[UNK]", epochs=10, embedding_dim=128, hidden_dim=64):
        self.unk = unk
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.modelfile = modelfile
        self.modelsuffix = modelsuffix
        self.training_data = []
        self.vocab_data = {}
        if trainfile[-3:] == '.gz':
            with gzip.open(trainfile, 'rt') as f:
                self.training_data = read_conll(f)
        else:
            with open(trainfile, 'r') as f:
                self.training_data = read_conll(f)
        
        with open(vocabfile, 'r') as f:
            self.vocab_data = read_vocab(f)

        self.word_to_ix = {} # replaces words with an index (one-hot vector)
        self.tag_to_ix = {} # replace output labels / tags with an index
        self.ix_to_tag = [] # during inference we produce tag indices so we have to map it back to a tag

        for sent, tags in self.training_data:
            for word in sent:
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)
            for tag in tags:
                if tag not in self.tag_to_ix:
                    self.tag_to_ix[tag] = len(self.tag_to_ix)
                    self.ix_to_tag.append(tag)

        logging.info("word_to_ix:", self.word_to_ix)
        logging.info("tag_to_ix:", self.tag_to_ix)
        logging.info("ix_to_tag:", self.ix_to_tag)
        self.ix_to_word = {v:k for k, v in self.word_to_ix.items()}

        self.model = LSTMTaggerModel(self.embedding_dim, self.hidden_dim, len(self.word_to_ix), len(self.tag_to_ix), self.ix_to_word)
        self.clf = LSTMCLFModel(self.hidden_dim, len(self.word_to_ix))
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.clfoptimizer = optim.SGD(self.clf.parameters(), lr=0.01)
        self.clfscheduler = optim.lr_scheduler.OneCycleLR(self.clfoptimizer, max_lr=0.1, steps_per_epoch=len(self.training_data), epochs=self.epochs)

    def argmax(self, seq):
        output = []
        with torch.no_grad():
            inputs = prepare_sequence(seq, self.word_to_ix, self.unk)
            tag_scores = self.model(inputs)
            for i in range(len(inputs)):
                output.append(self.ix_to_tag[int(tag_scores[i].argmax(dim=0))])
        return output

    def argmax_combine(self, seq):
        output = []
        with torch.no_grad():
            semi_vecs = semiChara(seq, self.unk)
            clf_out = self.clf(semi_vecs)
            # inputs_a = prepare_sequence(seq, self.word_to_ix, self.unk)
            sen = []
            for i in range(len(clf_out)):
                sen.append(self.ix_to_word[int(clf_out[i].argmax(dim=0))])
            # print(seq)
            # print(sen)
            # exit()
            inputs = clf_out.argmax(dim=1)
            tag_scores = self.model(inputs)
            for i in range(len(inputs)):
                output.append(self.ix_to_tag[int(tag_scores[i].argmax(dim=0))])
        return output


    def train_clf(self):
        loss_function = nn.NLLLoss()

        # noisy_map = noisy_word_map(self.vocab_data)

        self.clf.train()
        loss = float("inf")
        for epoch in range(self.epochs):
            epoch_loss = 0
            for sentence, tags in tqdm.tqdm(self.training_data):
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                self.clf.zero_grad()

                # Step 2. Get our inputs ready for the network, that is, turn them into
                # Tensors of word indices.
                semi_vecs = prepare_noisy_sequence(sentence, 1.0, self.unk)
                true_word = prepare_sequence(sentence, self.word_to_ix, self.unk)

                # Step 3. Run our forward pass.
                clf_scores = self.clf(semi_vecs)

                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss = loss_function(clf_scores, true_word)
                epoch_loss += loss.item()
                loss.backward()
                self.clfoptimizer.step()
                self.clfscheduler.step()

            print("Epoch: {}, Loss: {}".format(epoch, epoch_loss / len(self.training_data)))
            if epoch == self.epochs-1:
                epoch_str = '' # last epoch so do not use epoch number in model filename
            else:
                epoch_str = str(epoch)
            savefile = self.modelfile + '_clf' + epoch_str + self.modelsuffix
            print("saving model file: {}".format(savefile), file=sys.stderr)
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.clf.state_dict(),
                        'optimizer_state_dict': self.clfoptimizer.state_dict(),
                        'loss': loss,
                        'unk': self.unk,
                        'word_to_ix': self.word_to_ix,
                        'tag_to_ix': self.tag_to_ix,
                        'ix_to_tag': self.ix_to_tag,
                    }, savefile)

    def train(self):
        loss_function = nn.NLLLoss()

        self.model.train()
        loss = float("inf")
        for epoch in range(self.epochs):
            epoch_loss = 0
            for sentence, tags in tqdm.tqdm(self.training_data):
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                self.model.zero_grad()

                # Step 2. Get our inputs ready for the network, that is, turn them into
                # Tensors of word indices.
                sentence_in = prepare_sequence(sentence, self.word_to_ix, self.unk)
                targets = prepare_sequence(tags, self.tag_to_ix, self.unk)

                # Step 3. Run our forward pass.
                tag_scores = self.model(sentence_in)

                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss = loss_function(tag_scores, targets)
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            print("Epoch: {}, Loss: {}".format(epoch, epoch_loss / len(self.training_data)))
            if epoch == self.epochs-1:
                epoch_str = '' # last epoch so do not use epoch number in model filename
            else:
                epoch_str = str(epoch)
            savefile = self.modelfile + epoch_str + self.modelsuffix
            print("saving model file: {}".format(savefile), file=sys.stderr)
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss,
                        'unk': self.unk,
                        'word_to_ix': self.word_to_ix,
                        'tag_to_ix': self.tag_to_ix,
                        'ix_to_tag': self.ix_to_tag,
                    }, savefile)

    def decode(self, inputfile):
        if inputfile[-3:] == '.gz':
            with gzip.open(inputfile, 'rt') as f:
                input_data = read_conll(f, input_idx=0, label_idx=-1)
        else:
            with open(inputfile, 'r') as f:
                input_data = read_conll(f, input_idx=0, label_idx=-1)

        if not os.path.isfile(self.modelfile + self.modelsuffix):
            raise IOError("Error: missing model file {}".format(self.modelfile + self.modelsuffix))

        saved_model = torch.load(self.modelfile + self.modelsuffix)
        self.model.load_state_dict(saved_model['model_state_dict'])
        self.optimizer.load_state_dict(saved_model['optimizer_state_dict'])
        epoch = saved_model['epoch']
        loss = saved_model['loss']
        self.unk = saved_model['unk']
        self.word_to_ix = saved_model['word_to_ix']
        self.tag_to_ix = saved_model['tag_to_ix']
        self.ix_to_tag = saved_model['ix_to_tag']
        self.model.eval()

        saved_clf = torch.load(self.modelfile + '_clf' + self.modelsuffix)
        self.clf.load_state_dict(saved_clf['model_state_dict'])
        self.clfoptimizer.load_state_dict(saved_clf['optimizer_state_dict'])
        self.clf.eval()

        decoder_output = []
        for sent in tqdm.tqdm(input_data):
            decoder_output.append(self.argmax_combine(sent))
        return decoder_output

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--inputfile", dest="inputfile", default=os.path.join('data', 'input', 'dev.txt'), help="produce chunking output for this input file")
    optparser.add_option("-t", "--trainfile", dest="trainfile", default=os.path.join('data', 'train.txt.gz'), help="training data for chunker")
    optparser.add_option("-v", "--vocabfile", dest="vocabfile", default=os.path.join('data', 'train.vocab'), help="vocab data for classifier")
    optparser.add_option("-m", "--modelfile", dest="modelfile", default=os.path.join('data', 'chunker'), help="filename without suffix for model files")
    optparser.add_option("-s", "--modelsuffix", dest="modelsuffix", default='.tar', help="filename suffix for model files")
    optparser.add_option("-e", "--epochs", dest="epochs", default=5, help="number of epochs [fix at 5]")
    optparser.add_option("-u", "--unknowntoken", dest="unk", default='[UNK]', help="unknown word token")
    optparser.add_option("-f", "--force", dest="force", action="store_true", default=False, help="force training phase (warning: can be slow)")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    modelfile = opts.modelfile
    if opts.modelfile[-4:] == '.tar':
        modelfile = opts.modelfile[:-4]
    chunker = LSTMTagger(opts.trainfile, opts.vocabfile, modelfile, opts.modelsuffix, opts.unk)
    # use the model file if available and opts.force is False
    # if os.path.isfile(opts.modelfile + opts.modelsuffix) and not opts.force:
    #     decoder_output = chunker.decode(opts.inputfile)
    # else:
    #     print("Warning: could not find modelfile {}. Starting training.".format(modelfile + opts.modelsuffix), file=sys.stderr)
    #     chunker.train_clf()
    #     chunker.train()
    #     decoder_output = chunker.decode(opts.inputfile)

    if not os.path.isfile(opts.modelfile + '_clf' + opts.modelsuffix) or opts.force:
        chunker.train_clf()

    if not os.path.isfile(opts.modelfile + opts.modelsuffix) or opts.force:
        chunker.train()
    
    decoder_output = chunker.decode(opts.inputfile)

    print("\n\n".join([ "\n".join(output) for output in decoder_output ]))
