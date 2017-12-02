import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter
import data_loader
import time
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import preprocessing
from random import getrandbits
from tqdm import tqdm
import logging
import csv
import os

# Helper functions for CUDA
def cuda(obj):
    if torch.cuda.is_available():
        obj = obj.cuda()
    return obj


def variable(obj, volatile=False):
    if isinstance(obj, (list, tuple)):
        return [variable(o, volatile=volatile) for o in obj]
    obj = cuda(obj)
    obj = Variable(obj, volatile=volatile)
    return obj


def set_variable_repr():
    Variable.__repr__ = lambda x: 'Variable {}'.format(tuple(x.size()))
    Parameter.__repr__ = lambda x: 'Parameter {}'.format(tuple(x.size()))


class EntNet(nn.Module):
    def __init__(self, voc_size, max_len, n_slots, mem_size, n_sent, max_q_len):
        """Implementation of entity network
        https://arxiv.org/pdf/1612.03969.pdf"""
        super(EntNet, self).__init__()

        # Initialization of sizes
        self.voc_size = voc_size                                                                # V
        self.max_len = max_len                                                                  # L
        self.n_slots = n_slots                                                                  # N
        self.mem_size = mem_size                                                                # M
        self.sent = n_sent                                                                      # S
        self.max_q_len = max_q_len                                                              # Q

        # Initialization of layers
        self.U = nn.Linear(mem_size, mem_size)                                                  # [MxM]
        self.V = nn.Linear(mem_size, mem_size)                                                  # [MxM]
        self.W = nn.Linear(mem_size, mem_size)                                                  # [MxM]
        self.E = nn.Embedding(voc_size, mem_size, padding_idx=0)                                # [VxM]
        self.rnn = nn.LSTMCell(mem_size, mem_size)
        self.dropout = nn.Dropout(0.3)

        # Mask for positional encoding
        self.f = nn.Parameter(torch.randn(self.max_len), requires_grad=True)

        # Keys and memories
        self.w = nn.Parameter(torch.randn(self.n_slots, self.mem_size), requires_grad=True)     # NxM

        # Functions to be used
        self.relu = nn.ReLU()

        # Output module
        self.outputmodule = OutputModule(self.E, self.mem_size, self.voc_size, self.max_q_len)

        self.init_model_weights()

    def init_model_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal(m.weight.data)
                m.bias.data.zero_()

    def forward(self, input, question, ans):

        batch_size = input.data.shape[0]                                                        # B
        input = input.view(batch_size, -1)                                                      # Bx(SL)

        # Positional encoding
        e = self.E(input)                                                                         # [Bx(SL)xM]
        # e = e.view(batch_size, self.sent, self.max_len, self.mem_size)                          # [BxSxLxM]
        # f = self.f.unsqueeze(0).unsqueeze(0).unsqueeze(3)                                       # [1x1xLx1]
        # f = f.expand_as(e)                                                                      # [BxSxLxM]
        # s = torch.sum(f*e, 2)                                                                   # [BxSxM]
        s = e

        # Dynamic memory
        memories = []
        for j in range(self.n_slots):
            mem_j = variable(torch.Tensor(batch_size, self.mem_size).zero_())
            for t in range(self.sent):
                g = torch.sigmoid(torch.bmm(s[:, t, :].unsqueeze(1), mem_j.unsqueeze(2))
                                  + torch.bmm(s[:, t, :].unsqueeze(1), self.w[j, :].unsqueeze(0).expand(
                    batch_size, self.mem_size).unsqueeze(2))).squeeze(2)

                h_cand = self.relu(self.U(mem_j) + self.V(self.w[j, :].unsqueeze(0).expand(
                    batch_size, self.mem_size)) + self.W(s[:, t, :]))

                h_cand = self.dropout(h_cand)
                mem_j = mem_j + g*h_cand
                mem_j = mem_j / torch.norm(mem_j)
            memories.append(mem_j)

        h = torch.stack(memories, 1)                                                                # BxNxM

        # Output of the model
        output = self.outputmodule(question, h, ans)
        return output


class OutputModule(nn.Module):
    def __init__(self, E, mem_size, voc_size, max_q_len):
        super(OutputModule, self).__init__()
        """ Query presented"""

        # Sizes
        self.mem_size = mem_size                                                                    # M
        self.max_q_len = max_q_len                                                                  # Q

        self.f_q = nn.Parameter(torch.randn(self.max_q_len), requires_grad=True)                    # Q

        # Layers
        self.E = E                                                                                  # VxM
        self.R = nn.Linear(self.mem_size, voc_size)                                                 # MxV
        self.H = nn.Linear(self.mem_size, self.mem_size)                                            # BxM

        # Functions
        self.relu = nn.ReLU()
        self.s = nn.Softmax()

        self.init_model_weights()

    def init_model_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal(m.weight.data)
                m.bias.data.zero_()

    def forward(self, input, h, ans):

        # Positional encoding
        e = self.E(input)                                                                           # BxQxM
        # f_q = self.f_q.unsqueeze(0).unsqueeze(2).expand_as(e)                                       # BxQxM
        # q = torch.sum(f_q * e, 1)                                                                   # BxM
        q = torch.mean(e, 1)

        # Answers encoding
        a1 = ans[:, :, 0]
        a2 = ans[:, :, 1]
        a1 = self.E(a1)                                                                             # BxAxM
        a1 = torch.mean(a1, 1)                                                                        # BxM
        a2 = self.E(a2)
        a2 = torch.mean(a2, 1)                                                                        # BxM


        # Output
        p = self.s(torch.bmm(h, q.unsqueeze(2)).squeeze())
        u = torch.sum(p.unsqueeze(-1)*h, 1)
        # y = self.R(self.relu(q + self.H(u)))
        y1 = torch.sum(self.relu(q + self.H(u))*a1, 1)
        y2 = torch.sum(self.relu(q + self.H(u))*a2, 1)

        y = self.s(torch.stack((y1, y2), 1))
        return y


class BabiDataset(Dataset):
    def __init__(self, data_dir, task_id):

        train, test = data_loader.load_task_data(data_dir, task_id)
        self.train, self.test, self.voc = data_loader.convert_stories(train, test)
        self.train_stories, self.train_questions, self.train_answers = self.train

    def __len__(self):
        return len(self.train_stories)

    def __getitem__(self, idx):
        story = self.train_stories[idx]
        question = self.train_questions[idx]
        answer = self.train_answers[idx]
        return story, question, answer

    def get_max_q_len(self):
        return self.train_questions.shape[1]

    def get_max_len(self):
        return self.train_stories.shape[2]

    def get_n_sent(self):
        return self.train_stories.shape[1]

    def get_voc_size(self):
        return int(max(np.max(self.train_stories), np.max(self.train_questions)))

    def get_voc(self):
        return self.voc

class StoriesDataset(Dataset):
    def __init__(self, datafile, voc={}, lens=()):
        self.raw_data = preprocessing.form_dataset(datafile)
        if not voc:
            voc = preprocessing.get_voc(self.raw_data)

        self.lens = lens
        if not lens:
            _, self.lens = preprocessing.encode_ind(self.raw_data, voc)

        enc_data, _ = preprocessing.encode_ind(self.raw_data, voc)

        self.data = preprocessing.transform_data(enc_data, self.lens, is_end=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        story, qa = self.data[idx]
        story = np.array(story)
        q = np.array(qa[0])
        a1 = np.array(qa[1][0])
        a2 = np.array(qa[1][1])
        first = getrandbits(1)
        if first == 1:
            a = np.stack((a1, a2), 1)
            cor = 1
        else:
            a = np.stack((a2, a1), 1)
            cor = 0
        return story, q, a, cor

    def get_lens(self):
        return self.lens

    def get_voc(self):
        return preprocessing.get_voc(self.raw_data)

def train_model(n_epochs, model, dataloaders):

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    tick = time.time()
    best = 0
    for epoch in range(n_epochs + 1):
        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            correct = 0
            total = 0
            for i_batch, sample_batched in enumerate(tqdm(dataloaders[phase])):
                stories, questions, ans, cor = sample_batched
                stories, questions, ans, cor = variable(stories), variable(questions), variable(ans), variable(cor)
                # if i_batch == 2:
                #     break
                optimizer.zero_grad()

                answers = model(stories, questions, ans)                                                     # Bx2

                loss = F.cross_entropy(answers, cor)

                _, labels = torch.max(answers, 1)
                correct += cor.eq(labels.long()).sum().data[0]
                total += cor.data.shape[0]

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            acc = correct*1.0/total
            tock = time.time()
            if epoch % 1 == 0:
                print('{} elapsed: {} \tepoch: {}\t\t -- loss: {} \t -- accuracy: {}'.format(phase.upper(), tock - tick,
                                                                                             epoch,
                                                                                             loss.data[0], acc))
                if phase == 'dev':
                    if acc > best:
                        best = acc
                        save_weights(model, "entity_trained.pt")

    return answers


def predictions(story, question, answer, target, voc):
    story = story.cpu().numpy()
    question = question.cpu().numpy()
    answer = answer.cpu().numpy()
    target = target.cpu().numpy()

    answer = voc[np.argmax(answer)]
    target = voc[target[0]]
    word_story = []
    story = [[voc[idx] for idx in sent if idx != 0] for sent in story]
    question = [voc[idx] for idx in question if idx != 0]
    for sent in story:
        if sent != []:
            string = " ".join(sent)
            word_story.append(string)
    question = " ".join(question)
    return word_story, question, answer, target


def save_voc(voc, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        for key, value in voc.items():
            writer.writerow([key, value])
    return


def restore_weights(model, filename):
    if not isinstance(filename, str):
        filename = str(filename)

    map_location = None

    # load trained on GPU models to CPU
    if not torch.cuda.is_available():
        map_location = lambda storage, loc: storage

    state_dict = torch.load(filename, map_location=map_location)

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    model.load_state_dict(state_dict)

    logging.info(f'Model restored: {os.path.basename(filename)}')


def save_weights(model, filename):
    if not isinstance(filename, str):
        filename = str(filename)

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    torch.save(model.state_dict(), filename)

    logging.info(f'Model saved: {os.path.basename(filename)}')

if __name__ == "__main__":

    task_ids = [
        'qa1_single-supporting-fact',
        'qa2_two-supporting-facts',
        'qa3_three-supporting-facts',
        'qa4_two-arg-relations',
        'qa5_three-arg-relations',
        'qa6_yes-no-questions',
        'qa7_counting',
        'qa8_lists-sets',
        'qa9_simple-negation',
        'qa10_indefinite-knowledge',
        'qa11_basic-coreference',
        'qa12_conjunction',
        'qa13_compound-coreference',
        'qa14_time-reasoning',
        'qa15_basic-deduction',
        'qa16_basic-induction',
        'qa17_positional-reasoning',
        'qa18_size-reasoning',
        'qa19_path-finding',
        'qa20_agents-motivations',
    ]

    # data_dir = '/home/okovaleva/projects/semeval/bAbI/tasks_1-20_v1-2/en'
    train_data_dir = 'SemEval/train-data.xml'
    dev_data_dir = 'SemEval/dev-data.xml'
    task_id = task_ids[1]

    # Network params
    batch_size = 256
    n_slots = 30
    mem_size = 100
    n_epochs = 300

    # converted_dataset = BabiDataset(data_dir, task_id)
    converted_dataset_train = StoriesDataset(train_data_dir)
    voc_train = converted_dataset_train.get_voc()
    save_voc(voc_train, 'vocabulary.csv')
    lens_train = converted_dataset_train.get_lens()
    converted_dataset_dev = StoriesDataset(dev_data_dir, voc_train, lens_train)

    dataloader_train = DataLoader(converted_dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_dev = DataLoader(converted_dataset_dev, batch_size=batch_size, shuffle=True)

    dataloaders = {'train': dataloader_train, 'dev': dataloader_dev}

    # voc_size = converted_dataset.get_voc_size() + 1
    # max_len = converted_dataset.get_max_len()
    # max_q_len = converted_dataset.get_max_q_len()
    # n_sent = converted_dataset.get_n_sent()
    # voc = converted_dataset.get_voc()


    max_len = 1
    n_sent, max_q_len, _ = converted_dataset_train.get_lens()

    voc_size = len(voc_train) + 1

    set_variable_repr()

    entnet = cuda(torch.nn.DataParallel(EntNet(voc_size, max_len, n_slots, mem_size, n_sent, max_q_len)))

    # Train model
    train_model(n_epochs, entnet, dataloaders)



