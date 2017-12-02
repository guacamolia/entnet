from bs4 import BeautifulSoup
import re
import numpy as np

def form_dataset(datafile):

    handler = open(datafile).read()
    soup = BeautifulSoup(handler, 'lxml')

    raw_stories = []
    raw_questions_ans = []
    for story in soup.find_all('instance'):
        raw_stories.append(story.find('text').get_text())
        story_questions_ans = []
        for question in story.find('questions').find_all('question'):
            ans_dict = {}
            for answer in question.find_all('answer'):
                if answer['correct'] == "True":
                    ans_dict[1] = answer['text']                            # TRUE
                else:
                    ans_dict[0] = answer['text']                            # FALSE
            story_questions_ans.append((question['text'], ans_dict))
        raw_questions_ans.append(story_questions_ans)

    stories = []
    for s in raw_stories:
        tokens = re.findall(r'(\w+)', s.lower())
        stories.append(tokens)

    data = []
    for i, qas in enumerate(raw_questions_ans):
        for q, a in qas:
            question_tokens = re.findall(r'(\w+)', q.lower())
            a[0] = re.findall(r'(\w+)', a[0].lower())
            a[1] = re.findall(r'(\w+)', a[1].lower())
            data.append([stories[i], (question_tokens, a)])

    return data

def get_voc(dataset):
    voc = set()
    for i in range(len(dataset)):
        story, qa = dataset[i]
        for token in story:
            voc.add(token)
        for token in qa[0]:
            voc.add(token)
        for token in qa[1][0]:
            voc.add(token)
        for token in qa[1][1]:
            voc.add(token)

    w2idx = {word: ind+1 for ind, word in enumerate(voc)}
    return w2idx

def encode_ind(dataset, w2idx):

    data_enc = []
    story_lens = []
    q_lens = []
    ans_lens = []

    for i in range(len(dataset)):
        story, qa = dataset[i]
        story_enc = [w2idx[token] for token in story if token in w2idx.keys()]
        q_enc = [w2idx[token] for token in qa[0] if token in w2idx.keys()]
        a_enc = {}
        a_enc[0] = [w2idx[token] for token in qa[1][0] if token in w2idx.keys()]
        a_enc[1] = [w2idx[token] for token in qa[1][1] if token in w2idx.keys()]
        data_enc.append([story_enc, (q_enc, a_enc)])

        story_lens.append(len(story_enc))
        q_lens.append(len(q_enc))
        ans_lens.append(len(a_enc[0]))
        ans_lens.append(len(a_enc[1]))

    # max_story_len = max(story_lens)
    # max_q_len = max(q_lens)
    # max_ans_len = max(ans_lens)

    story_len = int(np.ceil(np.mean(story_lens) + 2*np.std(story_lens)))
    q_len = int(np.ceil(np.mean(q_lens) + 2*np.std(q_lens)))
    ans_len = int(np.ceil(np.mean(ans_lens) + 2*np.std(ans_lens)))
    # print("Story" , np.mean(story_lens), np.std(story_lens))
    # print("Question", np.mean(q_lens), np.std(q_lens))
    # print("Answers" , np.mean(ans_lens), np.std(ans_lens))

    return data_enc, (story_len, q_len, ans_len)


def zero_padding(data_row, m_len, is_end=True):
    if is_end:
        if len(data_row) < m_len:
            data_row = data_row + [0]*(m_len - len(data_row))
        elif len(data_row) > m_len:
            data_row = data_row[:m_len]
    else:
        if len(data_row) < m_len:
            data_row = [0]*(m_len - len(data_row)) + data_row
        elif len(data_row) > m_len:
            data_row = data_row[:m_len]
    return data_row


def transform_data(dataset, lens, is_end=True):
    data_pad = []
    for i in range(len(dataset)):
        story, qa = dataset[i]
        story_pad = zero_padding(story, lens[0], is_end)
        q_pad = zero_padding(qa[0], lens[1], is_end)
        ans_pad = {}
        ans_pad[0] = zero_padding(qa[1][0], lens[2], is_end)
        ans_pad[1] = zero_padding(qa[1][1], lens[2], is_end)

        data_pad.append([story_pad, (q_pad, ans_pad)])
    return data_pad

if __name__ == "__main__":

    datafile = 'SemEval/dev-data.xml'
    data = form_dataset(datafile)
    enc_data, lens = encode_ind(data)
