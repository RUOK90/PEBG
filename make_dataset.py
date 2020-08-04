import csv
from tqdm import tqdm
import math
import pickle

from args import *


questions_file = '/shared/vida/processed/payment_diagnosis_questions.csv'
problem_solving_dir = '/shared/vida/processed/payment_problem_solving'
train_users_file = '/shared/vida/processed/payment_problem_solving_train_users.csv'

debug = False

question_item_dic = {}
question_tags_dic = {}
tag_questions_dic = {}
with open(questions_file, 'r') as f_r:
    lines = [line for line in csv.reader(f_r)][1:]
    for line in tqdm(lines):
        question_id, correct_answer, part, tags, time_limit, updated_at, train_unknown, test_unknown = line
        tags = tags.split(';')
        for tag in tags:
            tag = f't{tag}'
            if question_id not in question_tags_dic:
                question_tags_dic[question_id] = set()
            question_tags_dic[question_id].add(tag)

            if tag not in tag_questions_dic:
                tag_questions_dic[tag] = set()
            tag_questions_dic[tag].add(question_id)

        updated_at_correct_answer = [int(updated_at), correct_answer]
        train_unknown = int(train_unknown)
        test_unknown = int(test_unknown)

        if question_id in question_item_dic:
            question_item_dic[question_id]['updated_at_correct_answer'].append(updated_at_correct_answer)
        else:
            question_item_dic[question_id] = {'updated_at_correct_answer': [updated_at_correct_answer]}


# get difficulty, average elapsed time
with open(train_users_file, 'r') as f_r:
    train_users = [line[0] for line in csv.reader(f_r)][1:]
    if debug:
        train_users = train_users[:1000]


def get_is_correct(updated_at_correct_answer_list, timestamp, user_answer):
    for updated_at, correct_answer in updated_at_correct_answer_list:
        if timestamp > updated_at:
            is_correct = (correct_answer == user_answer)
    return is_correct


question_attr_dic = {}
for user in tqdm(train_users):
    with open(f'{problem_solving_dir}/{user}', 'r') as f_r:
        lines = [line for line in csv.reader(f_r)][1:]
        for line in lines:
            timestamp, content_id, is_diagnosis, user_answer, elapsed_time, estimated_elapsed_time, platform, estimated_score, payment = line
            if content_id not in question_attr_dic:
                question_attr_dic[content_id] = {'correct_cnt': 0, 'incorrect_cnt': 0, 'elapsed_times': []}

            timestamp = int(timestamp)
            is_correct = get_is_correct(question_item_dic[content_id]['updated_at_correct_answer'], timestamp, user_answer)
            elapsed_time = min(int(elapsed_time) / 1000.0, Constants.MAX_ELAPSED_TIME)
            if is_correct:
                question_attr_dic[content_id]['correct_cnt'] += 1
            else:
                question_attr_dic[content_id]['incorrect_cnt'] += 1
            question_attr_dic[content_id]['elapsed_times'].append(elapsed_time)


for question, attr in question_attr_dic.items():
    attr['difficulty'] = attr['correct_cnt'] / (attr['correct_cnt'] + attr['incorrect_cnt'])
    attr['avg_elapsed_time'] = sum(attr['elapsed_times']) / len(attr['elapsed_times'])
    attr['tags'] = question_tags_dic[question]


questions = list(question_tags_dic.keys())
tags = list(tag_questions_dic.keys())


# get question-tag relation
QT = []
for i, question in enumerate(tqdm(questions)):
    if debug and i == 1000:
        break
    question_tags = question_tags_dic[question]
    for tag in tags:
        if tag in question_tags:
            QT.append([question, tag, 1])
        else:
            QT.append([question, tag, 0])


# get question-question relation
QQ = []
for i in tqdm(range(len(questions))):
    if debug and i == 100:
        break
    question_1_tags = question_tags_dic[questions[i]]
    for j in range(i+1, len(questions)):
        question_2_tags = question_tags_dic[questions[j]]
        if len(question_1_tags & question_2_tags) != 0:
            QQ.append([questions[i], questions[j], 1])
        else:
            QQ.append([questions[i], questions[j], 0])


# get tag-tag relation
TT = []
for i in tqdm(range(len(tags))):
    if debug and i == 1000:
        break
    tag_1_questions = tag_questions_dic[tags[i]]
    for j in range(i+1, len(tags)):
        tag_2_questions = tag_questions_dic[tags[j]]
        if len(tag_1_questions & tag_2_questions) != 0:
            TT.append([tags[i], tags[j], 1])
        else:
            TT.append([tags[i], tags[j], 0])


question_tag_data_dic = {'questions': questions, 'tags': tags, 'question_attr_dic': question_attr_dic, 'QT': QT, 'QQ': QQ, 'TT': TT}
if debug:
    file_name = '/shared/vida/processed/debug_question_tag_data_dic.pkl'
else:
    file_name = '/shared/vida/processed/question_tag_data_dic.pkl'
with open(file_name, 'wb') as f_w:
    pickle.dump(question_tag_data_dic, f_w, pickle.HIGHEST_PROTOCOL)
