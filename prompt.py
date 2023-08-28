import json
from PACKAGE import asyncThread, metric_realization, multi_rouge
import os
from rouge_score.scoring import AggregateScore, Score
import bert_score
from tqdm import tqdm

ej_api_key = ""
my_api_key = ""


class Summarize:
    @staticmethod
    def gpt_run(user_list):
        system = [
            '''Imagine you are a human annotator. You will receive a query and a article.Read the article and answer the question in about 200-400 words.The question may be a specific question or a general question which ask you to summarize the storyline. Both kinds are all answerable. Please read the article carefully.''',
            '''You are a helpful assistant that gives long answer to question based on a long story.''',
            '''You are a helpful assistant that gives long answer to question based on a long meeting.''']
        # messages = [[{"role": "user", "content": f"print the number {i}"}] for i in range(100)]
        messages = [[{"role": "system", "content": system[1]},
                     # {"role": "user", "content": user1},
                     # {"role": "assistant", "content": assistant1},
                     {"role": "user", "content": user}] for user in user_list]
        response_list = asyncThread.run(messages=messages,
                                        engine_name="gpt-3.5-turbo-16k-0613",
                                        temperature=0.7,
                                        max_tokens=600,
                                        top_p=0.9,
                                        api_key=ej_api_key,
                                        requests_per_minute=40)
        return response_list

    @staticmethod
    def get_input(data):
        input_user_list = []
        for item in data:
            query = item['Query']
            article = item['Article']
            # input_string = "Query:" + query + "\n\nArticle:" + article
            input_string = f"Write an answer based on the following question and the story.\n QUESTION:{query}\n STORY:{article}\n SUMMARY: \n"
            # input_string = f"Write an answer based on the following question and the meeting.\n QUESTION:{query}\n MEETING:{article}\n SUMMARY: \n"

            input_user_list.append(input_string)
        return input_user_list

    @staticmethod
    def get_qmsum_input(data):
        # segmentate
        input_user_list = []
        return input_user_list

    @staticmethod
    def traverse_summary(folder_path):
        # folder_path 指定要遍历的文件夹路径
        # folder_path = "SQuALITY/sparse/min"
        wait_process_files = ['max.json', 'mean.json', 'min.json', 'Academic.json', 'Committee.json', 'Product.json',
                              'dev.json', 'test.json', 'train.json']
        for root, dirs, files in os.walk(folder_path):
            # 遍历当前目录下的文件夹
            for dir_name in dirs:
                print("文件夹：", os.path.join(root, dir_name))
            # 遍历当前目录下的文件
            for file_name in files:
                # If it is origin file
                if file_name not in wait_process_files:
                    continue
                # Load data
                with open(os.path.join(root, file_name), 'r') as f:
                    data = json.load(f)
                # Get input
                input_user_list = Summarize.get_input(data)
                # Get response
                new_summary = Summarize.gpt_run(input_user_list)
                # Write response
                with open(root + '/summary/newSummary_' + file_name, 'w') as f:
                    temp = json.dumps(new_summary, indent=4)
                    f.write(temp)


class Evaluate:
    @staticmethod
    def load_pred(path):
        predictions = []
        pred_files = ['newSummary_max.json', 'newSummary_mean.json', 'newSummary_min.json', 'newSummary_Academic.json',
                      'newSummary_Committee.json', 'newSummary_Product.json', 'newSummary_dev.json',
                      'newSummary_test.json', 'newSummary_train.json']

        for pred_file in pred_files:
            file_path = os.path.join(path, 'summary/' + pred_file)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    item_pred = json.load(f)
                predictions.extend(item_pred)
        return predictions

    @staticmethod
    def load_ref(path):
        ref_files = ['max.json', 'mean.json', 'min.json', 'Academic.json', 'Committee.json', 'Product.json', 'dev.json',
                     'test.json', 'train.json']
        references = []
        for ref_file in ref_files:
            file_path = os.path.join(path, ref_file)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    item_ref = json.load(f)
                item_ref = [
                    [data_item['Summary_1'], data_item['Summary_2'], data_item['Summary_3'], data_item['Summary_4']]
                    for data_item in item_ref]
                references.extend(item_ref)
        return references

    @staticmethod
    def squality_rouge(path, predictions, references):
        print('Evaluate rouge score (use squality)')
        rouge_object = multi_rouge.Rouge()
        squality_rouge_score = rouge_object._compute(predictions=predictions, references=references)
        file_path = os.path.join(path, 'evaluation/evaluate_squality_rouge.json')
        with open(file_path, 'w') as f:
            f.write(str(squality_rouge_score))

    @staticmethod
    def bert(path, predictions, references):
        print('Evaluate bert score')

        # 批次大小
        batch_size = 16

        scores = {'p': [], 'r': [], 'f1': []}
        num_batches = (len(predictions) + batch_size - 1) // batch_size  # 计算需要的批次数量

        for i in tqdm(range(num_batches)):
            start = i * batch_size
            end = min(start + batch_size, len(predictions))

            pred_batch = predictions[start:end]
            ref_batch = references[start:end]

            p, r, f1 = bert_score.score(pred_batch, ref_batch, lang='en')
            # Add in scores
            for index in range(len(p)):
                scores['r'].append(float(p[index]))
                scores['p'].append(float(r[index]))
                scores['f1'].append(float(f1[index]))

        file_path = os.path.join(path, 'evaluation/evaluate_bert_score.json')
        with open(file_path, 'w') as f:
            temp = json.dumps(scores)
            f.write(temp)

    @staticmethod
    def another_rouge(path, predictions, references):
        print('Evaluate rouge score (use another way)')
        rouge_score = metric_realization.calculate_rouge(ref=references, pred=predictions)
        with open(os.path.join(path, 'evaluate_rouge.json'), 'w') as f:
            temp = json.dumps(rouge_score)
            f.write(temp)

    @staticmethod
    def evaluate(path):
        # Load predictions
        predictions = Evaluate.load_pred(path)

        # Load references
        references = Evaluate.load_ref(path)

        # Delete empty
        references = [references[index] for index, item in enumerate(predictions) if item != '']
        predictions = [predictions[index] for index, item in enumerate(predictions) if item != '']

        # Evaluate.squality_rouge(path, predictions, references)
        Evaluate.bert(path, predictions, references)
        # Evaluate.another_rouge(path, predictions, references)

    @staticmethod
    def traverse_path():
        paths = [
            'SQuALITY/sparse/max/',
            'SQuALITY/sparse/min/',
            'SQuALITY/sparse/mean/',
            'SQuALITY/LLM-embedding/min',
            # 'SQuALITY/LLM-embedding/max',
            'SQuALITY/LLM-embedding/mean']

        for path in paths:
            Evaluate.evaluate(path)

        # Load data

    @staticmethod
    def print_score(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file == 'evaluate_squality_rouge.json':
                    with open(os.path.join(root, file), 'r') as f:
                        rouge = f.read()
                        obj_rouge = eval(rouge)
                        print(root)
                        print(f"rouge1:{obj_rouge['rouge1'].mid.fmeasure * 100:.2f}")
                        print(f"rouge2:{obj_rouge['rouge2'].mid.fmeasure * 100:.2f}")
                        print(f"rougeL:{obj_rouge['rougeL'].mid.fmeasure * 100:.2f}")


# Summarize.traverse_summary('SQuALITY/LLM-embedding/min')
Evaluate.traverse_path()
# Evaluate.print_score('SQuALITY')
