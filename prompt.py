import json
from PACKAGE import asyncThread, metric_realization, multi_rouge
import os
import bert_score
from tqdm import tqdm

api_key = "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"


class Summarize:
    @staticmethod
    def gpt_run(user_list, requests_per_minute):
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
                                        api_key=api_key,
                                        requests_per_minute=requests_per_minute)
        return response_list

    @staticmethod
    def get_input(data):
        input_user_list = []
        for item in data:
            query = item['Query']
            article = item['Article']
            input_string = f"Write an answer based on the following question and the story.\n QUESTION:{query}\n STORY:{article}\n SUMMARY: \n"
            input_user_list.append(input_string)
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

                # Set requests_per_minute
                requests_per_minute = 20
                if root.endswith('min'):
                    requests_per_minute = 40
                # Load data
                with open(os.path.join(root, file_name), 'r') as f:
                    data = json.load(f)
                # Get input
                input_user_list = Summarize.get_input(data)
                # Get response
                new_summary = Summarize.gpt_run(input_user_list, requests_per_minute)
                # Write response
                with open(root + '/summary/newSummary_' + file_name, 'w') as f:
                    temp = json.dumps(new_summary, indent=4)
                    f.write(temp)


class Evaluate:
    @staticmethod
    def load_pref_ref(path):
        predictions = []
        references = []

        ref_files = [
            'max.json', 'mean.json', 'min.json', 'Academic.json', 'Committee.json', 'Product.json', 'dev.json',
            'test.json',
            'train.json']

        for ref_file in ref_files:
            ref_file_path = os.path.join(path, ref_file)
            pred_file_path = os.path.join(path, 'gpt3_summary_' + ref_file)

            if os.path.exists(ref_file_path):
                with open(ref_file_path, 'r') as f:
                    item_ref = json.load(f)
                item_ref = [
                    [data_item['Summary_1'], data_item['Summary_2'], data_item['Summary_3'], data_item['Summary_4']]
                    for data_item in item_ref]
                references.extend(item_ref)

            if os.path.exists(pred_file_path):
                with open(pred_file_path, 'r') as f:
                    item_pred = json.load(f)
                predictions.extend(item_pred)

        return predictions, references

    @staticmethod
    def squality_rouge(path, predictions, references):
        print('Evaluate rouge score (use squality)')
        rouge_object = multi_rouge.Rouge()
        squality_rouge_score = rouge_object._compute(predictions=predictions, references=references, use_stemmer=True)
        file_path = os.path.join(path, 'evaluate_squality_rouge.json')
        with open(file_path, 'w') as f:
            f.write(str(squality_rouge_score))

    @staticmethod
    def bert(path, predictions, references):
        print('Evaluate bert score')

        # 批次大小
        batch_size = 16

        bert_scores = {'p': [], 'r': [], 'f1': [], 'average_p': 0, 'average_r': 0, 'average_f1': 0}
        num_batches = (len(predictions) + batch_size - 1) // batch_size  # 计算需要的批次数量

        for i in tqdm(range(num_batches)):
            start = i * batch_size
            end = min(start + batch_size, len(predictions))

            pred_batch = predictions[start:end]
            ref_batch = references[start:end]

            p, r, f1 = bert_score.score(pred_batch, ref_batch, lang='en')
            # Add in bert_scores
            for index in range(len(p)):
                bert_scores['r'].append(float(p[index]))
                bert_scores['p'].append(float(r[index]))
                bert_scores['f1'].append(float(f1[index]))

        # Calculate average bert
        average_p = sum(bert_scores['p']) / len(bert_scores['p'])
        average_r = sum(bert_scores['r']) / len(bert_scores['r'])
        average_f1 = sum(bert_scores['f1']) / len(bert_scores['f1'])
        bert_scores['average_p'] = average_p
        bert_scores['average_r'] = average_r
        bert_scores['average_f1'] = average_f1

        file_path = os.path.join(path, 'evaluation/evaluate_bert_score.json')
        with open(file_path, 'w') as f:
            temp = json.dumps(bert_scores)
            f.write(temp)

    @staticmethod
    def another_rouge(path, predictions, references):
        print('Evaluate rouge score (use another way)')
        rouge_score = metric_realization.calculate_rouge(ref=references, pred=predictions)
        with open(os.path.join(path, 'evaluation/evaluate_rouge.json'), 'w') as f:
            temp = json.dumps(rouge_score)
            f.write(temp)

    @staticmethod
    def bleurt(path, predictions, references):
        print('Evaluate bleurt score')
        rouge_score = metric_realization.calculate_bert_score(ref=references, pred=predictions)
        with open(os.path.join(path, 'evaluation/evaluate_bleurt.json'), 'w') as f:
            temp = json.dumps(rouge_score)
            f.write(temp)

    @staticmethod
    def gpt_eval(path, predictions, references, bart=False):
        # Get prompt
        metric_list = ['coh', 'con', 'flu', 'rel']
        metric_type = metric_list[3]
        prompt = open('GPTeval/prompts/' + metric_type + '_detailed.txt').read()
        # Get messages
        messages = []
        for index, prediction in enumerate(predictions):
            reference = references[index]
            cur_prompt = prompt.replace('{{Document}}', reference).replace('{{Summary}}', prediction)
            messages.append([{"role": "system", "content": cur_prompt}])
        # Send request
        response_list = asyncThread.run(messages=messages,
                                        engine_name="gpt-3.5-turbo-16k-0613",
                                        temperature=1,
                                        max_tokens=5,
                                        top_p=1,
                                        api_key=api_key,
                                        requests_per_minute=200)
        # Del non-numeric
        num_list = ['1', '2', '3', '4', '5']
        response_list = [item for item in response_list if item[0] in num_list]
        # Calaulate Average
        # Save
        save_path = os.path.join(path, 'evaluation/gpt3_' + metric_type + '_gpteval.json')
        if bart:
            save_path = os.path.join(path, 'evaluation/bart_' + metric_type + '_gpteval.json')
        with open(save_path, 'w') as f:
            print(f'write to {path}')
            temp = json.dumps(response_list)
            f.write(temp)

    @staticmethod
    def evaluate(path, bert=False, rouge=False, another_rouge=False, bleurt=False, bart=False, gpteval=False):
        # Load predictions, references
        if gpteval:
            predictions, references = GPTeval.load_pred_ref(path, bart)
        elif bart:
            predictions = BartEvaluation.load_pred(path)
            references = BartEvaluation.load_ref(path)
        else:
            predictions, references = Evaluate.load_pref_ref(path)

        # Delete empty
        references = [references[index] for index, item in enumerate(predictions) if item != '']
        predictions = [predictions[index] for index, item in enumerate(predictions) if item != '']

        # Test
        if rouge:
            Evaluate.squality_rouge(path, predictions, references)
        if bert:
            Evaluate.bert(path, predictions, references)
        if another_rouge:
            Evaluate.another_rouge(path, predictions, references)
        if bleurt:
            Evaluate.bleurt(path, predictions, references)
        if gpteval:
            Evaluate.gpt_eval(path, predictions, references, bart)

    @staticmethod
    def traverse_path(root, bert=False, rouge=False, another_rouge=False, bleurt=False, bart=False, gpteval=False):
        for path, dirs, files in os.walk(root):
            if files and dirs:
                Evaluate.evaluate(path, bert, rouge, another_rouge, bleurt, bart, gpteval)
            else:
                continue


class BartEvaluation:
    @staticmethod
    # Load pred
    def load_pred(path):
        file_path = os.path.join(path, 'generated_predictions.json')
        with open(file_path, 'r') as f:
            return json.load(f)

    @staticmethod
    # Load ref
    def load_ref(path):
        file_path = os.path.join(path, 'test.json')
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                item_ref = json.load(f)
        else:
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                if os.path.isfile(file_path) and filename != 'generated_predictions.json':
                    with open(file_path, 'r') as f:
                        item_ref = json.load(f)
                        item_ref = item_ref[250:510]
        item_ref = [
            [data_item['Summary_1'], data_item['Summary_2'], data_item['Summary_3'], data_item['Summary_4']]
            for data_item in item_ref]
        return item_ref


class GPTeval:
    @staticmethod
    # Load ref,pred
    def load_pred_ref(path, bart=False):
        predictions, references = [], []
        # Load train data (same with bart)
        ref_file_path = os.path.join(path, 'test.json')
        if os.path.exists(ref_file_path):
            # Load reference
            with open(ref_file_path, 'r') as f:
                references = json.load(f)
            references = [data_item['Summary_1'] for data_item in references]
            # Load prediction
            pred_type = 'gpt3_summary_test.json'
            if bart:
                pred_type = 'bart_summary.json'
            pred_file_path = os.path.join(path, 'summary/' + pred_type)
            with open(pred_file_path, 'r') as f:
                predictions = json.load(f)
        else:
            ref_files = ['max.json', 'mean.json', 'min.json']
            ref_file_paths = [os.path.join(path, ref_file) for ref_file in ref_files]
            for index, ref_file_path in enumerate(ref_file_paths):
                if os.path.exists(ref_file_path):
                    # Load reference
                    with open(ref_file_path, 'r') as f:
                        references = json.load(f)
                    references = [data_item['Summary_1'] for data_item in references]
                    references = references[250:510]
                    # Load prediction
                    pred_type = 'gpt3_summary_' + ref_files[index]
                    if bart:
                        pred_type = 'bart_summary.json'
                    pred_file_path = os.path.join(path, 'summary/' + pred_type)
                    with open(pred_file_path, 'r') as f:
                        predictions = json.load(f)
                    if not bart:
                        predictions = predictions[250:510]
        return predictions, references

# waiting to process: oracle LLM-embedding sparse —— bart rel
Evaluate.traverse_path('SQuALITY/oracle', gpteval=True, bart=True)
