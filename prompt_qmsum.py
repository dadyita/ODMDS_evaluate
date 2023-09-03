import json
from PACKAGE import asyncThread, metric_realization, multi_rouge
import os
import bert_score
from tqdm import tqdm
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain

api_key = ''


# llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k-0613", openai_api_key=api_key, temperature=0.7, max_tokens=600)


class Evaluate:
    @staticmethod
    def load_pred(path, model_name):
        predictions = []
        pred_file = model_name + '_summary.json'
        file_path = os.path.join(path, 'summary/' + pred_file)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                predictions = json.load(f)
        return predictions

    @staticmethod
    def load_ref(path):
        ref_file = 'test.json'
        references = []
        file_path = os.path.join(path, ref_file)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                references = json.load(f)
            references = [
                data_item['Summary']
                for data_item in references]
        return references

    @staticmethod
    def squality_rouge(path, predictions, references, model_name):
        # Calculate
        print('Evaluate rouge score (use squality)')
        rouge_object = multi_rouge.Rouge()
        squality_rouge_score = rouge_object._compute(predictions=predictions, references=references)
        # Save
        file_name = model_name + '_squality_rouge.json'
        file_path = os.path.join(path, 'evaluation/' + file_name)
        with open(file_path, 'w') as f:
            f.write(str(squality_rouge_score))

    @staticmethod
    def bert(path, predictions, references, model_name):
        print('Evaluate bert score')
        # 批次大小
        batch_size = 256
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
        # Save
        file_name = model_name + '_bert_score.json'
        file_path = os.path.join(path, 'evaluation/' + file_name)
        with open(file_path, 'w') as f:
            temp = json.dumps(bert_scores)
            f.write(temp)

    @staticmethod
    def another_rouge(path, predictions, references, model_name):
        print('Evaluate rouge score (use another way)')
        rouge_score = metric_realization.calculate_rouge(ref=references, pred=predictions)
        # Save
        file_name = model_name + '_rouge.json'
        file_path = os.path.join(path, 'evaluation/' + file_name)
        with open(file_path, 'w') as f:
            temp = json.dumps(rouge_score)
            f.write(temp)

    @staticmethod
    def bleurt(path, predictions, references,model_name):
        print('Evaluate bleurt score')
        rouge_score = metric_realization.calculate_bert_score(ref=references, pred=predictions)
        file_name = model_name + '_bleurt.json'
        file_path = os.path.join(path, 'evaluation/' + file_name)
        with open(file_path, 'w') as f:
            temp = json.dumps(rouge_score)
            f.write(temp)

    @staticmethod
    def evaluate(path, model_name, bert=False, rouge=False, another_rouge=False, bleurt=False):
        # Load predictions
        predictions = Evaluate.load_pred(path, model_name)

        # Load references
        references = Evaluate.load_ref(path)

        # Delete empty
        references = [references[index] for index, item in enumerate(predictions) if item != '']
        predictions = [predictions[index] for index, item in enumerate(predictions) if item != '']

        if rouge:
            Evaluate.squality_rouge(path, predictions, references, model_name)
        if another_rouge:
            Evaluate.another_rouge(path, predictions, references,model_name)
        if bert:
            Evaluate.bert(path, predictions, references, model_name)
        if bleurt:
            Evaluate.bleurt(path, predictions, references, model_name)

    @staticmethod
    def traverse_path(root, model_name, bert=False, rouge=False, another_rouge=False, bleurt=False):
        for path, dirs, files in os.walk(root):
            if files and dirs:
                Evaluate.evaluate(path, model_name, bert, rouge, another_rouge, bleurt)

    @staticmethod
    def print_score(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file == 'evaluate_squality_rouge.json':
                    with open(os.path.join(root, file), 'r') as f:
                        rouge = f.read()
                        obj_rouge = eval(rouge)
                        print(root)
                        print(f"rouge1:\n{obj_rouge['rouge1'].mid.fmeasure * 100:.2f}")
                        print(f"rouge2:\n{obj_rouge['rouge2'].mid.fmeasure * 100:.2f}")
                        print(f"rougeL:\n{obj_rouge['rougeL'].mid.fmeasure * 100:.2f}")


class Summarize:
    @staticmethod
    def split_string(string, num_segments):
        # 计算每段的大小，整除操作防止有余数
        # 取上整
        chunk_size = 1 + len(string) // num_segments
        chunks = []

        for i in range(0, len(string), chunk_size):
            chunks.append(string[i:i + chunk_size])

        return chunks

    @staticmethod
    def split_meeting(articles):
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k-0613", openai_api_key=api_key, temperature=0.7, max_tokens=600)
        tokens = [llm.get_num_tokens(article) for article in articles]

        split_counts = [1 + token // 14000 for token in tokens]

        split_meetings = []
        for index, article in enumerate(articles):
            split_meetings.append(Summarize.split_string(article, split_counts[index]))
        return tokens, split_meetings

    @staticmethod
    def make_split_meeting_files():
        path = 'QMSum/dense'
        wait_process_files = ['Academic.json', 'Committee.json', 'Product.json']
        for root, dirs, files in os.walk(path):
            # 遍历当前目录下的文件
            for file_name in files:
                # If it is origin file
                if file_name not in wait_process_files:
                    continue

                # 检测有没有创建split文件夹
                if not os.path.lexists(os.path.join(root, 'split')):
                    os.makedirs(os.path.join(root, 'split'))

                # Load data
                with open(os.path.join(root, file_name), 'r') as f:
                    data = json.load(f)
                    articles = [item['Article'] for item in data]
                tokens, split_meetings = Summarize.split_meeting(articles)

                # Write meetings
                with open(os.path.join(root, 'split/split_' + file_name), 'w') as f:
                    print(root)
                    temp = json.dumps(split_meetings, indent=4)
                    f.write(temp)

    @staticmethod
    def traverse_path(folder_path):
        # folder_path 指定要遍历的文件夹路径
        # folder_path = "SQuALITY/sparse/min"
        wait_process_files = ['max.json', 'mean.json', 'min.json', 'Academic.json', 'Committee.json', 'Product.json',
                              'dev.json', 'test.json', 'train.json']
        for root, dirs, files in os.walk(folder_path):
            # 遍历当前目录下的文件夹
            # for dir_name in dirs:
            #     print("文件夹：", os.path.join(root, dir_name))
            # 遍历当前目录下的文件
            for file_name in files:
                # If it is origin file
                if file_name not in wait_process_files:
                    continue

                Summarize.traverse_sub_path(root)
                break

    @staticmethod
    def traverse_sub_path(path):
        filenames = ['Academic.json',
                     'Committee.json', 'Product.json']
        for filename in filenames:
            print(path)
            print(filename)
            with open(os.path.join(path, filename), 'r') as f:
                data = json.load(f)
                queries = [item['Query'] for item in data]
            with open(os.path.join(path, 'split/split_' + filename), 'r') as f:
                articles = json.load(f)

            if os.path.exists(os.path.join(path, 'summary/gpt3_summary_' + filename)):
                with open(os.path.join(path, 'summary/gpt3_summary_' + filename), 'r') as f:
                    data = json.load(f)
                processed_count = len(data)
            else:
                processed_count = 0
            # 记录分割的数量，如果小于10则继续提交
            # split_text_count = 0

            save_intermediate_outputs = []
            save_final_output = []

            for index, article in enumerate(articles):
                # 根据传输的token数控制休眠时间
                # if split_text_count != 0 and split_text_count + len(article) > 10:
                #     sleep_time = int(split_text_count / 10 * 60)
                #     print(f"sleep:{sleep_time} seconds")
                #     time.sleep(sleep_time)
                #     split_text_count = len(article)
                # else:
                #     split_text_count += len(article)

                # 运行并处理
                query = queries[index]
                intermediate_outputs = Summarize.intermediate_summary(query, article)
                final_output = Summarize.final_summary(query, intermediate_outputs)
                save_intermediate_outputs.append(intermediate_outputs)
                save_final_output.append(final_output)

                # 达到一定量则保存
                if len(save_final_output) > processed_count and len(save_final_output) % 5 == 0:
                    with open(os.path.join(path, 'summary/gpt3_intermediate_summary_' + filename), 'w') as f:
                        temp = json.dumps(save_intermediate_outputs, indent=4)
                        f.write(temp)

                    with open(os.path.join(path, 'summary/gpt3_summary_' + filename), 'w') as f:
                        temp = json.dumps(save_final_output, indent=4)
                        f.write(temp)
                if len(save_final_output) == 10:
                    break

    @staticmethod
    def intermediate_summary(query, docs):
        map_prompts = [
            f"Write an answer based on the following question and the given meeting.Try to answer thoroughly and do not leave out useful information.\n QUESTION:{query}\n MEETING:{doc}\n SUMMARY: \n"
            for doc in docs]
        system = "You are a helpful assistant that gives long answer to question based on a long meeting."
        messages = [[{"role": "system", "content": system},
                     {"role": "user", "content": map_prompt}] for map_prompt in map_prompts]
        intermediate_outputs = asyncThread.run(messages=messages,
                                               engine_name="gpt-3.5-turbo-16k-0613",
                                               temperature=0.7,
                                               max_tokens=600,
                                               top_p=0.9,
                                               api_key=api_key,
                                               requests_per_minute=20)

        return intermediate_outputs

    @staticmethod
    def final_summary(query, intermediate_outputs):
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k-0613", openai_api_key=api_key, temperature=0.7, max_tokens=600)
        combine_prompt = """
        Combine the information of the following text together to form a long passage answering the question.
        Try to answer thoroughly and do not leave out useful information.
        QUESTION:
        "{question}"
        TEXT:
        {text}
        SUMMARY:
        """

        combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text", "question"])
        combine_llm_chain = LLMChain(llm=llm, prompt=combine_prompt_template)

        feed_text = "\n".join(intermediate_outputs)

        output = combine_llm_chain.run({
            'text': feed_text,
            'question': query
        })
        return output


Evaluate.traverse_path('QMSum', model_name='bart', another_rouge=True, bert=True)
