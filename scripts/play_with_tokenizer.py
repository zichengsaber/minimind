import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json

def check_pretrained_data():
    # 获取当前脚本的绝对路径（如 /path/to/project/scripts/main.py）
    def read_texts_from_jsonl(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line) # only has text field
                yield data['text']

    script_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.normpath(os.path.join(script_path, '../dataset/minimind_dataset/pretrain_hq.jsonl'))
    texts = read_texts_from_jsonl(data_path)
    # print yeield 的内容 5 lines
    i = 0
    for text in texts:
        print(text)
        if i > 5:
            break
        i += 1

def eval_tokenizer(name:str):
    from transformers import AutoTokenizer
    # 获取当前脚本的绝对路径（如 /path/to/project/scripts/main.py）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 计算模型目录的路径：脚本目录的上级目录下的 model 文件夹（即 /path/to/project/model）
    model_dir = os.path.normpath(os.path.join(script_dir, f"../{name}"))
    # 加载预训练的tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        # add_generation_prompt=True
    )
    print(new_prompt)

    # 获取实际词汇表长度（包括特殊符号）
    actual_vocab_size = len(tokenizer)
    print('tokenizer实际词表长度：', actual_vocab_size)

    model_inputs = tokenizer(new_prompt)
    print(model_inputs)
    print('encoder长度：', len(model_inputs['input_ids']))
    # keys : input_ids, attention_mask, token_type_ids
    input_ids = model_inputs['input_ids']
    response = tokenizer.decode(input_ids, skip_special_tokens=False)
    print(response)
    print('decoder和原始文本是否一致：', response == new_prompt)

if __name__ == "__main__":
    eval_tokenizer('tokenizer')
    # check_pretrained_data()