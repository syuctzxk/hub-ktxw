import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
# pip install peft
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
import torch


# 数据加载和预处理
import json


def build_ner_examples(sentence_path, tag_path, output_path=None):
    """
    将 sentence.txt + tag.txt 转换为 LoRA 微调用的 instruction-style NER 样本
    """
    examples = []
    with open(sentence_path, 'r', encoding='utf-8') as f_sent, \
         open(tag_path, 'r', encoding='utf-8') as f_tag:
        sentences = [line.strip() for line in f_sent if line.strip()]
        tags = [line.strip().split() for line in f_tag if line.strip()]

    assert len(sentences) == len(tags), "句子数与标签数不一致！"

    for sent, tag_seq in zip(sentences, tags):
        entities = []
        current_entity = ""
        current_type = None

        # 遍历字符与标签
        for ch, tag in zip(sent, tag_seq):
            if tag.startswith("B-"):
                # 开始新实体
                if current_entity:
                    entities.append({"实体": current_entity, "类型": current_type})
                current_entity = ch
                current_type = tag.split("-")[1]
            elif tag.startswith("I-") and current_type == tag.split("-")[1]:
                current_entity += ch
            else:
                # O 或者标签不匹配，结束当前实体
                if current_entity:
                    entities.append({"实体": current_entity, "类型": current_type})
                    current_entity = ""
                    current_type = None

        # 最后一个实体别忘了加进去
        if current_entity:
            entities.append({"实体": current_entity, "类型": current_type})

        # 构造 instruction 格式
        example = {
            "instruction": "请从以下文本中抽取所有命名实体，输出实体名称及其类型（人名PER、地点LOC、机构ORG）：",
            "input": sent,
            "output": json.dumps(entities, ensure_ascii=False)
        }
        examples.append(example)

    # 保存为 JSON 格式
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            json.dump(examples, f_out, ensure_ascii=False, indent=2)

        print(f"✅ 已生成 {len(examples)} 条样本，保存至 {output_path}")

    return examples


# 初始化tokenizer和模型
def initialize_model_and_tokenizer(model_path):
    """初始化tokenizer和模型"""
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True,
        local_files_only=True
    )

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,  # 使用半精度减少内存占用
        device_map=None,
        local_files_only=True
    )

    return tokenizer, model


# 数据处理函数
def process_func(example, tokenizer, max_length=384):
    """
    处理单个样本（用于LoRA微调大模型的实体识别任务）
    每个样本包含文本输入和对应的实体标注结果
    """
    # 示例输入：
    # example = {
    #     "instruction": "请从以下文本中抽取实体：",
    #     "input": "阿里巴巴集团总部位于杭州。",
    #     "output": "[{'实体': '阿里巴巴集团', '类型': 'ORG'}, {'实体': '杭州', '类型': 'LOC'}]"
    # }

    # ===== 系统提示词部分 =====
    instruction_text = (
        "<|im_start|>system\n"
        "进行中文命名实体识别任务，输出 JSON 数组 [{'实体': 'XXX', '类型': 'YYY'}]\n"
        "<|im_end|>\n"
    )

    # ===== 用户输入部分 =====
    user_prompt = f"<|im_start|>user\n{example['instruction']} {example['input']}<|im_end|>\n<|im_start|>assistant\n"

    # 拼接完整的 instruction prompt
    full_prompt = instruction_text + user_prompt

    # Tokenize 输入部分（system + user）
    instruction = tokenizer(full_prompt, add_special_tokens=False)

    # Tokenize 输出部分（模型应答）
    response = tokenizer(f"{example['output']}", add_special_tokens=False)

    # 拼接最终序列
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]

    # 构造 labels：system/user 部分的 token 忽略（标 -100），只在 assistant 部分计算 loss
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]

    # 截断
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }



# 配置LoRA
def setup_lora(model):
    """设置LoRA配置并应用到模型"""
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    return model


# 训练配置
def setup_training_args():
    """设置训练参数"""
    return TrainingArguments(
        output_dir="./output_Qwen1.5_ner",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=100,
        do_eval=True,
        eval_steps=50,
        num_train_epochs=5,
        save_steps=50,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none"  # 禁用wandb等报告工具
    )


# 预测函数
def predict_intent(model, tokenizer, text, device='cpu'):
    """预测单个文本的意图"""
    messages = [
        {"role": "system", "content": "现在进行意图分类任务"},
        {"role": "user", "content": text}
    ]

    # 应用聊天模板
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize输入
    model_inputs = tokenizer([formatted_text], return_tensors="pt").to(device)

    # 生成预测
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,  # 降低温度以获得更确定的输出
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 提取生成的文本（去掉输入部分）
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response.strip()


# 批量预测
def batch_predict(model, tokenizer, test_texts, device='cuda'):
    """批量预测测试集的意图"""
    pred_labels = []

    for text in tqdm(test_texts, desc="预测意图"):
        try:
            pred_label = predict_intent(model, tokenizer, text, device)
            pred_labels.append(pred_label)
        except Exception as e:
            print(f"预测文本 '{text}' 时出错: {e}")
            pred_labels.append("")  # 出错时添加空字符串

    return pred_labels


# 主函数
def main():
    """主执行函数"""
    # 1. 加载数据
    print("加载数据...")
    # 示例调用
    train_examples = build_ner_examples("/Users/bimeiqiao/develop/bd_home_work/Week07/Week07/msra/train/sentences.txt",
                       "/Users/bimeiqiao/develop/bd_home_work/Week07/Week07/msra/train/tags.txt",
                       "/Users/bimeiqiao/develop/bd_home_work/Week07/Week07/msra/train/train_ner_instruction_data.json")
    test_examples = build_ner_examples("/Users/bimeiqiao/develop/bd_home_work/Week07/Week07/msra/test/sentences.txt",
                       "/Users/bimeiqiao/develop/bd_home_work/Week07/Week07/msra/test/tags.txt",
                       "/Users/bimeiqiao/develop/bd_home_work/Week07/Week07/msra/test_ner_instruction_data.json")

    # 2. 初始化模型和tokenizer
    print("初始化模型和tokenizer...")
    model_path = "/Users/bimeiqiao/develop/bd_home_work/Week07/Week07/models/Qwen/Qwen3-0.6B"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 3. 处理数据
    print("处理训练数据...")
    process_func_with_tokenizer = lambda example: process_func(example, tokenizer)
    # 4. 划分训练集和验证集
    train_ds = Dataset.from_list(train_examples[:400])
    train_tokenized = train_ds.map(process_func_with_tokenizer)

    eval_ds = Dataset.from_list(test_examples[-200:])
    eval_tokenized = eval_ds.map(process_func_with_tokenizer)

    # 5. 设置LoRA
    print("设置LoRA...")
    model.enable_input_require_grads()
    model = setup_lora(model)

    # 6. 配置训练参数
    print("配置训练参数...")
    training_args = setup_training_args()

    # 7. 创建Trainer并开始训练
    print("开始训练...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8  # 优化GPU内存使用
        ),
    )

    trainer.train()

    # 8. 保存模型
    print("保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained("./output_Qwen1.5_ner")


# 单独测试函数
def test_single_example():
    # 下载模型
    # modelscope download --model Qwen/Qwen3-0.6B  --local_dir Qwen/Qwen3-0.6B
    model_path = "models/Qwen/Qwen3-0.6B/"
    model_path = "/Users/bimeiqiao/develop/bd_home_work/Week07/Week07/models/Qwen/Qwen3-0.6B"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 加载训练好的LoRA权重
    model.load_adapter("./output_Qwen1.5_ner/")
    model.cpu()

    # 测试预测
    test_text = "去北京如何走？"
    result = predict_intent(model, tokenizer, test_text)
    print(f"输入: {test_text}")
    print(f"预测意图: {result}")


if __name__ == "__main__":

    # 执行主函数
    result_df = main()

    # 单独测试
    # test_single_example()
