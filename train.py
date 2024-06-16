import logging
from dataclasses import dataclass
from typing import Optional, List

import torch
from peft import LoraConfig
from datasets import disable_caching, load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    HfArgumentParser,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

disable_caching()

logger = logging.getLogger(__name__)


@dataclass
class SFTTrainingArguments:
    model_name_or_path: str
    data_files: List[str]
    eval_data_files: List[str] = None
    tokenizer_name_or_path: Optional[str] = None
    use_fast: bool = True
    additional_special_tokens: List[str] = None
    max_seq_length: int = 2048
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention_2: bool = False
    use_peft: bool = False
    peft_target_model: Optional[str] = "llm-jp"
    peft_target_modules: Optional[List[str]] = None
    peft_lora_r: int = 8
    peft_lora_alpha: int = 32
    peft_lora_dropout: float = 0.05

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("load_in_8bit and load_in_4bit are mutually exclusive")
        if self.peft_target_model and self.peft_target_modules is None:
            if self.peft_target_model == "llm-jp":
                self.peft_target_modules = ["c_attn", "c_proj", "c_fc"]
            elif self.peft_target_model == "llama":
                # https://github.com/serp-ai/LLaMA-8bit-LoRA/blob/main/finetune_peft_8bit.py
                self.peft_target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
            elif self.peft_target_model == "llama-all":
                # https://note.com/kan_hatakeyama/n/ncd09c52d26c7
                self.peft_target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head",
                    "embed_tokens",
                ]
            else:
                logger.warning(
                    f"peft_target_model '{self.peft_target_model}' is not supported, "
                    f"so peft_target_modules is set to None."
                )

    def from_pretrained_kwargs(self, training_args):
        if self.load_in_8bit:
            kwargs = {"load_in_8bit": True}
        elif self.load_in_4bit:
            kwargs = {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            }
        elif training_args.bf16:
            kwargs = {"torch_dtype": torch.bfloat16}
        else:
            kwargs = {"torch_dtype": torch.float16}
        kwargs["use_flash_attention_2"] = self.use_flash_attention_2
        return kwargs

@dataclass
class TrainingArgumentsWrap(TrainingArguments):
    per_device_train_batch_size: int=1024
    logging_steps:int = 10
    lr_scheduler_type: str = 'constant'
    save_total_limit: int = 3
    save_strategy: str = "steps"
    save_steps: int = 50
    num_train_epochs: int = 1
    gradient_checkpointing: bool = False

def format(ds):
    if 'query' in ds and 'answer':
        text = ds['query']
        output = ds['answer']
    
    if 'instruction' in ds and 'output' in ds:
        if 'input' in ds:
            text = ds['instruction'] + "\n" + ds["input"]
        else:
            text = ds['instruction']
        output = ds['output']

    if 'question_ja' in ds and 'generated_solution_ja' in ds:
        text = ds['question_ja']
        output = ds['generated_solution_ja']

    if ('q1' in ds) and ('a1' in ds) and ('q2' in ds) and ('a2' in ds):
        text = ds['q1'] + "\n" \
        + "### アシスタント:\n" \
        + ds['a1'] + "\n" \
        + "### ユーザー:\n" \
        + ds['q2']  + "\n\n"
        output = ds['a2']

    if text is None or output is None:
        return None
    prompt = f"""### ユーザー:
{text}

### アシスタント:
{output}
"""
    return dict({"text": prompt})

def load_datasets(data_files):
    datasets = []
    for data_file in data_files:
        if 'json' in data_file:
            dataset = load_dataset("json", data_files=data_file, split="train")
        else:
            dataset = load_dataset(data_file, split="train")
        _new = []
        for v in dataset:
            _v  = format(v)
            if _v:
                _new.append(_v)
        _dataset = Dataset.from_list(_new)
        datasets.append(_dataset)
    return concatenate_datasets(datasets).shuffle(seed=42)


def main() -> None:
    parser = HfArgumentParser((TrainingArgumentsWrap, SFTTrainingArguments))
    training_args, sft_training_args = parser.parse_args_into_dataclasses()
    print('training_args', training_args)

    tokenizer_name_or_path: str = (
        sft_training_args.tokenizer_name_or_path or sft_training_args.model_name_or_path
    )
    logger.info(f"Loading tokenizer from {tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=sft_training_args.use_fast,
        additional_special_tokens=sft_training_args.additional_special_tokens,
        trust_remote_code=True,
    )

    logger.info("Loading data")

    dataset = load_datasets(sft_training_args.data_files)
    dataset = dataset.train_test_split(test_size=0.01)
    train_dataset = dataset['train']
    eval_dataset =dataset['test']

    logger.info("Formatting prompts")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


    logger.info(f"Loading model from {sft_training_args.model_name_or_path}")
    kwargs = sft_training_args.from_pretrained_kwargs(training_args)
    logger.debug(
        f"AutoModelForCausalLM.from_pretrained({sft_training_args.model_name_or_path}, trust_remote_code=True, **kwargs={kwargs})"
    )
    model = AutoModelForCausalLM.from_pretrained(
        sft_training_args.model_name_or_path,
        trust_remote_code=True,
        **kwargs,
    )
    model.to('cuda')
    peft_config: Optional[LoraConfig] = None
    if sft_training_args.use_peft:
        logger.info("Setting up LoRA")
        peft_config = LoraConfig(
            r=sft_training_args.peft_lora_r,
            target_modules=sft_training_args.peft_target_modules,
            lora_alpha=sft_training_args.peft_lora_alpha,
            lora_dropout=sft_training_args.peft_lora_dropout,
            fan_in_fan_out=True,
            bias="none",
            task_type="CAUSAL_LM",
        )
        if training_args.gradient_checkpointing:
            for param in model.parameters():
                param.requires_grad = False
                if param.ndim == 1:
                    param.data = param.data.to(torch.float32)
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

    logger.info("Setting up trainer")
    trainer = SFTTrainer(
        model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        data_collator=data_collator,
        peft_config=peft_config,
        max_seq_length=sft_training_args.max_seq_length,
        num_of_sequences=512,
        packing=True
    )

    logger.info("Training")
    trainer.train()

    logger.info("Saving model")
    trainer.save_model()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()