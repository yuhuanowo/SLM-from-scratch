import datasets
import transformers
import wandb
from transformers.integrations import WandbCallback
import os
import torch
from unsloth import UnslothTrainer, UnslothTrainingArguments
import time
from transformers import EarlyStoppingCallback
import multiprocessing
from datasets import Dataset, load_from_disk, disable_caching
import logging
import numpy as np
from tqdm import tqdm
import argparse 

# 設置日誌級別
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# 禁用HF datasets的緩存，防止大數據集造成緩存爆滿
disable_caching()

def main():
    # 添加命令列參數解析
    parser = argparse.ArgumentParser(description="YuLLM預訓練與持續預訓練")
    parser.add_argument("--continuous_pretraining", action="store_true", 
                       help="啟用持續預訓練模式，從現有模型繼續訓練")
    parser.add_argument("--base_model", type=str, default="",
                       help="持續預訓練時的基礎模型路徑")
    args = parser.parse_args()

    # 檢查是否為持續預訓練模式
    continuous_pretraining = args.continuous_pretraining
    base_model = args.base_model if args.base_model else None
    
    if continuous_pretraining:
        if not base_model:
            logger.info("已啟用持續預訓練模式，但未指定基礎模型，將使用default設置")
        else:
            logger.info(f"已啟用持續預訓練模式，基礎模型: {base_model}")
    else:
        logger.info("從零開始訓練新模型")
    
    # 檢查是否存在先前的模型checkpoint
    output_dir = "YuLLM_small_50M_v1"
    last_checkpoint = None

     # 設定 GPU 記憶體使用率為 90%
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)
        logger.info("已設定 GPU 記憶體使用率上限為 90%")
    
    
    if os.path.exists(output_dir):
        checkpoints = [f for f in os.listdir(output_dir) if f.startswith("checkpoint-")]
        if checkpoints:
            last_checkpoint = os.path.join(output_dir, sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1])
            print(f"找到先前的checkpoint: {last_checkpoint}")
    
    # 使用 wandb 替換 swanlab 初始化
    wandb.init(
        dir='./',
        project="YuLLM_small_50M_v1",
        resume="allow",  # 允許繼續之前的實驗
        name="YuLLM_small_50M_v1",
    )

    # 高效率載入大型數據集
    logger.info("開始載入數據集...")
    data_files = [os.path.join("dataset", f) for f in os.listdir("dataset") if f.endswith(".json")]
    
    # 檢查是否有緩存處理過的數據集
    processed_data_path = "./processed_dataset"
    if (os.path.exists(processed_data_path)):
        logger.info(f"使用預處理緩存資料: {processed_data_path}")
        try:
            tokenized_datasets = load_from_disk(processed_data_path)
            print("成功載入預處理資料集")
            print("數據集資訊:")
            print(tokenized_datasets)
        except Exception as e:
            logger.warning(f"無法載入緩存資料，重新處理: {str(e)}")
            tokenized_datasets = process_raw_datasets(data_files)
    else:
        logger.info("未找到預處理資料，開始處理原始資料...")
        tokenized_datasets = process_raw_datasets(data_files)
        # 保存處理後的數據集
        tokenized_datasets.save_to_disk(processed_data_path)
        logger.info(f"已保存預處理資料到: {processed_data_path}")
        
    # 配置訓練參數
    args = configure_training_args()
    
    # 載入或創建模型，传入持续预训练参数
    model, tokenizer = prepare_model(continuous_pretraining=continuous_pretraining, base_model=base_model)
    
    # 設定數據整理器
    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=16  # 提高GPU效率
    )

    from transformers.trainer_callback import TrainerCallback
    
    # 创建自定义内存清理回调类
    class MemoryCleanupCallback(TrainerCallback):
        """自定义回调类，用于定期清理内存"""
        
        def on_step_end(self, args, state, control, **kwargs):
            """每隔一定步数执行内存清理"""
            # 每隔100步清理一次内存
            if state.global_step % 500 == 0:
                logger.info(f"步骤 {state.global_step}: 执行内存清理...")
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                logger.info("内存清理完成")
            return control
        
        # 準備回調
    callbacks = [
        WandbCallback(),  # 使用 WandbCallback 替換 SwanLabCallback
        EarlyStoppingCallback(early_stopping_patience=3),  # 添加早停 
        MemoryCleanupCallback()  # 添加记忆体清理回调
    ]
    # 設置訓練器
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        callbacks=callbacks,
    )
    
    # 回收記憶體
    torch.cuda.empty_cache()
    #回收ram
    import gc
    gc.collect()
    # 從最後的checkpoint繼續訓練
    logger.info(f"開始訓練，{'從checkpoint繼續' if last_checkpoint else '從頭開始'}")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # 儲存模型
    logger.info("訓練完成，保存最終模型...")
    model.save_pretrained(f"./{output_dir}")
    tokenizer.save_pretrained(f"./{output_dir}/tokenizer")


def process_raw_datasets(data_files):
    """處理原始數據集，包含數據清洗、標記化等步驟"""
    logger.info(f"處理 {len(data_files)} 個數據文件...")
    
    # 確保有資料文件
    if not data_files:
        raise ValueError("沒有找到任何資料文件！請確認 dataset 目錄中有 .json 文件。")
    
    # 載入原始數據
    raw_datasets = datasets.load_dataset("json", data_files=data_files)
    # 保留需要的欄位
    columns_to_keep = ['text']
    columns_to_remove = [col for col in raw_datasets['train'].column_names if col not in columns_to_keep]
    raw_datasets = raw_datasets.remove_columns(columns_to_remove)
    # 數據集分割
    raw_datasets = raw_datasets["train"].train_test_split(test_size=0.01, seed=2333)
    print("數據集基本信息:")
    print(raw_datasets)
    
    # 檢查數據集是否為空
    if len(raw_datasets["train"]) == 0:
        raise ValueError("數據集為空！請檢查數據文件格式是否正確。")
    
    # 數據清洗
    logger.info("進行數據清洗...")
    raw_datasets = clean_datasets(raw_datasets)
    
    # 確保清洗後的數據集非空
    if len(raw_datasets["train"]) == 0:
        raise ValueError("數據清洗後，訓練集為空！請檢查清洗條件是否過於嚴格。")
    
    # 載入分詞器
    logger.info("載入 Gemma 2 分詞器...")
    context_length = 2048
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("google/gemma-2b")
        logger.info("成功載入 google/gemma-2b tokenizer")
    except Exception as e:
        logger.warning(f"載入 google/gemma-2b tokenizer 失敗: {str(e)}")
        logger.info("嘗試載入本地 gemma_tokenizer...")
        tokenizer = transformers.AutoTokenizer.from_pretrained("gemma_tokenizer")
    
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    
    # 標記化處理
    logger.info("開始標記化處理...")
    
    # 定義標記化函數
    def tokenize(element):
        # 優化標記化流程以減少記憶體使用
        texts_to_process = []
        if isinstance(element["text"], list):
            # 扁平化處理嵌套列表，避免多次迭代
            texts_to_process = [item for sublist in element["text"] if isinstance(sublist, list) for item in sublist] + \
                              [item for item in element["text"] if isinstance(item, str)]
        elif isinstance(element["text"], str):
            texts_to_process = [element["text"]]
        
        # 批量過濾無效文本，減少後續處理
        texts_to_process = [text.strip() for text in texts_to_process if isinstance(text, str) and text.strip()]
        if not texts_to_process:
            return {"input_ids": [], "attention_mask": []}
        
 
        
        # 確保特殊標記已設置
        if tokenizer.bos_token is None:
            tokenizer.bos_token = "<s>"
        if tokenizer.eos_token is None:
            tokenizer.eos_token = "</s>"

        # 為每個文本添加標記並預處理
        formatted_texts = []
        
        # 首先確保文本是可迭代的
        texts_to_process = []
        if isinstance(element["text"], list):
            # 如果已經是列表
            for item in element["text"]:
                if isinstance(item, list):
                    # 處理嵌套列表
                    texts_to_process.extend(item)
                elif isinstance(item, str):
                    texts_to_process.append(item)
        elif isinstance(element["text"], str):
            # 如果是單個字符串
            texts_to_process = [element["text"]]
        else:
            # 無效類型
            return {"input_ids": [], "attention_mask": []}
        
        # 處理收集到的所有文本
        for text in texts_to_process:
            if not isinstance(text, str) or not text.strip():
                continue
            
            # 為每個文本單獨添加標記，不混合不同文本
            formatted_text = tokenizer.bos_token + text.strip() + tokenizer.eos_token
            formatted_texts.append(formatted_text)
        
        if not formatted_texts:  # 如果沒有有效文本，返回空列表
            return {"input_ids": [], "attention_mask": []}
        
        # 標記化處理，不要在這裡截斷
        tokenized_dict = tokenizer(
            formatted_texts,
            truncation=False,  # 不截斷，我們將手動處理
            add_special_tokens=False,  # 我們已經添加了特殊標記
        )
        
        # 將短序列串聯起來
        all_input_ids = []
        all_attention_masks = []
        current_sequence = []
        current_mask = []
        
        for ids, mask in zip(tokenized_dict["input_ids"], tokenized_dict["attention_mask"]):
            # 如果當前序列加上新序列會超過最大長度
            if len(current_sequence) + len(ids) > context_length:
                # 如果當前序列非空且足夠長，則保存它
                if len(current_sequence) >= 25:
                    all_input_ids.append(current_sequence)
                    all_attention_masks.append(current_mask)
                
                # 開始一個新序列
                current_sequence = ids
                current_mask = mask
            else:
                # 添加到當前序列
                if not current_sequence:  # 如果是空序列
                    current_sequence = ids
                    current_mask = mask
                else:
                    current_sequence.extend(ids)
                    current_mask.extend(mask)
        
        # 別忘了處理最後一個序列
        if len(current_sequence) >= 25:
            all_input_ids.append(current_sequence)
            all_attention_masks.append(current_mask)
        
        # 處理過長序列（如果有）
        input_batch = []
        attention_masks = []
        
        for ids, mask in zip(all_input_ids, all_attention_masks):
            # 截斷到最大長度
            if len(ids) > context_length:
                ids = ids[:context_length]
                mask = mask[:context_length]
            
            # 檢查序列包含起始標記
            if tokenizer.bos_token_id in ids[:3]:  # 檢查前幾個標記
                input_batch.append(ids)
                attention_masks.append(mask)
        
        logger.debug(f"從 {len(formatted_texts)} 個文本創建了 {len(input_batch)} 個訓練樣本")
        #輸出一個樣本用來檢查
        
        return {"input_ids": input_batch, "attention_mask": attention_masks}
    
    # 使用多進程加速標記化
    num_proc = max(1, multiprocessing.cpu_count() - 1)  # 保留一個核心
    tokenized_datasets = raw_datasets.map(
        tokenize, 
        batched=True, 
        num_proc=num_proc,
        remove_columns=raw_datasets["train"].column_names,
        desc="標記化處理中"
    )
    
    # 過濾無效樣本
    logger.info("過濾無效樣本...")
    tokenized_datasets = tokenized_datasets.filter(
        lambda example: len(example["input_ids"]) > 0,
        num_proc=num_proc,
        desc="過濾樣本中"
    )
    
    # 對過大的數據集進行採樣
    if len(tokenized_datasets["train"]) > 10000000:  # 調整為合理的大小限制
        logger.info(f"訓練集太大 ({len(tokenized_datasets['train'])}條)，進行採樣...")
        tokenized_datasets["train"] = tokenized_datasets["train"].select(range(10000000))
    
    logger.info(f"標記化後數據統計: 訓練集 {len(tokenized_datasets['train'])} 條，測試集 {len(tokenized_datasets['test'])} 條")

    #輸出一條樣本用來檢查
    logger.info(f"樣本示例: {tokenized_datasets['train'][0]}")
    return tokenized_datasets


def clean_datasets(datasets_dict):
    """增強的數據清洗函數"""
    import re
    from datasets import DatasetDict
    
    # 過濾標準
    min_chars = 25
    max_chars = 10000
    min_words = 10
    max_words = 200000
    min_word_len_ratio = 0.4  # 有意義單詞占比
    max_repitition_ratio = 0.3  # 重複內容最大比例
    
    # 編譯常用正則表達式
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    email_pattern = re.compile(r'\S+@\S+\.\S+')
    repetitive_pattern = re.compile(r'(.{10,}?)\1{3,}')  # 檢測重複內容
    
    def clean_text(example):
        """增強的文本清理函數，處理各種類型的輸入"""
        text = example["text"]
        cleaned_texts = []
        
        # 處理文本列表或字符串，包含嵌套列表的情況
        texts_to_process = []
        if isinstance(text, list):
            for item in text:
                if isinstance(item, list):
                    texts_to_process.extend(item)
                elif isinstance(item, str):
                    texts_to_process.append(item)
        elif isinstance(text, str):
            texts_to_process = [text]
        
        for t in texts_to_process:
            if not isinstance(t, str) or not t.strip():
                continue
                
            # 基本清理
            t = t.strip()

            # 過濾太短或太長的文本
            if len(t) < min_chars or len(t) > max_chars:
                continue
            
            # 計算字詞數量
            words = t.split()
            if len(words) < min_words or len(words) > max_words:
                continue

            # 過濾URL和電子郵件過多的文本
            url_count = len(url_pattern.findall(t))
            email_count = len(email_pattern.findall(t))
            if url_count > 5 or email_count > 3:
                continue
                
            # 檢查重複內容
            if repetitive_pattern.search(t):
                continue
                
            # 有意義單詞檢查 (單詞長度>2的占比)
            meaningful_words = [w for w in words if len(w) > 2]
            if len(meaningful_words) / (len(words) + 1e-10) < min_word_len_ratio:
                continue
            
            # 通過所有過濾條件，保留文本
            cleaned_texts.append(t)
            
        example["text"] = cleaned_texts
        return example
    
    # 應用清洗函數
    cleaned_datasets = {}
    for split, dataset in datasets_dict.items():
        logger.info(f"清洗 {split} 數據集...")
        cleaned_datasets[split] = dataset.map(
            clean_text,
            num_proc=max(1, multiprocessing.cpu_count() - 1),
            desc=f"清洗 {split} 集"
        )
        
        # 過濾空文本
        initial_size = len(cleaned_datasets[split])
        cleaned_datasets[split] = cleaned_datasets[split].filter(
            lambda x: len(x["text"]) > 0,
            num_proc=max(1, multiprocessing.cpu_count() - 1)
        )
        final_size = len(cleaned_datasets[split])

        #輸出一條樣本用來檢查
        logger.info(f"樣本示例: {cleaned_datasets[split][0]}")

    
        
        logger.info(f"{split} 集清洗結果: {initial_size} -> {final_size} 樣本, 過濾率: {(initial_size-final_size)/initial_size:.2%}")
    
    # 正確使用 DatasetDict
    return DatasetDict(cleaned_datasets)


def configure_training_args():
    """配置訓練參數"""
    args = UnslothTrainingArguments(
        output_dir="YuLLM_small_50M_v1",
        overwrite_output_dir=True,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=2,
        gradient_accumulation_steps=128, #tatol = batch_size * gradient_accumulation_steps = 2 * 128 = 256
        num_train_epochs=1,
        weight_decay=0.1,
        warmup_ratio=0.05,
        optim="adamw_8bit",
        lr_scheduler_type="cosine_with_restarts",
        learning_rate=5e-4, # 5e-4
        save_steps=500,
        save_total_limit=5,
        dataloader_num_workers=18,
        dataloader_pin_memory=True,
        bf16=True,
        bf16_full_eval=True,
        torch_compile=True,
        # 添加提前停止
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_grad_norm=1.0,
        report_to=["wandb"],
    )
    
    logger.info("訓練參數配置完成")
    return args


def prepare_model(continuous_pretraining=False, base_model=None):
    """使用 Gemma 2 模型配置和分詞器，根据参数决定是创建新模型还是加载现有模型继续训练"""
    logger.info("载入分词器...")
    
    try:
        # 首先尝试从base_model加载分词器（持续预训练模式）
        if continuous_pretraining and base_model:
            logger.info(f"尝试从基础模型加载分词器: {base_model}")
            tokenizer = transformers.AutoTokenizer.from_pretrained(base_model)
        else:
            # 否则使用默认分词器
            logger.info("使用默认分词器...")
            tokenizer = transformers.AutoTokenizer.from_pretrained("gemma_tokenizer")
    except Exception as e:
        logger.warning(f"加载分词器失败: {str(e)}，尝试使用备选方案")
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained("google/gemma-2b")
        except:
            logger.critical("无法加载分词器！")
            raise
    
    tokenizer.pad_token = tokenizer.eos_token
    
    # 測試分詞器
    test_text = "機器學習的未來發展"
    tokens = tokenizer.tokenize(test_text)
    token_ids = tokenizer.encode(test_text)
    logger.info(f"測試分詞結果: {test_text} -> {tokens}")
    logger.info(f"測試分詞ID: {token_ids}")
    decoded = tokenizer.decode(token_ids)
    logger.info(f"解碼結果: {decoded}")
    
    # 根據模式決定是加載現有模型還是創建新模型
    if continuous_pretraining:
        # 持續預訓練模式：加載現有模型
        if base_model:
            logger.info(f"持續預訓練模式：從 {base_model} 加載模型")
            try:
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=torch.bfloat16,
                    use_cache=False
                )
                logger.info("成功加載現有模型進行持續預訓練")
                
                # 檢查模型詞彙表與分詞器是否匹配
                if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
                    logger.warning(f"模型詞彙表大小 ({model.get_input_embeddings().weight.shape[0]}) "
                                 f"與分詞器詞彙表大小 ({len(tokenizer)}) 不匹配！")
                
                return model, tokenizer
            except Exception as e:
                logger.error(f"加載模型失敗: {str(e)}，將回退到從配置創建新模型")
                # 加載失敗時回退到創建新模型
    
    # 从零开始训练：加载配置并创建新模型
    logger.info("从零开始训练：使用配置文件创建新模型")
    
    # 載入 Gemma 2 配置
    config = transformers.AutoConfig.from_pretrained(
        "./config.json",
        vocab_size=len(tokenizer),
        torch_dtype=torch.bfloat16,
        use_cache=False,  # 訓練時不使用KV緩存
        # 可選調整以適應您的硬體資源
        hidden_size=512,          # 縮小隱藏層大小
        num_hidden_layers=12,      # 減少 Transformer 層數
        num_attention_heads=8,    # 減少注意力頭數量
        intermediate_size=2048,   # 調整 FFN 隱藏層大小
        max_position_embeddings=2048,  # 保持原有上下文長度
        # 使用 Gemma 2 的特定配置選項
        rope_theta=10000.0,
        attention_bias=False,
        tie_word_embeddings=False,
        neftune_noise_alpha=5,  # 使用NEFTune技術改善生成品質
        attn_implementation='flash_attention_2',  # 使用 Flash Attention 2
    )
    
    # 創建模型 - 使用 Gemma 2 架構但從頭開始初始化
    model = transformers.AutoModelForCausalLM.from_config(config)
    logger.info("已创建全新模型")
    
    # 詳細檢查 Embedding 層
    if hasattr(model, "get_input_embeddings"):
        vocab_size = len(tokenizer)
        embedding_layer = model.get_input_embeddings()
        emb_shape = embedding_layer.weight.shape
        actual_vocab_size, emb_dim = emb_shape
        
        logger.info(f"Embedding 層已創建: {type(embedding_layer).__name__}")
        logger.info(f"Embedding 層形狀: {emb_shape}")
        logger.info(f"詞彙表大小: {actual_vocab_size}, Embedding 維度: {emb_dim}")
        
        # 驗證詞彙表大小和維度是否匹配
        if actual_vocab_size != vocab_size:
            logger.error(f"警告: Embedding詞彙表大小 ({actual_vocab_size}) 與分詞器詞彙表 ({vocab_size}) 不匹配!")
        
        if emb_dim != config.hidden_size:
            logger.error(f"警告: Embedding維度 ({emb_dim}) 與hidden_size ({config.hidden_size}) 不匹配!")
    else:
        logger.critical("模型沒有標準的 Embedding 層! 無法進行預訓練。")
        raise RuntimeError("模型缺少必要的 Embedding 層")
    
    # 改進的權重初始化
    def _init_weights(module):
        """使用更穩定的初始化方法"""
        if isinstance(module, torch.nn.Linear):
            # 使用Kaiming初始化，對ReLU激活函數更適合
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            # 使用截斷正態分布初始化Embedding層
            logger.info(f"初始化 Embedding 層: 形狀 {module.weight.shape}")
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    # 确保新创建的模型进行初始化
    model.apply(_init_weights)
    logger.info("模型權重已使用優化方法初始化完成")

    # 驗證 embedding 層功能
    try:
        logger.info("驗證 Embedding 層功能...")
        device = next(model.parameters()).device
        test_ids = torch.tensor([[1, 2, 3]], device=device)
        with torch.no_grad():
            # 獲取 embedding 輸出
            embeddings = model.get_input_embeddings()(test_ids)
        
        expected_shape = (1, 3, config.hidden_size)
        if embeddings.shape == expected_shape:
            logger.info(f"✓ Embedding 層驗證成功! 輸出形狀: {embeddings.shape}")
        else:
            logger.error(f"× Embedding 層驗證失敗! 預期形狀: {expected_shape}, 實際形狀: {embeddings.shape}")
    except Exception as e:
        logger.error(f"驗證 Embedding 層時出錯: {str(e)}")
    
    # 啟用梯度檢查點以節省記憶體
    model.gradient_checkpointing_enable()
    model = model.to(torch.bfloat16)
    
    model_size = sum(t.numel() for t in model.parameters())
    logger.info(f"模型大小: {model_size/1000**2:.1f}M 參數")
    
    return model, tokenizer





def test_model(
    checkpoint_path=None, 
    mode="interactive", 
    max_new_tokens=512,  # 降低默認最大生成標記數
    temperature=0.6,     # 增加溫度以獲得更流暢的生成
    guidance_scale=1.2,  # 新增：生成引導強度
    prompt_template=None,  # 新增：提示詞模板
    test_suite=None,     # 新增：測試集
    output_dir="model_evaluations",  # 新增：評估結果保存目錄
):
    """
    增強型模型測試函數 - 專為小型語言模型優化
    
    參數:
        checkpoint_path (str): 模型檢查點路徑，預設使用最新檢查點
        mode (str): 測試模式，可選"interactive"互動模式、"batch"批量模式、"benchmark"基準測試模式
        max_new_tokens (int): 生成的最大標記數上限，小模型建議設置較小的值
        temperature (float): 生成溫度，控制隨機性
        guidance_scale (float): 文本生成引導強度，大於1增強主題相關性，小於1增加創造性
        prompt_template (str): 提示詞模板，用於格式化輸入，如 "請針對以下問題提供簡明扼要的回答：{input}"
        test_suite (str): 預定義測試集名稱，如"general"、"domain"、"safety"等
        output_dir (str): 評估結果保存目錄
    """
    import os
    import time
    import json
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from datetime import datetime
    
    # 確保評估結果目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 自動尋找最新檢查點
    if checkpoint_path is None:
        output_dir = "YuLLM_small_50M_v1"
        if os.path.exists(output_dir):
            checkpoints = [f for f in os.listdir(output_dir) if f.startswith("checkpoint-")]
            if checkpoints:
                latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
                checkpoint_path = os.path.join(output_dir, latest_checkpoint)
                print(f"使用最新檢查點: {checkpoint_path}")
            else:
                checkpoint_path = output_dir
                print(f"未找到檢查點，使用完整模型: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"找不到模型目錄: {output_dir}")
    
    # 記錄測試時間與設置
    test_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_config = {
        "test_id": test_id,
        "checkpoint": checkpoint_path,
        "mode": mode,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "guidance_scale": guidance_scale,
        "timestamp": datetime.now().isoformat()
    }
    
    # 載入模型與tokenizer (使用更多優化)
    print(f"載入模型: {checkpoint_path}")
    print("載入 tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_path)
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token = "[PAD]"  # 使用专用的填充标记
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # 确保模型知道这个新标记
    
    print("載入模型...")
    start_time = time.time()
    
    # 增加更多載入選項以優化小型模型的記憶體使用和推理速度
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    
    # 嘗試更智能的模型加載方式
    try:
        # 首先嘗試使用 transformers.AutoModelForCausalLM
        model = transformers.AutoModelForCausalLM.from_pretrained(
            checkpoint_path, **model_kwargs
        )
        print("使用標準模型載入方式")
    except Exception as e:
        print(f"標準載入失敗，嘗試使用備用方法: {str(e)}")
        try:
            # 如果標準加載失敗，嘗試使用特定架構
            config = transformers.AutoConfig.from_pretrained(checkpoint_path)
            model_cls = transformers.models.auto.modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING[config.__class__]
            model = model_cls.from_pretrained(checkpoint_path, **model_kwargs)
        except Exception as e2:
            print(f"備用載入也失敗，最後嘗試: {str(e2)}")
            # 最後嘗試 4-bit 加載以節省記憶體
            model = transformers.AutoModelForCausalLM.from_pretrained(
                checkpoint_path, 
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                device_map="auto"
            )
            
    load_time = time.time() - start_time
    print(f"模型載入時間: {load_time:.2f} 秒")
    
    # 創建文字生成管道並應用優化
    pipe = transformers.pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
        device_map="auto"
    )
    
    # 提示詞模板 - 為小模型提供更明確的指引
    DEFAULT_TEMPLATES = {
        "general": "請簡明扼要地回答以下問題：\n{input}\n回答：",
        "factual": "請提供關於以下主題的準確事實信息：\n{input}\n回答：",
        "creative": "請創造性地回應以下提示：\n{input}\n回答：",
        "concise": "請用50字以內簡短回答：\n{input}\n回答：",
        "detailed": "請詳細解釋以下主題，包含重要細節：\n{input}\n回答：",
    }
    
    # 使用提供的模板或默認模板
    active_template = prompt_template if prompt_template else DEFAULT_TEMPLATES["general"]
    
    # 預設測試集
    DEFAULT_TEST_SUITES = {
        "general": [
            "人工智能的應用",
            "太陽系有哪些行星",
            "水的化學特性",
            "中國的四大發明",
            "如何保持健康的生活方式"
        ],
        "domain": [
            "解釋神經網絡的基本原理",
            "量子計算的優勢",
            "細胞分裂過程",
            "經濟學中的供需原理",
            "中國古代文學的特點"
        ],
        "reasoning": [
            "如果一個蘋果重150克，一個橙子重200克，10個蘋果和5個橙子總共多重？",
            "一個正方形的面積是25平方厘米，它的周長是多少？",
            "如果今天是星期二，那麼三天前是星期幾？",
            "某商品原價100元，打八折後又打九折，最終售價是多少？",
            "一列火車長200米，時速120公里，通過一個400米長的隧道需要多少秒？"
        ]
    }
    
    # 選擇要使用的測試集
    if test_suite and test_suite in DEFAULT_TEST_SUITES:
        default_prompts = DEFAULT_TEST_SUITES[test_suite]
    else:
        default_prompts = DEFAULT_TEST_SUITES["general"]
    
    # 調整小型模型的生成參數
    def get_optimized_generation_params(prompt_length):
        """根據提示詞長度動態調整生成參數"""
        # 對於小模型，根據提示詞長度調整生成長度
        tokens_budget = min(2048 - prompt_length, max_new_tokens)
        
        params = {
            'max_new_tokens': tokens_budget,
            'temperature': temperature,
            'top_p': 0.85,  # 稍微降低以增加確定性
            'top_k': 50,    # 增加到50以保持一定多樣性
            'repetition_penalty': 1.2,  # 適中的重複懲罰
            'no_repeat_ngram_size': 3,  # 3-gram防重複
            'do_sample': temperature > 0.1,
            'pad_token_id': tokenizer.eos_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'early_stopping': True,     
            'length_penalty': 0.8,      # 稍微偏好較短回應
        }
        
        return params
    
    # 引導式生成 - 針對小型模型增強
    def guided_generation(prompt, stream=False, **kwargs):
        """使用引导技术增强小模型生成质量，支持流式输出"""
        # 使用提示詞模板格式化輸入
        formatted_prompt = active_template.format(input=prompt)
        
        # 标记化并获取长度
        prompt_tokens = tokenizer.tokenize(formatted_prompt)
        prompt_length = len(prompt_tokens)
        
        # 获取优化的生成参数
        gen_params = get_optimized_generation_params(prompt_length)
        # 覆盖用户提供的参数
        gen_params.update(kwargs)
        
        # 測量生成時間
        start_time = time.time()
        
        # 如果需要流式輸出
        if stream:
            # 流式生成
            print(formatted_prompt, end="", flush=True)
            # 显式创建输入并设置注意力掩码
            inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True)
            input_ids = inputs.input_ids.to(model.device)
            attention_mask = inputs.attention_mask.to(model.device)
            streamer = transformers.TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            # 设置生成参数
            generation_kwargs = {
                **gen_params,
                "input_ids": input_ids,
                "attention_mask": attention_mask,  # 明确提供注意力掩码
                "streamer": streamer,
            }
            
            # 在线程中运行生成过程
            import threading
            thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # 收集生成的文本
            generated_text = ""
            for new_text in streamer:
                print(new_text, end="", flush=True)
                generated_text += new_text
            
            print()  # 最后添加换行
        else:
            # 常规生成 - 使用显式的注意力掩码
            inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True)
            outputs = pipe(
                formatted_prompt, 
                num_return_sequences=1,
                attention_mask=inputs.attention_mask,  # 添加这一行提供注意力掩码
                **gen_params
            )
            raw_output = outputs[0]["generated_text"]
            generated_text = raw_output[len(formatted_prompt):].strip()
        
        generation_time = time.time() - start_time
        
        # 後處理生成的文本
        generated_text = clean_generated_text(generated_text)
        
        # 計算各種指標
        gen_tokens = tokenizer.tokenize(generated_text)
        tokens_per_second = len(gen_tokens) / generation_time if generation_time > 0 else 0
        
        # 質量評估
        quality_scores = assess_text_quality(generated_text)
        
        result = {
            "prompt": prompt,
            "formatted_prompt": formatted_prompt,
            "output": generated_text,
            "raw_output": formatted_prompt + generated_text if stream else raw_output,
            "generation_time": generation_time,
            "tokens_per_second": tokens_per_second,
            "token_count": len(gen_tokens),
            "quality_scores": quality_scores
        }
        
        return result
    
    # 文本清理輔助函數
    def clean_generated_text(text):
        """改進的文本清理函數，減少換行符和無關聯內容"""
        import re
        from difflib import SequenceMatcher
        
        # 處理空輸出
        if not text.strip():
            return "沒有生成有效內容。"
        
        # 合併多個連續換行為單個換行
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 處理不完整句子 - 確保文本在合理的位置結束
        sentence_enders = ['.', '!', '?', '。', '！', '？', '…']
        
        # 如果最後一個字符不是句號等結束符，嘗試找到最後一個完整句子
        if text and text[-1] not in sentence_enders:
            for ender in sentence_enders:
                last_sentence_end = text.rfind(ender)
                if last_sentence_end != -1 and last_sentence_end > len(text) * 0.7:  # 至少保留70%的文本
                    text = text[:last_sentence_end+1]
                    break
        
        # 檢測並移除重複段落
        lines = text.split('\n')
        
        # 檢測整塊重複文本
        filtered_lines = []
        seen_content = set()
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                filtered_lines.append(line)
                continue
                
            # 檢查這一行是否與之前內容高度相似
            is_duplicate = False
            for seen in seen_content:
                # 使用編輯距離檢查相似度
                similarity = SequenceMatcher(None, line_stripped, seen).ratio()
                if similarity > 0.8:  # 80%相似度閾值
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_content.add(line_stripped)
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    # 文本質量評估
    def assess_text_quality(text):
        """評估生成文本的各方面質量"""
        if not text.strip():
            return {
                "length_score": 0,
                "coherence_score": 0,
                "repetition_score": 0,
                "overall_score": 0
            }
            
        # 計算文本長度得分 (0-1)
        length = len(text)
        length_score = min(1.0, length / 500)  # 500字符以上視為滿分
        
        # 計算連貫性得分 - 基於句子數量與文本長度比例
        sentences = [s for s in text.split('.') if s.strip()]
        coherence_score = min(1.0, len(sentences) / max(1, length / 50))
        
        # 計算重複性得分 - 檢測常見重複模式
        words = text.split()
        unique_ratio = len(set(words)) / (len(words) + 0.001)
        repetition_score = unique_ratio
        
        # 計算總體分數 (0-100)
        overall_score = ((length_score + coherence_score + repetition_score) / 3) * 100
        
        return {
            "length_score": round(length_score, 2),
            "coherence_score": round(coherence_score, 2),
            "repetition_score": round(repetition_score, 2),
            "overall_score": round(overall_score, 1)
        }
    
    # 結果格式化顯示
    def display_generation_result(result):
        """美化顯示生成結果"""
        print("\n" + "="*70)
        print(f"提示詞: {result['prompt']}")
        print("-"*70)
        print(f"格式化提示詞: {result['formatted_prompt']}")
        print("\n生成結果:")
        print("-"*70)
        print(result['output'])
        print("-"*70)
        print(f"生成時間: {result['generation_time']:.2f} 秒")
        print(f"生成速度: {result['tokens_per_second']:.2f} tokens/秒")
        print(f"生成標記數: {result['token_count']}")
        
        # 顯示質量評估
        q = result['quality_scores']
        print(f"質量評估: 總分 {q['overall_score']}/100")
        print(f"  - 長度: {q['length_score']:.2f}/1.0")
        print(f"  - 連貫性: {q['coherence_score']:.2f}/1.0")
        print(f"  - 無重複性: {q['repetition_score']:.2f}/1.0")
        
        return True
    
    # 執行互動模式
    def run_interactive_mode():
        """增強的互動測試模式"""
        print("\n===== 進入增強互動模式 =====")
        print("輸入 'exit', 'quit' 或 'q' 退出")
        print("輸入 'params' 查看或修改參數")
        print("輸入 'template' 更改提示詞模板")
        print("輸入 'analyze' 分析先前的生成結果")
        print("輸入 'stream' 切換流式輸出模式")
        
        # 當前參數與歷史記錄
        current_params = {
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'top_p': 0.92,
            'top_k': 40,
            'repetition_penalty': 1.03
        }
        #'top_p': 0.85,'top_k': 50, repetition_penalty': 1.2
        
        # 保存對話歷史
        generation_history = []
        current_template = active_template
        stream_mode = False  # 默認不使用流式輸出
        
        # 互動循環
        while True:
            try:
                prompt = input("\n請輸入提示詞" + (" [流式]" if stream_mode else "") + ": ")
                
                # 處理特殊命令
                if prompt.lower() in ['exit', 'quit', 'q']:
                    if generation_history:
                        save_path = os.path.join(output_dir, f"interactive_session_{test_id}.json")
                        with open(save_path, 'w', encoding='utf-8') as f:
                            json.dump({
                                "config": test_config,
                                "history": generation_history
                            }, f, ensure_ascii=False, indent=2)
                        print(f"會話記錄已保存至: {save_path}")
                    print("退出互動模式")
                    break
                elif prompt.lower() == 'stream':
                    stream_mode = not stream_mode
                    print(f"流式輸出模式: {'開啟' if stream_mode else '關閉'}")
                    continue
                elif prompt.lower() == 'params':
                    # 顯示和修改參數
                    print(f"當前參數: {current_params}")
                    param_to_change = input("要修改哪個參數? (直接回車跳過): ")
                    if param_to_change in current_params:
                        try:
                            new_value = input(f"輸入 {param_to_change} 的新值 (當前: {current_params[param_to_change]}): ")
                            if param_to_change in ['temperature', 'top_p', 'repetition_penalty']:
                                current_params[param_to_change] = float(new_value)
                            else:
                                current_params[param_to_change] = int(new_value)
                            print(f"參數已更新: {param_to_change} = {current_params[param_to_change]}")
                        except ValueError:
                            print("無效輸入，參數未修改")
                    continue
                elif prompt.lower() == 'template':
                    # 更改提示詞模板
                    print(f"當前模板: {current_template}")
                    print("可用模板:")
                    for i, (name, template) in enumerate(DEFAULT_TEMPLATES.items()):
                        print(f"{i+1}. {name}: {template}")
                    
                    choice = input("選擇模板編號 (或輸入 'custom' 自訂模板): ")
                    if choice.lower() == 'custom':
                        custom_template = input("輸入自訂模板 (使用 {input} 表示提示詞位置): ")
                        if '{input}' in custom_template:
                            current_template = custom_template
                            print("已設置自訂模板")
                        else:
                            print("錯誤：模板必須包含 {input} 標記")
                    else:
                        try:
                            idx = int(choice) - 1
                            if 0 <= idx < len(DEFAULT_TEMPLATES):
                                template_name = list(DEFAULT_TEMPLATES.keys())[idx]
                                current_template = DEFAULT_TEMPLATES[template_name]
                                print(f"已選擇模板: {template_name}")
                        except:
                            print("無效選擇，保持當前模板")
                    continue
                elif prompt.lower() == 'analyze':
                    # 分析生成歷史
                    if not generation_history:
                        print("沒有生成歷史可分析")
                        continue
                    
                    times = [h['result']['generation_time'] for h in generation_history]
                    scores = [h['result']['quality_scores']['overall_score'] for h in generation_history]
                    
                    print("\n===== 生成歷史分析 =====")
                    print(f"總計生成次數: {len(generation_history)}")
                    print(f"平均生成時間: {sum(times)/len(times):.2f} 秒")
                    print(f"平均質量分數: {sum(scores)/len(scores):.1f}/100")
                    continue
                
                if not prompt.strip():
                    continue
                
                # 使用當前模板生成回應，支持流式輸出
                result = guided_generation(prompt, stream=stream_mode, **current_params)
                
                # 如果不是流式輸出，顯示完整結果
                if not stream_mode:
                    display_generation_result(result)
                
                # 保存到歷史
                generation_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "prompt": prompt,
                    "template": current_template,
                    "params": current_params.copy(),
                    "stream_mode": stream_mode,
                    "result": result
                })
                
            except KeyboardInterrupt:
                print("\n操作被中斷，退出互動模式")
                break
            except Exception as e:
                print(f"發生錯誤: {e}")
                import traceback
                traceback.print_exc()
    
    # 執行批量測試模式
    def run_batch_mode(prompts=None):
        """增強的批量測試模式"""
        test_prompts = prompts if prompts else default_prompts
        print(f"\n===== 開始批量測試 ({len(test_prompts)} 個提示詞) =====")
        
        results = []
        scores = []
        
        for prompt in tqdm(test_prompts, desc="生成進度"):
            try:
                result = guided_generation(prompt)
                display_generation_result(result)
                results.append(result)
                scores.append(result['quality_scores']['overall_score'])
            except Exception as e:
                print(f"處理 '{prompt}' 時出錯: {e}")
                import traceback
                traceback.print_exc()
        
        # 生成評估報告
        if results:
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            
            print("\n===== 批量測試結果摘要 =====")
            print(f"完成評估提示詞: {len(results)}/{len(test_prompts)}")
            print(f"平均質量分數: {avg_score:.1f}/100")
            print(f"最高分數: {max_score:.1f}, 最低分數: {min_score:.1f}")
            
            # 寫入詳細結果
            output_file = os.path.join(output_dir, f"batch_results_{test_id}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({
                    "config": test_config,
                    "summary": {
                        "prompt_count": len(results),
                        "avg_score": avg_score,
                        "max_score": max_score,
                        "min_score": min_score
                    },
                    "results": results
                }, f, ensure_ascii=False, indent=2)
            
            print(f"詳細結果已保存至: {output_file}")
            
            # 生成簡易報表
            txt_report = os.path.join(output_dir, f"batch_report_{test_id}.txt")
            with open(txt_report, "w", encoding="utf-8") as f:
                f.write(f"模型測試報告 - {test_id}\n")
                f.write(f"模型檢查點: {checkpoint_path}\n")
                f.write(f"測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for i, result in enumerate(results):
                    f.write(f"提示詞 {i+1}: {result['prompt']}\n")
                    f.write(f"質量分數: {result['quality_scores']['overall_score']:.1f}/100\n")
                    f.write(f"生成時間: {result['generation_time']:.2f} 秒\n")
                    f.write("-" * 50 + "\n")
                    f.write(result['output'] + "\n\n")
                    f.write("=" * 50 + "\n\n")
            
            print(f"可讀報告已保存至: {txt_report}")
        
        return results
    
    # 執行基準測試模式
    def run_benchmark_mode():
        """模型效能基準測試"""
        print("\n===== 開始模型基準測試 =====")
        
        # 簡單/中等/複雜提示詞進行測試
        benchmark_sets = {
            "簡短": ["人工智能", "大海的顏色", "什麼是數學"],
            "中等": ["解釋光合作用過程", "中國歷史朝代順序", "計算機如何工作"],
            "複雜": ["量子力學與相對論的關係", "全球氣候變化的影響與對策", "人類意識的本質是什麼"]
        }
        
        # 不同溫度測試
        temperatures = [0.0, 0.5, 0.8]
        
        benchmark_results = {}
        
        # 進行基準測試
        for set_name, prompts in benchmark_sets.items():
            benchmark_results[set_name] = {}
            for temp in temperatures:
                print(f"\n測試 {set_name} 提示詞, 溫度 = {temp}")
                set_results = []
                
                for prompt in prompts:
                    try:
                        # 使用當前溫度生成
                        gen_params = {
                            'temperature': temp,
                            'attention_mask': True  # 确保启用注意力掩码
                        }
                        if temp == 0.0:
                            gen_params['do_sample'] = False
                        result = guided_generation(prompt, **gen_params)
                        set_results.append(result)
                        print(f"提示詞: {prompt}, 分數: {result['quality_scores']['overall_score']}, 時間: {result['generation_time']:.2f}秒")
                    except Exception as e:
                        print(f"處理 '{prompt}' 時出錯: {e}")
                
                # 計算此組合的平均指標
                if set_results:
                    avg_time = sum(r['generation_time'] for r in set_results) / len(set_results)
                    avg_score = sum(r['quality_scores']['overall_score'] for r in set_results) / len(set_results)
                    avg_tokens = sum(r['token_count'] for r in set_results) / len(set_results)
                    
                    benchmark_results[set_name][temp] = {
                        "avg_time": avg_time,
                        "avg_score": avg_score,
                        "avg_tokens": avg_tokens,
                        "details": set_results
                    }
        
        # 保存基準測試結果
        benchmark_file = os.path.join(output_dir, f"benchmark_{test_id}.json")
        with open(benchmark_file, "w", encoding="utf-8") as f:
            json.dump({
                "config": test_config,
                "results": benchmark_results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n基準測試完成，結果已保存至: {benchmark_file}")
        
        # 顯示摘要結果
        print("\n===== 基準測試摘要 =====")
        for set_name, temp_results in benchmark_results.items():
            print(f"\n{set_name} 提示詞:")
            for temp, metrics in temp_results.items():
                print(f"  溫度 {temp}: 分數 {metrics['avg_score']:.1f}/100, 時間 {metrics['avg_time']:.2f}秒, 標記數 {metrics['avg_tokens']:.0f}")
        
        return benchmark_results
    
    # 根據選擇的模式執行相應的測試
    try:
        if (mode == "interactive"):
            run_interactive_mode()
        elif (mode == "batch"):
            run_batch_mode()
        elif (mode == "benchmark"):
            run_benchmark_mode()
        else:
            print(f"未知測試模式: {mode}")
            print("可用模式: interactive, batch, benchmark")
    except Exception as e:
        print(f"測試過程中發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
#    main()
    test_model(checkpoint_path="YuLLM_small_50M_v1/checkpoint-9575", mode="interactive")
