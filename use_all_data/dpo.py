from transformers import AutoTokenizer
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
from trl.data_utils import maybe_apply_chat_template, maybe_extract_prompt
from use_all_data import train_seq_pred_aligned_phase1, train_DPO
from accelerate import PartialState


train_dataset, eval_dataset = train_DPO.get_data()

model, tokenizer = train_seq_pred_aligned_phase1.load_model_tokenizer()

dpo_config = DPOConfig(
    output_dir='./something',
    learning_rate=1e-5,
    per_device_train_batch_size=1,
)

class InspectDPOTrainer(DPOTrainer):
    def _prepare_dataset(
        self,
        dataset,
        processing_class,
        args: DPOConfig,
        dataset_name: str,
    ):  
        print(f"\n=== Preparing dataset: {dataset_name} ===\n")
        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc nor writer_batch_size
            map_kwargs["num_proc"] = args.dataset_num_proc
            map_kwargs["writer_batch_size"] = 10

        with PartialState().main_process_first():
            # Extract prompt if needed
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Extracting prompt in {dataset_name} dataset"
            dataset = dataset.map(maybe_extract_prompt, **map_kwargs)
            print("\nAfter maybe_extract_prompt:")
            for i in range(len(dataset)):
                print(dataset[i])

            # Apply the chat template if needed
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
            dataset = dataset.map(
                maybe_apply_chat_template, fn_kwargs={"tokenizer": processing_class, "tools": args.tools}, **map_kwargs
            )
            print("\nAfter maybe_apply_chat_template:")
            for i in range(len(dataset)):
                print(dataset[i])

            # Tokenize the dataset
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

            dataset = dataset.map(
                self.tokenize_row if not self.is_vision_model else self.process_row,
                # remove_columns=["chosen", "rejected"],
                remove_columns=[],
                fn_kwargs={
                    "processing_class": processing_class,
                    "max_prompt_length": args.max_prompt_length,
                    "max_completion_length": args.max_completion_length,
                    # for enc-dec, we add the special tokens ([bos_token] + prompt + [eos_token]; completion + [eos_token])
                    "add_special_tokens": False,
                },
                **map_kwargs,
            )
            print("\nAfter tokenization:")
            for i in range(len(dataset)):
                print(dataset[i])

        return dataset


trainer = InspectDPOTrainer(
    args=dpo_config, 
    model=model, 
    ref_model=None, 
    train_dataset=train_dataset, 
    processing_class=tokenizer)
# processed_dataset = trainer._prepare_dataset(train_dataset, tokenizer, dpo_config, "train")

print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens([128335, 220, 128615, 220, 128924, 220, 129024, 128009])))
