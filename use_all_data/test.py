
import train_thinking

eval_dataset = train_thinking.ReasoningDataset("eval", "sft")
for i in range(10):
        print(eval_dataset[i]["labels"])
