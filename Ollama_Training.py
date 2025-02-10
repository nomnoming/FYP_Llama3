from unsloth import FastLanguageModel 
import torch
import os
import json
from datasets import Dataset
import torch
import torch
torch.cuda.empty_cache()

# Set the environment variable to avoid CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Define the directory where the MITRE ATT&CK JSON files were copied
mitre_data_dir = '/workspace/mitre_data/'

# List all JSON files in the directory
json_files = []
for root, dirs, files in os.walk(mitre_data_dir):
    for file in files:
        if file.endswith('.json'):
            json_files.append(os.path.join(root, file))

# Now load and process each JSON file dynamically
dataset_list = []
for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)
        # Extract the relevant fields you want to train on (e.g., attack pattern, description, etc.)
        attack_info = {
            'threat': data.get('name', 'Unknown threat'),
            'vulnerability': data.get('description', 'No description'),
            'risk': data.get('platforms', 'No platforms'),
            'consequences': data.get('impact', 'Unknown impact')
        }
        dataset_list.append(attack_info)

# Convert the list of dictionaries into a dictionary of lists
if dataset_list:  # Ensure there is data before converting
    dataset_dict = {key: [entry[key] for entry in dataset_list] for key in dataset_list[0].keys()}
    dataset = Dataset.from_dict(dataset_dict)

    # Print dataset summary
    print(f"Dataset prepared: {len(dataset)} samples")
else:
    print("No data found in MITRE ATT&CK JSON files.")


OLLAMA_SERVER_URL = "http://ollama:11434"  # This assumes the container is on the same Docker network

#-----Declaration and Initalization of Variables--------------------------------------------------------
max_seq_length = 2048 
dtype = None 
load_in_4bit = True 

model, tokenizer = FastLanguageModel.from_pretrained(     
    model_name = "./Meta-Llama-3.1-8B",     
    max_seq_length = max_seq_length,     
    dtype = dtype,
    load_in_4bit = load_in_4bit, 
)

# Enables gradient checkpointing
model.config.use_gradient_checkpointing = True

# Initializes LoRA model with reduced rank and dropout
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj",],
    lora_alpha = 16,
    lora_dropout = 0.1, #lora_dropout will drop layers to prevent overfitting.
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

#--- Data Prep -------
from datasets import load_dataset
dataset = load_dataset(
    "csv",
    data_files = "./Dataset/FYP_KPMG renamed dataset.csv",
    split = "train",
)

print(dataset.column_names)
print(dataset[0])

#-----Merged-Prompt-------------------------------------------------------------------------------------
from unsloth import to_sharegpt, apply_chat_template

# Rename columns to remove spaces
dataset = dataset.rename_columns({
    "Threat Event": "Threat_Event"
})

# Add an empty "Output" column (not needed if training data already contains output)
dataset = dataset.map(lambda example: {"Output": "No response yet"})

# Merges the prompt together
dataset = to_sharegpt(
    dataset,
    merged_prompt = \
        "[[The threat is: {Threat_Event}]]"\
        "[[\nThis threat is caused by: {Vulnerability}]]"\
        "[[\nThis puts the {Asset} at risk]]"\
        "[[\nLeading to consequences like: {Consequence}]]",
    conversation_extension = 4,
    output_column_name = "Output",
)

#----- Print Statements -----
# Print how the dataset looks like NOW
from pprint import pprint
print ("\n\n Below is how the dataset looks like BEFORE standardization")
pprint(dataset[0])

# Converts all user, assistant and system tags to HF style.
from unsloth import standardize_sharegpt
dataset = standardize_sharegpt(dataset)
print("\n\n This is how the dataset looks like AFTER standardization")
pprint(dataset[0])

#-----Chat Template Formatting--------------------------------------------------------------------------
chat_template = """
summary contains contents of the cyber attack from the query.
attacker is known as adversary instead.
construct a detailed step-by-step scenario following the MITRE ATT&CK framework. 
Each point should be numbered and explicitly state the asset at stake, starting with Initial Access.
>>> Details of cyber attack:
{INPUT}
>>> Here is the structured attack scenario:
{OUTPUT}
""" 

from unsloth import apply_chat_template
dataset = apply_chat_template(
    dataset,
    tokenizer = tokenizer,
    chat_template = chat_template,
)
#--Now we will actually train the model-----------------------------------------------------------------
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# This is using 60 steps to speed the training up, but we can set the 'num_train_epochs = 1' for a full run, and turn off 'max_steps=None'
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # This can make training 5x faster for short sequences
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,
        warmup_steps =  5,
        
        #max_steps= 5,  # Ensure this is None or removed to avoid conflicts
        num_train_epochs=2,  # Train for 1 full epoch
        
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)
# Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

#----Display statistics showing us how well it learns.-----
trainer_stats = trainer.train()


#-----Show final memory and time stats-------
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# #-----Saving the trained model------------------------------------------------------------------------
model.save_pretrained("lora_model") # Local saving
tokenizer.save_pretrained("lora_model")

#-------Inference---------------------------------------------------------------------------------------
print("\n\n This is how the chat template looks like before inference")
print(chat_template)
print("\n\n")

if True: #The 'False' is to disable this block of code. Change to True to run it.
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model", # make sure this is the SAVED model; different from base model.
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit, 
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
pass

#----- We want to specify the variables in order to test the code.---------------------
messages = [
    {"role": "user", "content": "The threat is: Phishing email promising free software trial\n"\
                                 "This threat is caused by: No software verification process \n"\
                                 "This puts the Corporate laptop at risk\n"\
                                 "Leading to consequences like: Installs malicious software, leading to network compromise."},
]

# Explicitly pass chat_template while applying it
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

#Specify attention mask - helps the model distinguish between actual tokens and padding tokens
attention_mask = (input_ids != tokenizer.pad_token_id).long()

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(input_ids, attention_mask=attention_mask, streamer=text_streamer, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)

# #--- Convert the Llama to GGUF ------------------------------------------

# Change it to TRUE to get the codes to work.
# Save to 8bit Q8_0
if False: model.save_pretrained_gguf("model", tokenizer)
# Save to 16bit GGUF
if True: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
    
# ~END~
