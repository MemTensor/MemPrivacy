import json
import time
import copy

from tqdm import tqdm
from openai import OpenAI
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from transformers import AutoModelForCausalLM, AutoTokenizer

from metric import evaluate_privacy
from utils import _load_prompt


privacy_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "original_text": {"type": "string"},
            "privacy_type": {"type": "string"},
            "privacy_level": {
                "type": "string",
                "enum": ["PL1", "PL2", "PL3", "PL4"]
            }
        },
        "required": ["original_text", "privacy_type", "privacy_level"],
        "additionalProperties": False
    }
}

run_mode = 'vllm'   # vllm  gpt_local
if run_mode == 'vllm':
    model_name_or_path = "checkpoint-xxx"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    sampling_params = SamplingParams(
        temperature=0.1, 
        top_p=0.1, 
        repetition_penalty=1.05,
        max_tokens=6144,
        structured_outputs=StructuredOutputsParams(json=privacy_schema)
    ) 
    model = LLM(
        model=model_name_or_path,
        tensor_parallel_size=1, 
        pipeline_parallel_size=1, 
        dtype='float16',
        gpu_memory_utilization=0.9
    ) 

input_file = 'test_mem_privacy_annotated_final.jsonl'

def writer(system_prompt,query):
    if run_mode == 'vllm':
        messages = [
            {"role": "user", "content": system_prompt+query}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Set to False to strictly disable thinking
        )
        outputs = model.generate([text], sampling_params)
        for output in outputs:
            generated_text = output.outputs[0].text
        response = generated_text
        return response.strip()
    elif run_mode == 'gpt_local':
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8000/v1"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        chat_response = client.chat.completions.create(
            model='Qwen3-4B-privacy',
            messages=[
                {"role": "user", "content": system_prompt+query}, 
            ],
            temperature=0.1,
            top_p=0.1,
            presence_penalty=1.05
        )
        return chat_response.choices[0].message.content.strip()


false_pred_num = 0
all_data1 = []
all_data2 = []
eval_final = {}

system_prompt = _load_prompt("prompts/extract_privacy.txt")

start = time.time()
with open(input_file, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        print("line_num: ",line_num,flush=True)

        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Skip line {line_num}: JSON decode error -> {e}")
            continue

        dialogues = data.get("dialogues", [])
        save={
            "uuid":data.get("uuid"),
            "metadata":data.get("metadata"),
            "dialogues":[],
            "questions":data.get("questions")
        }
        tmp_dialogues=[]
        for dialogue in tqdm(dialogues):
            tmp_dialogue=copy.deepcopy(dialogue) 
            if dialogue.get("role") == "user":
                current_input = {
                    "role": "user",
                    "content": dialogue.get("content", "")
                }
                try:
                    pred_list_str=writer(system_prompt.format(real_name=data.get("metadata").get("user_name")),json.dumps(current_input, ensure_ascii=False, indent=2))
                    pred_list = json.loads(pred_list_str)
                    eval_product=evaluate_privacy([current_input],pred_list,dialogue.get("privacy_info", []),'product')
                    eval_mean=evaluate_privacy([current_input],pred_list,dialogue.get("privacy_info", []),'mean')
                    all_data1.append(eval_product)
                    all_data2.append(eval_mean)
                    tmp_dialogue["privacy_info_llm"]=pred_list
                except Exception:
                    false_pred_num+=1
                    tmp_dialogue["privacy_info_llm"]=[]
                
                eval_final={
                  "false_pred_num":false_pred_num,
                  "product":all_data1,
                  "mean":all_data2
                }  
                with open("xxx.json", "w", encoding="utf-8") as f:
                    json.dump(eval_final, f, ensure_ascii=False, indent=2)
                
                tmp_dialogues.append(tmp_dialogue)
            else:
                current_input = {
                    "role": "user",
                    "content": dialogue.get("content", "")
                }
                try:
                    pred_list_str=writer(system_prompt.format(real_name=data.get("metadata").get("user_name")),json.dumps(current_input, ensure_ascii=False, indent=2))
                    pred_list = json.loads(pred_list_str)
                    tmp_dialogue["privacy_info_llm"] = pred_list
                except Exception:
                    false_pred_num+=1
                    tmp_dialogue["privacy_info_llm"] = []
                tmp_dialogues.append(tmp_dialogue)

        save["dialogues"]=tmp_dialogues

        with open("xxx.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(save, ensure_ascii=False) + "\n")
        
                
                
end = time.time()
print(f"Total time taken: {end - start:.4f} seconds")

# CUDA_VISIBLE_DEVICES=1 nohup python eval.py >> xxx.log 2>&1 &