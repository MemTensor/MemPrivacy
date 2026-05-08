import os
import time
import json
import logging
from pathlib import PurePath
from typing import List, Optional
from multiprocessing import Pool

from tqdm import tqdm
from mem0 import Memory
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log,
)

from utils import _load_config, _load_prompt, call_llm, verify_mcq_answer
from privacy_masking import PrivacyStore, mask_dialogue, unmask_dialogue, complete_mask_dialogue


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load config and set up environment
# ---------------------------------------------------------------------------
_config = _load_config("eval_config.yaml")
os.environ["OPENAI_API_KEY"] = _config["openai_api_key"]
os.environ["OPENAI_BASE_URL"] = _config["openai_base_url"]

# ---------------------------------------------------------------------------
# Build mem0 config and initialise Memory instance
# ---------------------------------------------------------------------------

_mem0_cfg = _config["mem0_config"]

def load_mem0_config(user_id: str, db_path: str):

    mem0_config = {
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": f"{user_id}",
                "path": db_path,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": _config["embedding_model"]["model"],
            },
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": _config["memory_llm"]["model"],
                "temperature": _config["memory_llm"]["temperature"],
                "max_tokens": _config["memory_llm"]["max_tokens"],
            },
        },
    }

    return mem0_config

# ---------------------------------------------------------------------------
# Shared retry decorator built from mem0 retry settings
# ---------------------------------------------------------------------------
_mem0_retry = retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_random_exponential(
        min=_mem0_cfg["wait_time_lower"],
        max=_mem0_cfg["wait_time_upper"],
    ),
    stop=stop_after_attempt(_mem0_cfg["retry_times"]),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@_mem0_retry
def add_message(
    memory: Memory,
    messages: List[dict],
    user_id: str,
    timestamp: Optional[str] = None,
) -> None:
    """Add messages to mem0 memory for the given user."""
    memory.add(
        messages=messages,
        user_id=user_id,
        metadata={"timestamp": timestamp} if timestamp else None,
    )


@_mem0_retry
def search_memory(
    memory: Memory,
    query: str,
    user_id: str,
    limit: int = 20,
) -> str:
    """Search mem0 memory and return matching entries as a newline-joined string."""
    search_results = memory.search(
        query=query,
        user_id=user_id,
        limit=limit,
    )

    memories = [
        f"  - {item['memory']}"
        for item in search_results["results"]
    ]

    return "\n".join(memories)


def chunk_dialogues(dialogues: List[dict], turns_per_chunk: int = 1):
    """
    Split dialogues into chunks, each containing `turns_per_chunk` dialogue turns.
    One turn = user + assistant.

    Handles the case where the last turn only has a user message.
    """

    chunks = []
    i = 0
    n = len(dialogues)

    while i < n:

        turns = []
        turn_count = 0

        while i < n and turn_count < turns_per_chunk:

            user_msg = dialogues[i]
            assistant_msg = dialogues[i + 1] if i + 1 < n else None

            turns.append((user_msg, assistant_msg))

            i += 2
            turn_count += 1

        chunks.append(turns)

    return chunks


def process_single_user_data_add_memory(
    user_data: dict, 
    is_mask: bool,
    mask_level: list[str],
    is_mcq: bool,  # are Multiple-choice questions
    turns_per_chunk: int = 1,
    mask_mode: str = "type_specific"  # "type_specific" or "generic" or "complete"
):

    user_id = user_data["uuid"]
    sub_path = f"Mem0_{'mask' if is_mask else 'unmask'}_{''.join(mask_level)}_{mask_mode}_{'mcq' if is_mcq else 'qa'}"
    os.makedirs(os.path.join(_mem0_cfg["db_path"], sub_path), exist_ok=True)
    mem0_db_path = os.path.join(_mem0_cfg["db_path"], sub_path)
    mem0_config = load_mem0_config(user_id, mem0_db_path)
    memory = Memory.from_config(mem0_config)

    if is_mask and mask_mode != "complete":
        privacy_sub_path = os.path.join(_config["privacy"]["db_path"], f"Mem0_{'mask' if is_mask else 'unmask'}_{''.join(mask_level)}_{mask_mode}_{'mcq' if is_mcq else 'qa'}")
        os.makedirs(privacy_sub_path, exist_ok=True)
        privacy_db_path = os.path.join(privacy_sub_path, f"{user_id}.db")
        privacy_store = PrivacyStore(db_path=os.path.abspath(privacy_db_path), mask_mode=mask_mode)

    dialogues = user_data["dialogues"]
    privacy_info_key = "privacy_info_llm"

    dialogue_chunks = chunk_dialogues(dialogues, turns_per_chunk)

    for chunk in tqdm(dialogue_chunks, desc="Processing message chunks"):

        pure_messages = []
        timestamp = None

        for user_msg, assistant_msg in chunk:

            if timestamp is None:
                timestamp = user_msg.get("date", None)

            user_content = user_msg["content"]

            if is_mask:
                if mask_mode == "complete":
                    user_content = complete_mask_dialogue(
                        user_content,
                        user_msg[privacy_info_key] if user_msg.get(privacy_info_key, None) is not None else user_msg["privacy_info"],
                        mask_level
                    )
                else:
                    user_content = mask_dialogue(
                        user_content,
                        user_msg[privacy_info_key] if user_msg.get(privacy_info_key, None) is not None else user_msg["privacy_info"],
                        privacy_store,
                        mask_level
                    )

            pure_messages.append({
                "role": user_msg["role"],
                "content": user_content
            })

            if assistant_msg:

                assistant_content = assistant_msg["content"]

                if is_mask:
                    if mask_mode == "complete":
                        assistant_content = complete_mask_dialogue(
                            assistant_content,
                            assistant_msg[privacy_info_key] if assistant_msg.get(privacy_info_key, None) is not None else assistant_msg["privacy_info"],
                            mask_level
                        )
                    else:
                        assistant_content = mask_dialogue(
                            assistant_content,
                            assistant_msg[privacy_info_key] if assistant_msg.get(privacy_info_key, None) is not None else assistant_msg["privacy_info"],
                            privacy_store,
                            mask_level
                        )

                pure_messages.append({
                    "role": assistant_msg["role"],
                    "content": assistant_content
                })
        
        add_message(memory, pure_messages, user_id, timestamp)
    
    print(f"Memory added to {mem0_db_path} successfully")
    if is_mask and mask_mode != "complete":
        print(f"Privacy store added to {privacy_db_path} successfully")



def process_single_user_data_qa(
    user_data: dict, 
    is_mask: bool,
    mask_level: list[str],
    is_mcq: bool,  # are Multiple-choice questions
    turns_per_chunk: int = 1,
    mask_mode: str = "type_specific"  # "type_specific" or "generic" or "complete"
):

    user_id = user_data["uuid"]
    user_name = user_data["metadata"]["user_name"]
    sub_path = f"Mem0_{'mask' if is_mask else 'unmask'}_{''.join(mask_level)}_{mask_mode}_{'mcq' if is_mcq else 'qa'}"
    os.makedirs(os.path.join(_mem0_cfg["db_path"], sub_path), exist_ok=True)
    mem0_db_path = os.path.join(_mem0_cfg["db_path"], sub_path)
    mem0_config = load_mem0_config(user_id, mem0_db_path)
    memory = Memory.from_config(mem0_config)

    if is_mask and mask_mode != "complete":
        privacy_sub_path = os.path.join(_config["privacy"]["db_path"], f"Mem0_{'mask' if is_mask else 'unmask'}_{''.join(mask_level)}_{mask_mode}_{'mcq' if is_mcq else 'qa'}")
        os.makedirs(privacy_sub_path, exist_ok=True)
        privacy_db_path = os.path.join(privacy_sub_path, f"{user_id}.db")
        privacy_store = PrivacyStore(db_path=os.path.abspath(privacy_db_path), mask_mode=mask_mode)

    results = []
    total_score = 0
    total_valid = 0
    total_num = len(user_data["questions"])
    for question in tqdm(user_data["questions"], desc="Processing questions"):

        query = question["question"]
        answer = question["answer"]

        memories_text = search_memory(
            memory=memory,
            query=query,
            user_id=user_id,
            limit=20
        )

        is_valid = True
        if is_mcq:

            if is_mask and mask_mode != "complete":
                memories_text = unmask_dialogue(memories_text, privacy_store)
            
            prompt_path = _config["prompts"]["answer_prompt_2"]
            prompt = _load_prompt(prompt_path)
            query_prompt = prompt.format(
                question=query,
                user_memories=memories_text,
                options_text="\n".join(question["all_options"])
            )

            response = call_llm(
                query=query_prompt,
                llm_type="answer_llm",
                extract_json=True,
                return_parsed_json=True,
            )

            if isinstance(response, dict):
                is_correct, is_valid = verify_mcq_answer(response.get("answer", ""), answer)
                response_content = json.dumps(response, ensure_ascii=False)
            else:
                is_correct = False
                response_content = str(response)
                is_valid = False

            score = 1 if is_correct else 0
            
        else:
            answer_prompt_path = _config["prompts"]["answer_prompt_1"]
            judge_prompt_path = _config["prompts"]["judge_prompt"]
            answer_prompt = _load_prompt(answer_prompt_path)
            judge_prompt = _load_prompt(judge_prompt_path)

            query_prompt = answer_prompt.format(
                user_name=user_name,
                user_memories=memories_text,
                question=query
            )

            response_content = call_llm(
                query=query_prompt,
                llm_type="answer_llm",
                extract_json=False,
                return_parsed_json=False,
            )

            if is_mask and mask_mode != "complete" and response_content.strip():
                response_content = unmask_dialogue(response_content, privacy_store)

            judge_prompt = judge_prompt.format(
                question=query,
                reference_answer=answer,
                response=response_content
            )

            judgment_response = call_llm(
                query=judge_prompt,
                llm_type="judgment_llm",
                extract_json=True,
                return_parsed_json=True,
            )
            
            if isinstance(judgment_response, dict):
                result_label = judgment_response.get("judgment", "unknown")
                is_correct = (result_label == "correct")
                is_valid = (result_label in {"correct", "partially_correct","incorrect"})
                score = 1 if is_correct else (0.5 if result_label == "partially_correct" else 0)
            else:
                is_correct = False
                score = 0
                is_valid = False

        results.append({
            "user_id": user_id,
            "user_name": user_name,
            "question": query,
            "answer": answer,
            "response": response_content,
            "score": score,
            "is_valid": is_valid
        })
        total_score += score
        total_valid += is_valid

    return results, total_score, total_valid, total_num


def _worker_add_memory(args):
    user_data, is_mask, mask_level, is_mcq, turns_per_chunk, mask_mode = args
    return process_single_user_data_add_memory(user_data, is_mask, mask_level, is_mcq, turns_per_chunk, mask_mode)


def process_all_users_data_add_memory(
    user_data_path: str,
    is_mask: bool,
    mask_level: list[str],
    is_mcq: bool,
    turns_per_chunk: int = 1,
    mask_mode: str = "type_specific",
    user_num: int = None,
    num_workers: int = 10,
):

    with open(user_data_path, "r", encoding="utf-8") as f:
        user_data_list = [json.loads(line) for line in f]

    if user_num is not None:
        user_data_list = user_data_list[:user_num]

    tasks = [(u, is_mask, mask_level, is_mcq, turns_per_chunk, mask_mode) for u in user_data_list]

    with Pool(num_workers) as pool:

        for _ in tqdm(
            pool.imap_unordered(_worker_add_memory, tasks),
            total=len(tasks),
            desc="Adding memory"
        ):
            pass
    
    print("Finished adding memory!")


def _worker_qa(args):
    user_data, is_mask, mask_level, is_mcq, turns_per_chunk, mask_mode = args
    return process_single_user_data_qa(user_data, is_mask, mask_level, is_mcq, turns_per_chunk, mask_mode)


def process_all_users_data_qa(
    user_data_path: str,
    is_mask: bool,
    mask_level: list[str],
    is_mcq: bool,
    turns_per_chunk: int = 1,
    mask_mode: str = "type_specific",
    user_num: int = None,
    num_workers: int = 10,
):

    with open(user_data_path, "r", encoding="utf-8") as f:
        user_data_list = [json.loads(line) for line in f]

    if user_num is not None:
        user_data_list = user_data_list[:user_num]

    tasks = [(u, is_mask, mask_level, is_mcq, turns_per_chunk, mask_mode) for u in user_data_list]

    all_results = []
    all_total_score = 0
    all_total_valid = 0
    all_total_num = 0

    with Pool(num_workers) as pool:

        for user_results, total_score, total_valid, total_num in tqdm(
            pool.imap_unordered(_worker_qa, tasks),
            total=len(tasks),
            desc="Processing questions"
        ):
            all_results.extend(user_results)
            all_total_score += total_score
            all_total_valid += total_valid
            all_total_num += total_num

    all_accuracy = all_total_score / all_total_num if all_total_num > 0 else 0
    all_accuracy_valid = all_total_score / all_total_valid if all_total_valid > 0 else 0

    output_dict = {
        "total_score": all_total_score,
        "total_valid": all_total_valid,
        "total_num": all_total_num,
        "accuracy": all_accuracy,
        "accuracy_valid": all_accuracy_valid,
        "records": all_results,
    }

    user_data_name = PurePath(user_data_path).stem
    output_path = os.path.join(
        _config["output_path"],
        f"Mem0_{user_data_name}_{'mask' if is_mask else 'unmask'}_{''.join(mask_level)}_{mask_mode}_{time.strftime('%Y%m%d%H%M%S')}.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=4)

    print("accuracy:", all_accuracy)
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    import time
    import multiprocessing as mp

    mp.set_start_method("spawn")

    user_data_path = ""
    is_mask = True
    mask_level = ["PL2", "PL3", "PL4"]
    is_mcq = True
    turns_per_chunk = 5
    user_num = 10
    num_workers = 10
    mask_mode = "type_specific"  # "generic", "type_specific", "complete"

    start_time = time.time()

    process_all_users_data_add_memory(
        user_data_path=user_data_path,
        is_mask=is_mask,
        mask_level=mask_level,
        is_mcq=is_mcq,
        turns_per_chunk=turns_per_chunk,
        mask_mode=mask_mode,
        user_num=user_num,
        num_workers=num_workers,
    )

    process_all_users_data_qa(
        user_data_path=user_data_path,
        is_mask=is_mask,
        mask_level=mask_level,
        is_mcq=is_mcq,
        turns_per_chunk=turns_per_chunk,
        mask_mode=mask_mode,
        user_num=user_num,
        num_workers=num_workers,
    )

    print(f"is_mask: {is_mask}")
    print(f"mask_level: {mask_level}")
    print(f"is_mcq: {is_mcq}")
    print(f"turns_per_chunk: {turns_per_chunk}")
    print(f"mask_mode: {mask_mode}")
    print(f"user_num: {user_num}")
    print(f"num_workers: {num_workers}")

    end_time = time.time()
    elapsed = end_time - start_time

    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"Time taken: {int(hours)}h {int(minutes)}m {seconds:.2f}s")