import re
import math
from typing import List, Dict, Any
from openai import OpenAI

client = OpenAI(api_key='',base_url='')
EMBEDDING_MODEL = "text-embedding-3-small"

############################################################
# Tokenizer
############################################################

def mixed_tokenize(text: str):
    """
    Chinese/Japanese split by character.
    Consecutive Latin letters/digits/symbols kept as a whole.
    """
    cj_range = r'\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u30fb'
    pattern = re.compile(fr'[{cj_range}]|[^{cj_range}\s]+')
    return [match.group() for match in pattern.finditer(text)]


############################################################
# Longest Common Contiguous Substring
############################################################

def longest_common_contiguous(tokens1, tokens2):
    dp = [[0]*(len(tokens2)+1) for _ in range(len(tokens1)+1)]
    max_len = 0
    for i in range(1, len(tokens1)+1):
        for j in range(1, len(tokens2)+1):
            if tokens1[i-1] == tokens2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                max_len = max(max_len, dp[i][j])
    return max_len


############################################################
# Embedding Batch Generation
############################################################


def normalize(v):
    norm = math.sqrt(sum(x*x for x in v))
    if norm == 0:
        return v
    return [x / norm for x in v]


def build_embedding_cache(pred_list, ref_list, model=EMBEDDING_MODEL):
    """
    Collect all unique privacy types and generate embeddings once.
    """
    all_types = set()

    for item in pred_list:
        all_types.add(item["privacy_type"])
    for item in ref_list:
        all_types.add(item["privacy_type"])

    all_types = list(all_types)

    if not all_types:
        return {}

    response = client.embeddings.create(
        model=model,
        input=all_types
    )

    embedding_cache = {}
    for text, data in zip(all_types, response.data):
        embedding_cache[text] = normalize(data.embedding)

    return embedding_cache


def cosine_similarity(v1, v2):
    dot = sum(a*b for a,b in zip(v1, v2))
    norm1 = math.sqrt(sum(a*a for a in v1))
    norm2 = math.sqrt(sum(b*b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


############################################################
# Rules
############################################################

NONE_STRICT_TYPES = {
    "Communication Content", "Accommodation Record", "Accommodation Records", 
    "Company/Organization Info", "Credit Report", "Credit Reports", "Debt/Loan", 
    "Debt/Loan Info", "Document Info", "Document Scans", "Financial", 
    "Itinerary/Trajectory", "Job Intent/Status", "Judicial Record", 
    "Medical Health", "Medical Record", "Minor Information", "Minor Info", 
    "Political Views/Stance", "Relationship Info", "Sensitive Identity", 
    "Transaction Record", "Vulnerability Details"
}

LEVEL_MAP = {"PL2":2, "PL3":3, "PL4":4}


############################################################
# Sub-score calculations
############################################################

def score_original_text(pred_text, ref_text, full_dialogue, ref_type):
    if pred_text not in full_dialogue:
        return 0.0

    if pred_text == ref_text:
        return 1.0

    if ref_type not in NONE_STRICT_TYPES:
        return 0.0

    t1 = mixed_tokenize(pred_text)
    t2 = mixed_tokenize(ref_text)

    if not t1 or not t2:
        return 0.0

    lcc = longest_common_contiguous(t1, t2)
    p = lcc / len(t1)
    r = lcc / len(t2)

    if p + r == 0:
        return 0.0

    return 2 * p * r / (p + r)


def score_privacy_level(pred_level, ref_level):
    if pred_level == ref_level:
        return 1.0

    if pred_level not in LEVEL_MAP or ref_level not in LEVEL_MAP:
        return 0.0

    return 0.0


def score_privacy_type(pred_type, ref_type, embedding_cache):
    if pred_type == ref_type:
        return 1.0

    if pred_type not in embedding_cache or ref_type not in embedding_cache:
        return 0.0

    return cosine_similarity(
        embedding_cache[pred_type],
        embedding_cache[ref_type]
    )


############################################################
# Matching
############################################################

def match_items(pred_list, ref_list, full_dialogue, embedding_cache, mode="mean"):

    if not pred_list:
        return [], {i: 0 for i in range(len(ref_list))}, []
    
    if not ref_list:
        return [0]*len(pred_list), {}, []

    all_scores = []

    for i, p in enumerate(pred_list):
        for j, r in enumerate(ref_list):

            s_text = score_original_text(
                p["original_text"],
                r["original_text"],
                full_dialogue,
                r["privacy_type"]
            )

            s_level = score_privacy_level(
                p["privacy_level"],
                r["privacy_level"]
            )

            s_type = score_privacy_type(
                p["privacy_type"],
                r["privacy_type"],
                embedding_cache
            )

            if mode == "product":
                total = s_text * s_level * s_type
            else:
                total = (s_text + s_level + s_type) / 3

            all_scores.append((i, j, total, s_text, s_level, s_type))

    all_scores.sort(key=lambda x: x[3], reverse=True)

    used_pred = set()
    used_ref = set()

    pred_scores = [0]*len(pred_list)
    ref_scores = {i: 0 for i in range(len(ref_list))}
    subscore_records = []

    for i, j, total, s_text, s_level, s_type in all_scores:
        if i in used_pred:
            continue

        if j not in used_ref:
            pred_scores[i] = total
            ref_scores[j] = total
            used_pred.add(i)
            used_ref.add(j)

            subscore_records.append((s_text, s_level, s_type))

    if len(subscore_records) < len(pred_list):
        for _ in range(len(pred_list) - len(subscore_records)):
            subscore_records.append((0, 0, 0))

    return pred_scores, ref_scores, subscore_records


############################################################
# Metrics
############################################################

def compute_metrics(pred_scores, ref_scores):

    precision = sum(pred_scores)/len(pred_scores) if pred_scores else 0
    recall = sum(ref_scores.values())/len(ref_scores) if ref_scores else 0

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


############################################################
# Main
############################################################

def evaluate_privacy(
    messages: List[Dict[str, Any]],
    pred_list: List[Dict[str, Any]],
    ref_list: List[Dict[str, Any]],
    mode="mean"
):
    if not pred_list and not ref_list:
        return {
            "overall": {
                "precision": 1,
                "recall": 1,
                "f1": 1
            },
            "per_level_metrics": {
                "PL2": {
                    "precision": 1,
                    "recall": 1,
                    "f1": 1
                },
                "PL3": {
                    "precision": 1,
                    "recall": 1,
                    "f1": 1
                },
                "PL4": {
                    "precision": 1,
                    "recall": 1,
                    "f1": 1
                }
            },
            "mean_subscores": {
                "original_text_mean": 1,
                "privacy_level_mean": 1,
                "privacy_type_mean": 1
            }
        }

    full_dialogue = "\n".join([m["content"] for m in messages])

    embedding_cache = build_embedding_cache(pred_list, ref_list)

    pred_scores, ref_scores, subscore_records = match_items(
        pred_list,
        ref_list,
        full_dialogue,
        embedding_cache,
        mode
    )

    precision, recall, f1 = compute_metrics(pred_scores, ref_scores)

    # PL accuracy
    per_level_metrics = {}

    for level in ["PL2", "PL3", "PL4"]:

        pred_L = [p for p in pred_list if p["privacy_level"] == level]
        ref_L = [r for r in ref_list if r["privacy_level"] == level]

        if not pred_L and not ref_L:
            per_level_metrics[level] = {
                "precision": 1,
                "recall": 1,
                "f1": 1
            }
            continue
        
        if not pred_L or not ref_L:
            per_level_metrics[level] = {
                "precision": 0,
                "recall": 0,
                "f1": 0
            }
            continue

        pred_scores_L, ref_scores_L, _ = match_items(
            pred_L,
            ref_L,
            full_dialogue,
            embedding_cache,
            mode
        )

        precision_L, recall_L, f1_L = compute_metrics(
            pred_scores_L,
            ref_scores_L
        )

        per_level_metrics[level] = {
            "precision": precision_L,
            "recall": recall_L,
            "f1": f1_L
        }

    # mean sub scores
    if subscore_records:
        mean_text = sum(x[0] for x in subscore_records) / len(pred_list)
        mean_level = sum(x[1] for x in subscore_records) / len(pred_list)
        mean_type = sum(x[2] for x in subscore_records) / len(pred_list)
    else:
        mean_text = mean_level = mean_type = 0

    return {
        "overall": {
            "precision": precision,
            "recall": recall,
            "f1": f1
        },
        "per_level_metrics": per_level_metrics,
        "mean_subscores": {
            "original_text_mean": mean_text,
            "privacy_level_mean": mean_level,
            "privacy_type_mean": mean_type
        }
    }