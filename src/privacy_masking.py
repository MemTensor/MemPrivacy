import re
import json
import sqlite3
import logging
import traceback
from typing import Dict, List, Optional, Tuple

import yaml
import json_repair
from openai import OpenAI


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_yaml_config(config_path: str = "privacy_config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# SQLite Privacy Store
# ---------------------------------------------------------------------------

class PrivacyStore:
    """
    Manages privacy items in a local SQLite database.

    Each privacy item is stored with:
      - original_text   (unique key for dedup)
      - privacy_type
      - privacy_level   (PL2 / PL3 / PL4)
      - mask            (e.g. <Real_Name_1>)

    Mask naming convention:
      <{privacy_type with spaces replaced by _}_{sequence_number}>
    """

    def __init__(self, db_path: str = "privacy_store.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS privacy_items (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                original_text TEXT    NOT NULL UNIQUE,
                privacy_type  TEXT    NOT NULL,
                privacy_level TEXT    NOT NULL,
                mask          TEXT    NOT NULL UNIQUE
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_privacy_type  ON privacy_items(privacy_type)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_privacy_level ON privacy_items(privacy_level)"
        )
        self._conn.commit()

    # ---- mask helpers ----

    def _type_to_mask_prefix(self, privacy_type: str) -> str:
        return privacy_type.replace(" ", "_").replace("/", "_")

    def _next_mask(self, privacy_type: str) -> str:
        prefix = self._type_to_mask_prefix(privacy_type)
        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM privacy_items WHERE privacy_type = ?",
            (privacy_type,),
        ).fetchone()
        seq = (row["cnt"] if row else 0) + 1
        return f"<{prefix}_{seq}>"

    # ---- CRUD ----

    def get_or_create(
        self, original_text: str, privacy_type: str, privacy_level: str
    ) -> str:
        """
        Return the mask for *original_text*.  If the text already exists in the
        database, return its existing mask (regardless of type/level).  Otherwise
        create a new entry and return a freshly generated mask.
        """
        row = self._conn.execute(
            "SELECT mask FROM privacy_items WHERE original_text = ?",
            (original_text,),
        ).fetchone()
        if row:
            return row["mask"]

        mask = self._next_mask(privacy_type)
        self._conn.execute(
            "INSERT INTO privacy_items (original_text, privacy_type, privacy_level, mask) "
            "VALUES (?, ?, ?, ?)",
            (original_text, privacy_type, privacy_level, mask),
        )
        self._conn.commit()
        return mask

    def query_by_mask(self, mask: str) -> Optional[Dict]:
        row = self._conn.execute(
            "SELECT * FROM privacy_items WHERE mask = ?", (mask,)
        ).fetchone()
        return dict(row) if row else None

    def query_by_original_text(self, original_text: str) -> Optional[Dict]:
        row = self._conn.execute(
            "SELECT * FROM privacy_items WHERE original_text = ?",
            (original_text,),
        ).fetchone()
        return dict(row) if row else None

    def query_by_privacy_type(self, privacy_type: str) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM privacy_items WHERE privacy_type = ?",
            (privacy_type,),
        ).fetchall()
        return [dict(r) for r in rows]

    def query_by_privacy_level(self, privacy_level: str) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM privacy_items WHERE privacy_level = ?",
            (privacy_level,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all(self) -> List[Dict]:
        rows = self._conn.execute("SELECT * FROM privacy_items").fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self._conn.close()


# ---------------------------------------------------------------------------
# 1. mask_dialogue  —  replace privacy items with masks
# ---------------------------------------------------------------------------

def mask_dialogue(
    dialogue_text: str,
    privacy_items: List[Dict],
    store: PrivacyStore,
    mask_levels: Optional[List[str]] = None,
) -> str:
    """
    Given dialogue text and a list of identified privacy items, replace each
    qualifying item's *original_text* in the dialogue with a mask token, then
    return the masked dialogue.

    Only items whose *privacy_level* is in *mask_levels* will be replaced and
    stored.  Items at other levels are silently ignored (neither replaced nor
    stored).

    Args:
        dialogue_text:  The raw dialogue string.
        privacy_items:  List of dicts, each with keys
                        ``original_text``, ``privacy_type``, ``privacy_level``.
        store:          A ``PrivacyStore`` instance.
        mask_levels:    Which privacy levels to mask, e.g. ``["PL3", "PL4"]``.
                        Defaults to ``["PL3", "PL4"]`` if not provided.

    Returns:
        The masked dialogue string.
    """
    if mask_levels is None:
        mask_levels = ["PL3", "PL4"]

    replacements: List[Tuple[str, str]] = []

    for item in privacy_items:
        level = item.get("privacy_level", "")
        if level not in mask_levels:
            continue

        original = item["original_text"]
        ptype = item["privacy_type"]

        mask = store.get_or_create(original, ptype, level)
        replacements.append((original, mask))

    # Sort by length descending so longer matches are replaced first,
    # avoiding partial replacement of overlapping substrings.
    replacements.sort(key=lambda x: len(x[0]), reverse=True)

    masked_text = dialogue_text
    for original, mask in replacements:
        masked_text = masked_text.replace(original, mask)

    return masked_text


# ---------------------------------------------------------------------------
# 2. unmask_dialogue  —  restore masks back to original text
# ---------------------------------------------------------------------------

_MASK_PATTERN = re.compile(r"<[A-Za-z_]+_\d+>")


def unmask_dialogue(masked_text: str, store: PrivacyStore) -> str:
    """
    Find all ``<Type_N>`` mask tokens in *masked_text* and replace them with
    the corresponding original privacy text from the store.

    Args:
        masked_text:  Text that may contain mask tokens.
        store:        A ``PrivacyStore`` instance.

    Returns:
        The unmasked text.
    """

    def _replace(match: re.Match) -> str:
        mask = match.group(0)
        record = store.query_by_mask(mask)
        if record:
            return record["original_text"]
        return mask

    return _MASK_PATTERN.sub(_replace, masked_text)


# ---------------------------------------------------------------------------
# 3. LLM-based privacy detection  (OpenAI-compatible API, YAML config)
# ---------------------------------------------------------------------------

def _build_llm_client(config: dict):
    """Build an OpenAI client from the *llm* section of the YAML config."""
    llm_cfg = config["llm"]
    return OpenAI(
        base_url=llm_cfg["base_url"],
        api_key=llm_cfg["api_key"],
    )


def _load_prompt_template(config: dict) -> str:
    prompt_path = config["privacy"]["prompt_path"]
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# 4. Full pipeline:  detect  →  mask  →  store  →  return
# ---------------------------------------------------------------------------

def detect_and_mask_dialogue(
    message_text: str,
    config: dict,
    store: PrivacyStore,
    mask_levels: Optional[List[str]] = None,
    real_name: str = "",
) -> Tuple[str, List[Dict]]:
    """
    End-to-end convenience function.

    1. Call the privacy-detection LLM on *message_text*.
    2. For each identified item whose level is in *mask_levels*, generate a mask,
       store it locally, and replace it in the dialogue text.
    3. Return the masked dialogue and the raw list of detected privacy items.

    Args:
        message_text:       The text of the message.
        config:             Full YAML config dict.
        store:              ``PrivacyStore`` instance.
        mask_levels:        Which levels to mask (default from config or
                            ``["PL3", "PL4"]``).
        real_name:          The real name of the user.(Optional, can help the LLM to better detect the privacy items)

    Returns:
        ``(masked_dialogue_text, detected_privacy_items)``
    """
    if mask_levels is None:
        mask_levels = config.get("privacy", {}).get("mask_levels", ["PL3", "PL4"])

    client = _build_llm_client(config)
    prompt_template = _load_prompt_template(config)
    user_prompt = prompt_template.format(
        real_name=real_name,
        input_dialogue_content=message_text
    )

    request_params: dict = {
        "model": config["llm"]["model"],
        "messages": [
            {"role": "user", "content": user_prompt},
        ],
    }
    if config["llm"].get("max_tokens"):
        request_params["max_tokens"] = int(config["llm"]["max_tokens"])
    if config["llm"].get("temperature") is not None:
        request_params["temperature"] = float(config["llm"]["temperature"])
    if config["llm"].get("timeout"):
        request_params["timeout"] = int(config["llm"]["timeout"])

    response = client.chat.completions.create(**request_params)
    content = response.choices[0].message.content.strip()
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE).strip()
    parsed = json_repair.loads(content)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON list from LLM, got {type(parsed)}")
    privacy_items = parsed

    masked_text = mask_dialogue(message_text, privacy_items, store, mask_levels)

    return masked_text, privacy_items


if __name__ == "__main__":
    config = load_yaml_config("privacy_config.yaml")
    store = PrivacyStore(config["privacy"]["db_path"])

    real_name = "Jean-Baptiste"
    message_text = "That’s great to hear—knowing there are ergonomic seats and full accessibility features really puts my mind at ease. You can use the 6-digit one-time SMS code 829417 to complete the payment.Also, for receiving the e-ticket and booking confirmation, I logged into your ticketing system using my personal email: jean-baptiste517@global-inbox.co. Please make sure all confirmation documents are sent to this address."
    mask_levels = ["PL2", "PL3", "PL4"]

    masked_text, privacy_items = detect_and_mask_dialogue(message_text, config, store, mask_levels=mask_levels, real_name=real_name)
    print(masked_text)
    print(privacy_items)