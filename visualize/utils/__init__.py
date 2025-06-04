import json
from typing import List, Dict, Any

def read_jsonl_file(file_path: str) -> List[Dict[Any, Any]]:
    """Reads a JSONL file and returns a list of dictionaries."""
    data: List[Dict[Any, Any]] = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line: # Ensure non-empty lines before parsing
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping line due to JSON decode error: {e} in line: '{line}'")
                    # Optionally, re-raise or handle more gracefully depending on requirements
    return data

__all__ = ['read_jsonl_file']
