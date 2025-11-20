from typing import List, Dict

def encode_to_toon(data: List[Dict]) -> str:
    """Encodes a list of dicts into a simple TOON-like table format.

    Each dict is converted into a row separated by '|' and fields by ';'.
    Example:
        [{'id': 1, 'title': 'Scene 1'}, {'id': 2, 'title': 'Scene 2'}]
    becomes:
        id;title|1;Scene 1|2;Scene 2
    """
    if not data:
        return ""
    # header
    headers = list(data[0].keys())
    lines = [";".join(headers)]
    for item in data:
        row = ";".join(str(item.get(h, "")) for h in headers)
        lines.append(row)
    return "|".join(lines)
