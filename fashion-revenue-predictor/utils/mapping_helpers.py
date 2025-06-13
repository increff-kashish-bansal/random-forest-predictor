from difflib import get_close_matches
from typing import List, Dict

def suggest_mappings(uploaded_cols: List[str], required_cols: List[str]) -> Dict[str, str]:
    """
    Suggest mappings between uploaded columns and required columns using fuzzy matching.
    
    Args:
        uploaded_cols: List of column names from uploaded file
        required_cols: List of required column names in schema
        
    Returns:
        Dictionary mapping required column names to suggested uploaded column names
    """
    mappings = {}
    for req_col in required_cols:
        # Get best match with minimum similarity threshold of 0.6
        matches = get_close_matches(req_col.lower(), 
                                  [col.lower() for col in uploaded_cols], 
                                  n=1, 
                                  cutoff=0.6)
        if matches:
            # Find original case version of the matched column
            original_case = next(col for col in uploaded_cols if col.lower() == matches[0])
            mappings[req_col] = original_case
        else:
            mappings[req_col] = None
            
    return mappings
