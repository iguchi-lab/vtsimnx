from typing import Any, Dict, List, Tuple, Optional

from .parse import parse

def resolve(raw_config: Dict[str, Any], output_path: Optional[str] = "parsed_input_data.json") -> Dict[str, Any]:
  parse(raw_config)
  
