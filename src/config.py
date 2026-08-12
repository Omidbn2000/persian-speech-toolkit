# src/myproject/config.py
import json
from pathlib import Path
from typing import Any, Optional

class Config:
    def __init__(self, config_path: Optional[str] = None):
        self.root_dir = self._find_root()
        
        if config_path is None:
            config_path = self.root_dir / "config" / "config.json"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _find_root(self) -> Path:
        """Find project root by looking for config folder or .git"""
        current = Path(__file__).resolve()
        
        # Try to find project root
        for parent in current.parents:
            # Look for config folder
            if (parent / "config").exists():
                return parent
            # Look for .git
            if (parent / ".git").exists():
                return parent
            # Look for setup.py
            if (parent / "setup.py").exists():
                return parent
        
        # If we're in src/myproject/, go up two levels to project root
        if current.parent.name == "myproject" and current.parent.parent.name == "src":
            return current.parent.parent.parent
        
        return Path.cwd()
    
    def _load_config(self):
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config not found at {self.config_path}, using defaults")
            return self._get_defaults()
        except json.JSONDecodeError as e:
            print(f"Error parsing config.json: {e}")
            return self._get_defaults()
    
    def _get_defaults(self):
        return {
            "model_paths": {
                "tts_onnx": "models/Mana-Persian-Piper/fa_IR-mana-medium.onnx"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def get_path(self, key: str) -> Optional[Path]:
        path_str = self.get(key)
        if path_str:
            return self.root_dir / path_str
        return None

# Create a singleton instance
config = Config()