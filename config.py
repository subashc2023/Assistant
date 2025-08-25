
import os
import pathlib
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

DEFAULTS = {
    'openai': 'openai/gpt-4.1-mini',
    'anthropic': 'anthropic/claude-3-7-sonnet-20250219',
    'gemini': 'gemini/gemini-2.5-pro',
    'groq': 'groq/llama-3.3-70b-versatile',
}

PROVIDERS = {
    'groq': {
        'prefixes': ['groq/'],
        'keywords': ['llama', 'mixtral', 'gemma'],
        'env_vars': ['GROQ_API_KEY'],
        'max_tokens_cap': None
    },
    'anthropic': {
        'prefixes': ['anthropic/', 'claude-'],
        'keywords': ['claude', 'sonnet', 'haiku', 'opus'],
        'env_vars': ['ANTHROPIC_API_KEY'],
        'max_tokens_cap': 8192
    },
    'openai': {
        'prefixes': ['openai/', 'gpt-', 'o1-', 'o3-', 'oai-'],
        'keywords': ['gpt', 'openai'],
        'env_vars': ['OPENAI_API_KEY'],
        'max_tokens_cap': 8192
    },
    'gemini': {
        'prefixes': ['gemini/', 'google/'],
        'keywords': ['gemini', 'google'],
        'env_vars': ['GEMINI_API_KEY', 'GOOGLE_API_KEY'],
        'max_tokens_cap': 8192
    }
}


@dataclass
class AppConfig:

    model: str = DEFAULTS['groq']
    max_tokens: int = 32768
    max_tool_hops: int = 50

    tool_result_max_chars: int = 8000
    tool_timeout_seconds: float = 30.0
    max_parallel_tools: int = 4
    tool_preview_lines: int = 0

    system_prompt: str = "You are a helpful assistant."
    log_level: str = "INFO"
    log_json: bool = False

    model_aliases: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def load(cls, 
             yaml_path: Optional[pathlib.Path] = None,
             cli_args: Optional[Dict[str, Any]] = None,
             legacy_prompt_path: Optional[pathlib.Path] = None) -> 'AppConfig':
        config = cls()

        if yaml_path and yaml_path.exists():
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    yaml_data = yaml.safe_load(f) or {}

                config.model_aliases = yaml_data.get('model_aliases', {})

                for key in [k for k in dir(config) if not k.startswith('_') and k != 'model_aliases']:
                    if key in yaml_data and yaml_data[key] is not None:
                        setattr(config, key, yaml_data[key])

                if 'system_prompt' in yaml_data:
                    config.system_prompt = yaml_data['system_prompt']

            except Exception as e:
                print(f"[Warn] Could not read {yaml_path}: {e}")

        if cli_args:
            if cli_args.get('provider'):
                config.model = DEFAULTS.get(cli_args['provider'], config.model)

            if cli_args.get('model'):
                model = cli_args['model']
                config.model = config.model_aliases.get(model, model)

            if cli_args.get('system_prompt_file'):
                prompt_path = pathlib.Path(cli_args['system_prompt_file'])
                if prompt_path.exists():
                    try:
                        config.system_prompt = prompt_path.read_text(encoding='utf-8')
                    except Exception as e:
                        print(f"[Warn] Could not read system prompt file: {e}")

            if 'system_prompt' in cli_args and cli_args['system_prompt'] is not None:
                config.system_prompt = cli_args['system_prompt']

            for key in ['max_tokens', 'max_tool_hops', 'tool_result_max_chars',
                       'tool_timeout_seconds', 'max_parallel_tools', 'tool_preview_lines',
                       'log_level', 'log_json']:
                if key in cli_args and cli_args[key] is not None:
                    setattr(config, key, cli_args[key])

        if (not config.system_prompt or config.system_prompt == "You are a helpful assistant."):
            if legacy_prompt_path and legacy_prompt_path.exists():
                try:
                    config.system_prompt = legacy_prompt_path.read_text(encoding='utf-8')
                    print("[Info] Using legacy prompt.txt; consider moving to tinyclient_config.yaml")
                except Exception:
                    pass

        config.max_tokens = config._apply_token_cap(config.model, config.max_tokens)
        
        return config
    
    def _apply_token_cap(self, model: str, requested_tokens: int) -> int:
        provider = detect_provider(model)
        if provider and provider in PROVIDERS:
            cap = PROVIDERS[provider].get('max_tokens_cap')
            if cap and requested_tokens > cap:
                print(f"[Info] Capping max_tokens to {cap} for {provider}")
                return cap
        return requested_tokens
    
    def reload(self, yaml_path: Optional[pathlib.Path] = None,
               cli_args: Optional[Dict[str, Any]] = None) -> None:
        new_config = self.load(yaml_path, cli_args)

        for key in dir(new_config):
            if not key.startswith('_'):
                setattr(self, key, getattr(new_config, key))


@dataclass
class Metrics:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost: float = 0.0
    
    def update(self, prompt: int = 0, completion: int = 0, cost: float = 0.0) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        if cost:
            self.total_cost += cost
    
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens
    
    def summary(self) -> str:
        return f"{self.prompt_tokens}/{self.completion_tokens}/{self.total_tokens} (${self.total_cost:.4f})"


def detect_provider(model_name: str) -> Optional[str]:
    if not model_name:
        return None

    lower = model_name.lower()

    for provider, config in PROVIDERS.items():
        if any(lower.startswith(p) for p in config['prefixes']):
            return provider
        if any(k in lower for k in config['keywords']):
            return provider

    return None

def get_provider_env_vars(model_name: str) -> Tuple[Optional[str], List[str]]:
    provider = detect_provider(model_name)
    if provider and provider in PROVIDERS:
        return provider.title(), PROVIDERS[provider]['env_vars']
    return None, []


def check_provider_auth(model_name: str) -> Tuple[bool, List[str]]:
    _, env_vars = get_provider_env_vars(model_name)
    missing = [var for var in env_vars if not os.getenv(var)]
    return len(missing) == 0, missing