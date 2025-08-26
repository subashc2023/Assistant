import os
import pathlib
import re
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import litellm

PROVIDERS = {
    'groq': {
        'default_model': 'groq/llama-3.3-70b-versatile',
        'pattern': re.compile(r'^groq/|llama|mixtral|gemma', re.I),
        'env_vars': ['GROQ_API_KEY'],
        'max_tokens_cap': None,
        'special_models': {
            'openai/gpt-oss-120b': 'groq/openai/gpt-oss-120b',
        }
    },
    'anthropic': {
        'default_model': 'anthropic/claude-3-7-sonnet-20250219',
        'pattern': re.compile(r'^anthropic/|claude', re.I),
        'env_vars': ['ANTHROPIC_API_KEY'],
        'max_tokens_cap': 8192
    },
    'openai': {
        'default_model': 'openai/gpt-4.1-mini',
        'pattern': re.compile(r'^openai/|^gpt-|^o[13]-|^oai-', re.I),
        'env_vars': ['OPENAI_API_KEY'],
        'max_tokens_cap': 8192
    },
    'gemini': {
        'default_model': 'gemini/gemini-2.5-pro',
        'pattern': re.compile(r'^gemini/|^google/', re.I),
        'env_vars': ['GEMINI_API_KEY'],
        'max_tokens_cap': 8192
    }
}

DEFAULTS = {name: p['default_model'] for name, p in PROVIDERS.items()}

MODEL_ALIASES: Dict[str, str] = {
    'llamaS': 'groq/llama-3.1-8b-instant',
    'llamaL': 'groq/llama-3.3-70b-versatile',
    'oss': 'groq/openai/gpt-oss-120b',

    'haiku': 'anthropic/claude-3-5-haiku-latest',
    'sonnet': 'anthropic/claude-3-7-sonnet-20250219',
    'opus': 'anthropic/claude-3-opus-20240229',

    '4o': 'openai/gpt-4o',
    '4omini': 'openai/gpt-4o-mini',
    '4.1': 'openai/gpt-4.1',
    '4.1mini': 'openai/gpt-4.1-mini',

    'flash': 'gemini/gemini-2.5-flash',
    'pro': 'gemini/gemini-2.5-pro',
    'lite': 'gemini/gemini-2.5-lite',
}

def detect_provider(model_name: str) -> Optional[str]:
    if not model_name:
        return None

    for provider, config in PROVIDERS.items():
        if config['pattern'].search(model_name):
            return provider

    return None

def resolve_groq_special_model(model: str) -> str:
    groq_config = PROVIDERS.get('groq', {})
    special_models = groq_config.get('special_models', {})

    for special, full in special_models.items():
        if model == special or model == full:
            return full

    for special, full in special_models.items():
        if model == special.lstrip('groq/'):
            return full

    return model

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
    use_color: bool = True

    retry_delays: List[float] = field(default_factory=lambda: [1.0, 3.0, 5.0])

    model_aliases: Dict[str, str] = field(default_factory=lambda: dict(MODEL_ALIASES))
    
    @classmethod
    def load(cls, cli_args: Optional[Dict[str, Any]] = None) -> 'AppConfig':
        config = cls()

        if cli_args:
            if cli_args.get('provider'):
                config.model = DEFAULTS.get(cli_args['provider'], config.model)

            if cli_args.get('model'):
                config.model = cli_args['model']

            if cli_args.get('system_prompt_file'):
                prompt_path = pathlib.Path(cli_args['system_prompt_file'])
                if prompt_path.exists():
                    try:
                        config.system_prompt = prompt_path.read_text(encoding='utf-8')
                    except (OSError, IOError) as e:
                        print(f"[Warn] Could not read system prompt file: {e}")

            if 'system_prompt' in cli_args and cli_args['system_prompt'] is not None:
                config.system_prompt = cli_args['system_prompt']

            for key in ['max_tokens', 'max_tool_hops', 'tool_result_max_chars',
                       'tool_timeout_seconds', 'max_parallel_tools', 'tool_preview_lines',
                       'log_level', 'log_json', 'use_color']:
                if key in cli_args and cli_args[key] is not None:
                    setattr(config, key, cli_args[key])

            if (('use_color' not in cli_args) or (cli_args.get('use_color') is None)) and os.getenv('NO_COLOR'):
                config.use_color = False

        try:
            litellm.model_alias_map = dict(config.model_aliases)
        except Exception:
            pass

        effective_model = config.resolve_effective_model(config.model)
        config.max_tokens = config._apply_token_cap(effective_model, config.max_tokens)

        return config
    
    def _apply_token_cap(self, model: str, requested_tokens: int) -> int:
        model = resolve_groq_special_model(model)

        provider = detect_provider(model)
        if provider and provider in PROVIDERS:
            cap = PROVIDERS[provider].get('max_tokens_cap')
            if cap and requested_tokens > cap:
                # Silently cap without emitting an info line
                return cap
        return requested_tokens

    def resolve_effective_model(self, model: Optional[str] = None) -> str:
        name = model or self.model

        resolved = resolve_groq_special_model(name)
        if resolved != name:
            return resolved

        try:
            alias_map = getattr(litellm, 'model_alias_map', {}) or {}
            if name in alias_map:
                resolved = alias_map[name]
                return resolve_groq_special_model(resolved)
        except Exception:
            pass

        if name in self.model_aliases:
            resolved = self.model_aliases[name]
            return resolve_groq_special_model(resolved)

        return name

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

def get_provider_env_vars(model_name: str) -> Tuple[Optional[str], List[str]]:
    model_name = resolve_groq_special_model(model_name)

    provider = detect_provider(model_name)
    if provider and provider in PROVIDERS:
        return provider.title(), PROVIDERS[provider]['env_vars']
    return None, []

def check_provider_auth(model_name: str) -> Tuple[bool, List[str]]:
    _, env_vars = get_provider_env_vars(model_name)
    missing = [var for var in env_vars if not os.getenv(var)]
    return len(missing) == 0, missing