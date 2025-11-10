from .builder import build_config

# 後方互換: 旧API parse をエイリアスとして公開
parse = build_config

__all__ = ["build_config", "parse"]
