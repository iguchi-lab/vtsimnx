from .builder import build_config

# 後方互換（かつての builder.parse を維持）
parse = build_config

__all__ = ["build_config", "parse"]
