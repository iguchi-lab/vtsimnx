from .builder import build_config, build_config_with_warnings, build_config_with_warning_details

# 後方互換（かつての builder.parse を維持）
parse = build_config

__all__ = ["build_config", "build_config_with_warnings", "build_config_with_warning_details", "parse"]
