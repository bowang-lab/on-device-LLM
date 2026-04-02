"""
Shared color palette for all Eurorad figures.

Import MODEL_COLORS from this module to keep colors consistent across
bar plots, radar plots, and any future visualizations.
"""

# ─── Per-model colors ───────────────────────────────────────────────────────
# Proprietary models: each gets a distinct hue
# On-device models: pastel-to-saturated gradient within each family

MODEL_COLORS = {
    # Proprietary
    "DeepSeek-R1":    "#A8D5A2",  # sage green
    "GPT-5-mini":     "#D1C4E9",  # light violet
    "GPT-5.1":        "#B39DDB",  # violet
    "Gemini 3.1":     "#FFD87E",  # pastel golden yellow
    # gpt-oss family (salmon/pink gradient: lighter = smaller / base, darker = larger / FT)
    "OSS-20B (H)":    "#F5C6C2",  # pastel pink
    "OSS-120B (H)":   "#F28B82",  # salmon (swapped from GPT-5.1)
    "gpt-oss-120B":   "#F5C6C2",  # pastel pink  (alias for finetune base)
    "120B (FT)":      "#F28B82",  # salmon        (alias for finetune FT)
    # Qwen family (blue gradient: lighter = smaller / base, darker = larger / FT)
    "Qwen 9B":        "#B8D4F0",  # pastel blue
    "Qwen 27B":       "#8BB8E0",  # medium blue
    "Qwen 35B":       "#B8D4F0",  # pastel blue   (base in finetune plot)
    "Qwen 35B (FT)":  "#5E9CD0",  # deeper blue
    "9B (FT)":        "#5E9CD0",  # deeper blue   (alias for finetune)
    "35B (FT)":       "#5E9CD0",  # deeper blue   (alias for finetune)
}

# Aliases used across scripts with different naming conventions
MODEL_COLORS["Gemini 3.1 Pro"] = MODEL_COLORS["Gemini 3.1"]
MODEL_COLORS["Qwen3.5 9B"]     = MODEL_COLORS["Qwen 9B"]
MODEL_COLORS["Qwen3.5 27B"]    = MODEL_COLORS["Qwen 27B"]
MODEL_COLORS["Qwen3.5 35B"]    = MODEL_COLORS["Qwen 35B"]
