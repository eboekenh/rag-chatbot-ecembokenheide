#!/usr/bin/env bash
# Ollama Guard — PreToolUse hook (Linux/macOS)
# Blocks shell commands involving Ollama when Ollama is not running on port 11434.

INPUT=$(cat)
COMMAND=$(printf '%s' "$INPUT" | python3 -c \
  "import sys, json; d=json.load(sys.stdin); print(d.get('toolInput',{}).get('command',''))" 2>/dev/null)

if echo "$COMMAND" | grep -qiE "ollama|qa_generator_ollama|test_ollama"; then
    if ! curl -sf http://localhost:11434/api/version > /dev/null 2>&1; then
        printf '{"stopReason":"Ollama is not running on port 11434. Start it first: ollama serve"}\n'
        exit 2
    fi
fi
exit 0
