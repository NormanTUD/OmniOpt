#!/bin/bash

input=".omniopt.py"

echo "_omniopt() {"
echo "  local state"
echo "  _arguments \\"

grep "add_argument(" "$input" | while IFS= read -r line; do
    # Extrahiere Argumentnamen (z. B. "config_yaml")
    argname=$(echo "$line" | sed -n "s/.*add_argument(\s*['\"]\([^'\"]\+\)['\"].*/\1/p")
    if [[ -z "$argname" ]]; then
        continue
    fi

    # Nur -- lange Optionen verarbeiten, kurze (-x) überspringen
    if [[ "$argname" == -* && "$argname" != --* ]]; then
        continue
    fi

    # Doppelte Bindestriche korrekt
    arg="--${argname#--}"

    # Extrahiere Hilfe-Text
    help=$(echo "$line" | sed -n "s/.*help\s*=\s*['\"]\([^'\"]\+\)['\"].*/\1/p")

    # Typ-Hint (für ZSH Completion)
    if echo "$line" | grep -q "type\s*=\s*str"; then
        if [[ "$arg" == *"config_"* || "$arg" == *"run_dir" || "$arg" == *"run_program" || "$arg" == *"continue_previous_job" ]]; then
            typehint="_files"
        else
            typehint=""
        fi
    else
        typehint=""
    fi

    # Platzhalter-Name für Completion
    placeholder=$(echo "$argname" | tr 'a-z-' 'A-Z_')

    # Ausgeben mit echten Zeilenumbrüchen, korrekt escaped
    echo "    '$arg=[$help]:____${placeholder}:${typehint}' \\"
done

# Zusätzliche statische Argumente anhängen
cat <<EOF
}
EOF
