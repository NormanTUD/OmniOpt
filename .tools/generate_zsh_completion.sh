#!/bin/bash

input="omniopt"

echo "_omniopt() {"
echo "  local state"
echo "  _arguments \\"

grep "add_argument(" "$input" | while IFS= read -r line; do
    argname=$(echo "$line" | sed -n "s/.*add_argument(\s*['\"]\([^'\"]\+\)['\"].*/\1/p")
    if [[ -z "$argname" ]]; then
        continue
    fi

    if [[ "$argname" == -* && "$argname" != --* ]]; then
        continue
    fi

    arg="--${argname#--}"

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

    placeholder=$(echo "$argname" | tr 'a-z-' 'A-Z_')

    echo "    '$arg=[$help]:____${placeholder}:${typehint}' \\"
done

cat <<EOF
}
EOF
