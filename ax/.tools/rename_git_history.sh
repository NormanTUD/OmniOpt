#!/bin/bash

function get_response {
	d="$1"
	d=$(echo "$d" | jq -Rsa .)
	data="{ \"model\": \"mistral\", \"prompt\": $d }"
	curl -q http://localhost:11434/api/generate -d "$data" | jq -r '.response' | tr -d '\n'
}

commits=$(git log --grep="fix" --pretty=format:"%h")

for commit in $commits; do
	#echo "Commit: $commit"
	git_diff="This is my commit diff. Create a short but concise git message for this: $(git diff $commit^ $commit | cat)"

	echo $(get_response "$git_diff")
done
