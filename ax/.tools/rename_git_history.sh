#!/bin/bash

function get_response {
        d="$1"
        d=$(echo "$d" | jq -Rsa .)
        data="{ \"model\": \"mistral\", \"prompt\": $d }"
        echo $data
        curl -q http://localhost:11434/api/generate -d "$data" | jq -r '.response' | tr -d '\n'
}

commits=$(git log --grep="fix" --pretty=format:"%h")

for commit in $commits; do
        git_diff="This is my commit diff. Create a short but concise git message for this: $(git diff $commit^ $commit | cat)"
	echo "Commit:"
	echo "$commit"

        new_commit_msg=$(get_response "$git_diff")
	echo "$new_commit_msg"
        git checkout $commit
        GIT_COMMITTER_DATE="$(git show --format=%aD $commit | head -1)" git commit --amend -m "$new_commit_msg"
        git checkout -
        git push --force-with-lease
done
