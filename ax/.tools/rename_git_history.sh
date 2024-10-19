#!/bin/bash

function get_response {
        d="$1"
        d=$(echo "$d" | jq -Rsa .)
        data="{ \"model\": \"mistral\", \"prompt\": $d }"
        #echo $data
        curl -s http://localhost:11434/api/generate -d "$data" | jq -r '.response' | tr -d '\n'
}

# Alle commits mit "fix" im Commit-Message
commits=$(git log --grep="fix" --pretty=format:"%h")

for commit in $commits; do
        # Generiere den git diff und die entsprechende Anfrage für die AI
	echo "Commit: $commit"
        git_diff="This is my commit diff. Create a short but concise git message for this: $(git diff $commit^ $commit | cat)"

	echo "$git_diff"

        # Hole die neue Commit-Nachricht
        new_commit_msg=$(get_response "$git_diff")

	echo "New commit message:"
	echo "$new_commit_msg"

        # Rebase den Commit und ändere die Commit-Nachricht
        git rebase --interactive --autosquash $commit^ <<EOF
reword $commit
EOF

        # Führe den git commit --amend aus, um die neue Nachricht zu setzen
        GIT_COMMITTER_DATE="$(git show --format=%aD $commit | head -1)" git commit --amend --no-edit -m "$new_commit_msg"

        # Pushe den Commit mit force
        #git push --force-with-lease
done
