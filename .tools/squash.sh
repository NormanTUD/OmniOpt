#!/bin/bash

tags=($(git tag --sort=creatordate))

tags+=("HEAD")

for (( i=0; i<${#tags[@]}-1; i++ )); do
	start_tag=${tags[i]}
	end_tag=${tags[i+1]}

	echo "Squashing commits from $start_tag to $end_tag..."

	git reset --soft $start_tag

	git commit -m "Squashed commit for range $start_tag to $end_tag"
done

#echo "Squashing complete. Pushing changes..."
#git push origin main --force
