#!/bin/bash

if ! command -v plantuml >/dev/null 2>/dev/null; then
	echo "plantuml is not installed. Try to install it, e.g. via"
	echo "sudo apt-get install plantuml"
	echo "Cannot continue."
	exit 1
fi

plantuml -DMODE_DARK=false -tsvg architecture.puml -o output_light
plantuml -DMODE_DARK=true -tsvg architecture.puml -o output_dark

exit 0
