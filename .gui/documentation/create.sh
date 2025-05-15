#!/bin/bash
plantuml -DMODE_DARK=true -tsvg architecture.puml -o output_dark
plantuml -DMODE_DARK=false -tsvg architecture.puml -o output_light
