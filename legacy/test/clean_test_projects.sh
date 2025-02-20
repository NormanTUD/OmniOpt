#!/bin/bash

cd ..

testproject_path=test/projects/

for testproject in $(ls "$testproject_path"); do
    perl projects/cleanprojects.pl "$testproject_path/$testproject"
done
