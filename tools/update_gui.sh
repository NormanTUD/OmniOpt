#!/bin/bash

rm -rf gui
scp -r service@imageseg.scads.de:/var/www/html/omnioptgui/ .
mv omnioptgui gui
rm gui/*~
