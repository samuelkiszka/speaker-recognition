#!/bin/bash
#
# Script to download and set up Vosk speech recognition models
#
cd ../models/ || exit

wget https://alphacephei.com/vosk/models/vosk-model-small-cs-0.4-rhasspy.zip
unzip vosk-model-small-cs-0.4-rhasspy.zip
rm vosk-model-small-cs-0.4-rhasspy.zip