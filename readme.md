## Installation

1. Install VB-Cable
2. Set "CABLE Input" as default output device

As for now, the files are:

- basic_gui.py: uses the vb-cable, gives a gui of output with timestamp, including clean transcription.

- server.py: flask server that gets raw data from chrome extension and transcribes the data. moreover - writes it to a raw file.

- client.py: a client that reads from the channels file, and auto-starts recording 

Command to convert the raw data:

ffmpeg -i inp_file out.wav

chrome://extensions/shortcuts
Alt+Shift+1