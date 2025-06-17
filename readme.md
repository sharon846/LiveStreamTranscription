## Installation

1. Install VB-Cable
2. Set "CABLE Input" as default output device

As for now, the files are:

- basic_gui.py: uses the vb-cable, gives a gui of output with timestamp, including clean transcription.

- selenium_control: utilizes selenium to open and close the tabs.

- server.py: flask server that gets raw data from chrome extension and transcribes the data. moreover - writes it to a raw file.

Command to convert the raw data:

ffmpeg -i inp_file out.wav