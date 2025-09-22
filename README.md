# research-meeting-ai

This project serves as a tool to conduct live transcription of research meetings, and an AI based assistant which summarizes text, finds related papers, and allows you to take notes.

The code in this github is an overview of the back end software we used to get this up and running. 

test_transcription.py is a document to test the transcription model that we are using (OpenAi's faster_whisper) on your terminal without running/using server.

The command to run the app is:
    streamlit run src/streamlit_app.py --server.headless true

    You can access it at http://localhost:8501

The command to manually kill Streamlit processes is:
  pkill -f streamlit

