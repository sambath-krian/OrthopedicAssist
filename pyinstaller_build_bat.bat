# pyinstaller --windowed --onefile GUI_OrthopedicAssist_7.py

#pyinstaller --windowed --icon=Icon.ico GUI_OrthopedicAssist_7.py
pyinstaller.exe --windowed --icon=OrthopedicAssit.ico GUI_OrthopedicAssist_7.py

#copy "config.yaml" "dist/GUI_OrthopedicAssist_7/."
#copy "dr_config.yaml" "dist/GUI_OrthopedicAssist_7/."
mkdir "dist/GUI_OrthopedicAssist_7/mediapipe/modules"
xcopy modules "dist/GUI_OrthopedicAssist_7/mediapipe/modules/." /s /e /Y
mkdir "dist/GUI_OrthopedicAssist_7/face_detect_dnn_sample"
xcopy modules "dist/GUI_OrthopedicAssist_7/face_detect_dnn_sample/." /s /e /Y

powershell Compress-Archive "dist/GUI_OrthopedicAssist_7" "dist/GUI_OrthopedicAssist_7.zip"
