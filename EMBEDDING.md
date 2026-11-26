# Buffalo L embedding prototype

Run these commands from `THE CODE` in PowerShell using the existing Python 3.12 environment:

```powershell
& '.venv/Scripts/python.exe' -m pip install onnxruntime==1.31.1
& '.venv/Scripts/python.exe' app.py
& '.venv/Scripts/python.exe' main.py
```

`app.py` opens the camera in **Showcase** mode. Press **Capture** to see the
detection, aligned image, face crop and an embedding status with three sample
values. Press **Retake** to clear the result. Showcase displays the webcam
embedding without saving the vector.

Use **Switch to Scan** to see the live landmark and pose measurements.
Choose a save folder, then press **Enroll new person** and follow the prompts.
Each new person gets a unique ID. The scan saves five reference views in a new
`scan_<timestamp>` folder, with original PNGs in `images/`, vectors in
`embeddings/`, and a `references.csv` linking them to the person's ID. When
**Scan complete** appears, you can give the person a display name.

Use **Select scan** to choose a completed scan. **Name reference scan** edits
its display name. **Rescan selected person** keeps that person's ID and name
and writes a new scan beside the old one. Existing scans with `person_001`
are still compatible.

Use **Select gallery** to test against the people whose scan folders share a
parent directory. The gallery uses the newest completed scan for each ID.
**Select scan** tests against just that scan. Select a separate test photo
folder and leave **Score cutoff** blank to inspect the scores on a first run.
With a cutoff, each face is assigned to the person with the best accepted
score, or marked `unknown`. A tie between people also returns `unknown`.
Select a result row to see the face box and its name in the photo. Use
**Print results** to send the table to the terminal.

`main.py` runs the batch test without opening windows. It reads the existing
`kepek/landmarking test2/nyers` images and writes a CSV, crops, recognizer inputs
and local NumPy vectors to `kepek/embedding test1`. The older test outputs are
not rewritten. The batch fixtures use unique JPG names.

Keep both checkpoints under `weights/`: the existing
`yolo11s-pose_widerface.pt` from [YOLO11Face](https://github.com/zjykzj/YOLO11Face)
and `w600k_r50.onnx` extracted from `buffalo_l.zip` at the
[official InsightFace Model Zoo release](https://github.com/deepinsight/insightface/releases/tag/model-zoo).
The recognizer used here has SHA-256
`4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43`.
InsightFace's published weights are for non-commercial research. Do not stage
weights, personal captures, generated vectors, or the virtual environment.

The tested setup is Python 3.12.10, Ultralytics 8.4.173, PyTorch 2.14.1+cpu,
OpenCV 5.0.0, NumPy 2.5.2, ONNX Runtime 1.31.1 and Tk 8.6.
