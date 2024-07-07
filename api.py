from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import pipeline
import tempfile
from starlette.responses import RedirectResponse

app = FastAPI()

# Load the video classification pipeline
pipe = pipeline("video-classification", model="MANMEET75/videomae-base-finetuned-HumanActivityRecognition")


@app.get("/", include_in_schema=False)
async def read_root():
    return RedirectResponse(url="/docs")

@app.post("/classify-video/")
async def classify_video(video_file: UploadFile = File(...)):
    # Check if the file is an MP4
    if not video_file.filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only MP4 files are allowed.")
    
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(await video_file.read())
        temp_video_path = temp_file.name

    # Perform video classification
    try:
        results = pipe(temp_video_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error classifying video: {str(e)}")

    # Prepare classification results
    classification_results = []
    for result in results:
        classification_results.append({
            "label": result['label'],
            "score": round(result['score'], 4)
        })

    return {
        "message": "Classification completed successfully!",
        "results": classification_results
    }