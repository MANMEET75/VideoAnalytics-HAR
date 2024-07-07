import streamlit as st
from transformers import pipeline
import tempfile

pipe = pipeline("video-classification", model="MANMEET75/videomae-base-finetuned-HumanActivityRecognition")

st.title("Push-Ups and Pull-Ups Video Classifier")

st.write(
    """
    Upload an MP4 video to classify whether it contains push-ups or pull-ups.
    """
)

uploaded_file = st.file_uploader("Choose an MP4 file", type="mp4")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_video_path = temp_file.name

    st.video(temp_video_path)
    
    with st.spinner('Classifying the video...'):
        results = pipe(temp_video_path)

    st.success("Classification Completed!")
    
    st.subheader("Classification Results:")
    for result in results:
        st.write(f"**Label:** {result['label']}  \n**Score:** {result['score']:.4f}")

    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Detailed Results")
        st.json(results)
        
    with col2:
        st.write("### Result Summary")
        for result in results:
            st.metric(label=result['label'], value=f"{result['score']:.4f}")

