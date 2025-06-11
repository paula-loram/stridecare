import streamlit as st
import tempfile
from PIL import Image
import cv2
import requests
import json
import pandas as pd
from fastapi import FastAPI
import openai
from io import BytesIO

# ! use only external url !


# backgroung color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #83bbf7;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the image from file
img = Image.open("cover.png")

# Resize the image (width=px, height=2px)
img = img.resize((1000, 500))

# Display the resized image
st.image(img)

st.title("StrideCare")
st.subheader("Analyze your stride, strengthen your run, prevent injuries.")

# input user's params:
st.markdown("<h3 style='font-size:28px;'>Enter your age</h3>", unsafe_allow_html=True)
age = st.number_input("Age", min_value=0, max_value=120, step=1)

st.markdown("<h3 style='font-size:28px;'>Enter your weight (kg)</h3>", unsafe_allow_html=True)
weight = st.number_input("Weight", min_value=0.0, max_value=300.0, step=0.1)

st.markdown("<h3 style='font-size:28px;'>Enter your weight (kg)</h3>", unsafe_allow_html=True)
weight = st.number_input("", min_value=0.0, max_value=300.0, step=0.1)

st.markdown("<h3 style='font-size:28px;'>Select your gender:</h3>", unsafe_allow_html=True)
gender = st.radio("Gender", ["Male", "Female"])

#creating dictionary out of user params
metadata = {
    "age": age,
    "height": height,
    "weight": weight,
    "gender": gender
}

#sending to fastAPI
metadata_dir = {'metadata': metadata}

#getting video uploaded by user
video_file = st.file_uploader("Upload a video of yourself running:", type=["mp4", "mov", "avi"])



if video_file is not None:
    # st.video(video_file)

    # tfile = tempfile.NamedTemporaryFile(delete=False)
    # st.write(video_file)

    if st.button("Predict"):
        st.write("Analyzing video...")
        url = "https://localhost:8000/generate_stickfigure"
        # cap = cv2.VideoCapture(video_file)
        response = requests.post(url, files={"Video": video_file.getvalue()})

        import time
        time.sleep(2)
        st.success("No injury detected.")
        FASTAPI_URL = "http://127.0.0.1:8000/get_stick_fig_video/"

        with st.spinner("Uploading and processing video... This might take a while."):
            try:
                files = {
                    "video": (
                        video_file.name,
                        video_file.getvalue(),
                        video_file.type
                    )
                }
                data = {
                    "data": json.dumps({
                        "age": age,
                        "weight": weight,
                        "height": height,
                        "gender": gender
                    })
                }
                response = requests.post(FASTAPI_URL, files=files, data=data)

                if response.status_code == 200:
                    st.success("Video uploaded and sent for processing successfully!")
                    result = response.json()
                    import base64
                    video_b64 = result.get("video") or result.get("video_base64")
                    angles = result.get("angles_array")  # <-- get the angles list from the response

                    if video_b64:
                        video_bytes = base64.b64decode(video_b64)
                        st.write("Processed video:")
                        st.video(video_bytes, format="video/mp4")
                    else:
                        st.error("No video found in the response.")

                    # Display angles as a dataframe if present
                    if angles:
                        df_angles = pd.DataFrame(angles)
                        st.write("Detected Angles:")
                        st.dataframe(df_angles)
                    else:
                        st.info("No angles data found in the response.")
                else:
                    st.error(f"Error uploading video: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the FastAPI server. Please ensure it's running.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
        # import time
        # time.sleep(2)
        # st.success("No injury detected."



#st.success("No injury detected.")
if st.button("Surprise"):
    st.balloons()

# background_tasks : call on another endpoint
#send metadata w video (op1)

#option 2: sequentially
#but starting with predict endpoint (big), then we can do get stickfigure
# coord + video embedded, returning new video in api
#easier: get videos in: coordinates as result of endpoint
#then second endpoint w metadata + coordinates --> model prediction to get injury classes
#can also display angle plot





####### LLM prompt #######

### Instantiating API ###
app = FastAPI()

### Receiving the meta-inputs from Streamlit ###






# GPT-based report generation

# Injury type dict #these are the same ''MODEL_OUTPUT_LABELS''
# Providing an informed description with no jargon
injury_types = {
    "No injury": "Runner has good form, and is not prone to injury",
    "Foot/Ankle": "Runner shows signs of stress or dysfunction in the foot or ankle, possibly indicating an issue with their arch-heel, an achilles issue, or ankle instability.",
    "Hip/Pelvis": "Runner exhibits movement patterns or weaknesses around the hip/pelvis, which could be caused by their glutes, hip impingement, or poor pelvic stability.",
    "Thigh": "A biomechanical indicator of strain or overuse in the thigh muscles, commonly involving the hamstrings or quadriceps, which may lead to strains or muscle imbalances.",
    "Lower Leg": "Runner may be at risk for conditions like shin splints or calf strain, often due to overuse, poor shock absorption, or inadequate lower-leg strength."
}


# OpenAI key
openai.api_key = st.secrets["ignore, it's on the front-end draft-llm.py"]


# GPT-based report generation
def injury_report(predicted_injury, age, gender, height, weight, max_tokens=250): # Tokens determine the number of words in the summary
    system_prompt = (
        "You are a friendly and helpful Sports Therapist. "
        "You are to explain the runner's injury assessment results using concise and simple language that's easy for the runner to understand. "
        "You avoid medical jargon, speak with warmth, and gently guide the runner on what to do next to address their injury, which involves a Recovery Strategy, Stretching Plan, and Strengthening Plan. "
        "Limit to about 200 words. "
    )

    prompt = f"""
    Our, StrideCare, running-injury assessment AI model predicted: {injury_types.get(predicted_injury, 'Unkown Injury')}

    The runner's details:
    - Age: {age}
    - Gender: {gender}
    - Height: {height}
    - Weight: {weight}

    In a friendly, concise, and detailed manner without jargon, please provide a complete-plan to help the runner address their injury.
    We want to provide a full-complete plan which addresses their {predicted_injury} injury:
        Recovery strategies (e.g. resting, walking, swimming, etc.) which is relevant to their {predicted_injury} injury,
        A specific and tailored Stretching plan addressing their {predicted_injury} injury, we want the names of stretches, how many repetitions/seconds and number of sets,
        and we want a specific and tailored Strengthening plan for the {predicted_injury} injury, including weighted and/or free-weight exercises with ideally as little equipment as possible, please provide the names of these exercises along with number of repetitions and sets.
    Remember, the plan is to ultimately help them run injury free with this tailored plan.
    """

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens
    )

    return response.choices[0].message.content.strip()


## Calling upon injury_report using streamlit
if st.button("Get Injury Report"):
    # Calls upon injury_report with the inputs from the streamlit website
    report = injury_report(predicted_injury, age, gender, height, weight)
    st.subheader("Injury Recovery Plan")
    st.write(report)
