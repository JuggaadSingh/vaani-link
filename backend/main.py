from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from transformers import AutoModel

import torch
import soundfile as sf

import subprocess
import os
import uuid


# ==========================================
# APP
# ==========================================

app = FastAPI(
    title="Vaani-Link API"
)


# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# LOAD STT MODEL
# ==========================================

MODEL_NAME = "ai4bharat/indic-conformer-600m-multilingual"

print("Loading IndicConformer model...")


model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
    token=os.getenv("HF_TOKEN")
)


model.eval()


print("STT model loaded successfully!")


# ==========================================
# WEBSOCKET ROOMS
# ==========================================

rooms = {}


# ==========================================
# HOME
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Vaani-Link backend is running"
    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ==========================================
# TRANSCRIBE AUDIO FUNCTION
# ==========================================

def transcribe_audio(
    audio_path,
    language="hi"
):

    print("Loading audio...")


    audio, sample_rate = sf.read(
        audio_path
    )


    print(
        f"Sample rate: {sample_rate}"
    )


    print(
        f"Audio shape: {audio.shape}"
    )


    # Convert audio to PyTorch tensor

    wav = torch.tensor(
        audio,
        dtype=torch.float32
    )


    # Convert stereo to mono

    if wav.ndim > 1:

        wav = wav.mean(
            dim=1
        )


    # Add batch dimension

    wav = wav.unsqueeze(
        0
    )


    print("Transcribing...")


    # Speech to Text

    with torch.no_grad():

        transcription = model(
            wav,
            language,
            "ctc"
        )


    return transcription


# ==========================================
# TRANSCRIBE ENDPOINT
# ==========================================

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = "hi"
):

    # Generate unique filenames

    unique_id = str(
        uuid.uuid4()
    )


    input_filename = (
        f"temp_{unique_id}.webm"
    )


    output_filename = (
        f"temp_{unique_id}.wav"
    )


    try:

        # ----------------------------------
        # SAVE UPLOADED AUDIO
        # ----------------------------------

        contents = await file.read()


        with open(
            input_filename,
            "wb"
        ) as audio_file:

            audio_file.write(
                contents
            )


        print(
            f"Saved audio: {input_filename}"
        )


        # ----------------------------------
        # CONVERT WEBM TO WAV
        # ----------------------------------

        print(
            "Converting audio to WAV..."
        )


        subprocess.run(

            [

                "ffmpeg",

                "-y",

                "-i",

                input_filename,

                "-ar",

                "16000",

                "-ac",

                "1",

                output_filename

            ],

            check=True

        )


        print(
            "Audio converted successfully!"
        )


        # ----------------------------------
        # TRANSCRIBE
        # ----------------------------------

        transcription = transcribe_audio(

            output_filename,

            language

        )


        print(
            "Transcription:",
            transcription
        )


        # ----------------------------------
        # RETURN RESULT
        # ----------------------------------

        return {

            "success": True,

            "transcription":
                transcription,

            "language":
                language

        }


    except Exception as error:

        print(
            "TRANSCRIPTION ERROR:"
        )


        print(error)


        return {

            "success": False,

            "error":
                str(error)

        }


    finally:

        # ----------------------------------
        # CLEAN UP TEMP FILES
        # ----------------------------------

        if os.path.exists(
            input_filename
        ):

            try:

                os.remove(
                    input_filename
                )

            except Exception:

                pass


        if os.path.exists(
            output_filename
        ):

            try:

                os.remove(
                    output_filename
                )

            except Exception:

                pass


# ==========================================
# WEBSOCKET COMMUNICATION
# ==========================================

@app.websocket(
    "/ws/{room_id}"
)
async def websocket_endpoint(

    websocket: WebSocket,

    room_id: str

):

    # Accept connection

    await websocket.accept()


    # Create room if it doesn't exist

    if room_id not in rooms:

        rooms[room_id] = []


    # Add device to room

    rooms[room_id].append(
        websocket
    )


    print(
        f"🟢 Device connected to room: {room_id}"
    )


    print(
        f"Devices in room: {len(rooms[room_id])}"
    )


    try:

        while True:


            # Receive message

            data = await websocket.receive_text()


            print(
                f"📩 Message received in {room_id}:"
            )


            print(
                data
            )


            # Broadcast message to
            # OTHER devices in same room

            disconnected_clients = []


            for client in rooms[room_id]:


                # Don't send message back
                # to sender

                if client == websocket:

                    continue


                try:

                    await client.send_text(
                        data
                    )


                except Exception:

                    disconnected_clients.append(
                        client
                    )


            # Remove disconnected devices

            for client in disconnected_clients:

                if client in rooms[room_id]:

                    rooms[room_id].remove(
                        client
                    )


    except WebSocketDisconnect:


        print(
            f"🔴 Device disconnected from room: {room_id}"
        )


        # Remove device

        if (
            room_id in rooms
            and websocket in rooms[room_id]
        ):

            rooms[room_id].remove(
                websocket
            )


        # Delete empty room

        if (
            room_id in rooms
            and len(rooms[room_id]) == 0
        ):

            del rooms[room_id]


    except Exception as error:


        print(
            "WEBSOCKET ERROR:"
        )


        print(
            error
        )


        # Remove problematic connection

        if (
            room_id in rooms
            and websocket in rooms[room_id]
        ):

            rooms[room_id].remove(
                websocket
            )


# ==========================================
# END
# ==========================================