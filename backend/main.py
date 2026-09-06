from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from faster_whisper import WhisperModel

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
# LOAD LIGHTWEIGHT STT MODEL
# ==========================================

print("Loading lightweight Whisper STT model...")

model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
)

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

def transcribe_audio(audio_path, language="hi"):

    print("Transcribing audio...")

    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=1
    )

    transcription = " ".join(
        segment.text.strip()
        for segment in segments
    )

    print("Detected language:", info.language)
    print("Transcription:", transcription)

    return transcription


# ==========================================
# TRANSCRIBE ENDPOINT
# ==========================================

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = "hi"
):

    unique_id = str(uuid.uuid4())

    input_filename = f"temp_{unique_id}.webm"
    output_filename = f"temp_{unique_id}.wav"

    try:

        # ----------------------------------
        # SAVE UPLOADED AUDIO
        # ----------------------------------

        contents = await file.read()

        with open(input_filename, "wb") as audio_file:
            audio_file.write(contents)

        print(f"Saved audio: {input_filename}")


        # ----------------------------------
        # CONVERT WEBM TO WAV
        # ----------------------------------

        print("Converting audio to WAV...")

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

        print("Audio converted successfully!")


        # ----------------------------------
        # TRANSCRIBE
        # ----------------------------------

        transcription = transcribe_audio(
            output_filename,
            language
        )


        # ----------------------------------
        # RETURN RESULT
        # ----------------------------------

        return {
            "success": True,
            "transcription": transcription,
            "language": language
        }


    except Exception as error:

        print("TRANSCRIPTION ERROR:")
        print(error)

        return {
            "success": False,
            "error": str(error)
        }


    finally:

        # ----------------------------------
        # CLEAN UP
        # ----------------------------------

        if os.path.exists(input_filename):

            try:
                os.remove(input_filename)
            except Exception:
                pass


        if os.path.exists(output_filename):

            try:
                os.remove(output_filename)
            except Exception:
                pass


# ==========================================
# WEBSOCKET COMMUNICATION
# ==========================================

@app.websocket("/ws/{room_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    room_id: str
):

    await websocket.accept()

    if room_id not in rooms:
        rooms[room_id] = []

    rooms[room_id].append(websocket)

    print(
        f"🟢 Device connected to room: {room_id}"
    )

    print(
        f"Devices in room: {len(rooms[room_id])}"
    )

    try:

        while True:

            data = await websocket.receive_text()

            print(
                f"📩 Message received in {room_id}:"
            )

            print(data)


            # ----------------------------------
            # BROADCAST TO OTHER DEVICES
            # ----------------------------------

            disconnected_clients = []

            for client in rooms[room_id]:

                if client == websocket:
                    continue

                try:

                    await client.send_text(data)

                except Exception:

                    disconnected_clients.append(client)


            # ----------------------------------
            # REMOVE DISCONNECTED CLIENTS
            # ----------------------------------

            for client in disconnected_clients:

                if client in rooms[room_id]:
                    rooms[room_id].remove(client)


    except WebSocketDisconnect:

        print(
            f"🔴 Device disconnected from room: {room_id}"
        )

        if (
            room_id in rooms
            and websocket in rooms[room_id]
        ):

            rooms[room_id].remove(websocket)


        if (
            room_id in rooms
            and len(rooms[room_id]) == 0
        ):

            del rooms[room_id]


    except Exception as error:

        print("WEBSOCKET ERROR:")
        print(error)

        if (
            room_id in rooms
            and websocket in rooms[room_id]
        ):

            rooms[room_id].remove(websocket)