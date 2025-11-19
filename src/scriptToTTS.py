import os
import argparse
from dotenv import load_dotenv
from elevenlabs import ElevenLabs, Voice, VoiceSettings

# Initialize the Eleven Labs client
load_dotenv()

api_key = os.getenv("ELEVENLABS_API_KEY")
if not api_key:
    raise EnvironmentError("ELEVENLABS_API_KEY is not set. Please configure it in your environment before running this script.")

client = ElevenLabs(api_key=api_key)

def generate_ai_voice(script_path, output_audio_path, voice_id="N2lVS1w4EtoT3dr4eOWO"):
    try:
        # Read the script
        with open(script_path, "r") as file:
            script_text = file.read()

        # Generate the audio using the Eleven Labs API (returns a generator)
        audio_generator = client.generate(
            text=script_text,
            voice=Voice(
                voice_id=voice_id,
                settings=VoiceSettings(
                    stability=0.8,
                    similarity_boost=0.6,
                    style=0.2,
                    use_speaker_boost=True
                )
            )
        )

        # Write the generated audio to the output file
        with open(output_audio_path, "wb") as audio_file:
            for chunk in audio_generator:  # Iterate over the audio chunks
                audio_file.write(chunk)

        print(f"Audio saved successfully to {output_audio_path}")

    except Exception as e:
        print(f"Error generating AI voice: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate narration audio using ElevenLabs.")
    parser.add_argument("script_path", nargs="?", default="data/script/narration_script.txt", help="Path to the narration script text file")
    parser.add_argument("output_audio_path", nargs="?", default="data/audio/narration_audio.mp3", help="Output path for the generated narration audio")
    parser.add_argument("--voice-id", default="N2lVS1w4EtoT3dr4eOWO", help="ElevenLabs voice ID to use")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_ai_voice(args.script_path, args.output_audio_path, voice_id=args.voice_id)
