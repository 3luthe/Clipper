from flask import Flask, render_template, request, jsonify
import boto3
import os
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables for AWS credentials
load_dotenv()

# Initialize the Amazon Polly client
polly_kwargs = {}

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_REGION')

if aws_access_key_id and aws_secret_access_key:
    polly_kwargs.update(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

if aws_region:
    polly_kwargs['region_name'] = aws_region

polly = boto3.client('polly', **polly_kwargs)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_voices', methods=['GET'])
def get_voices():
    try:
        # Get a list of available voices from Polly
        response = polly.describe_voices()

        voices = []
        for voice in response['Voices']:
            # Collect voice name and language
            voices.append({
                'VoiceId': voice['Id'],
                'Language': voice['LanguageName'],
                'LanguageCode': voice['LanguageCode']
            })

        return jsonify({'voices': voices})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.get_json()
    text = data.get('text')
    voice = data.get('voice', 'Joanna')  # Default voice is Joanna if none is selected

    if not text:
        return jsonify({'error': 'Text is required'}), 400

    try:
        # Use Amazon Polly to synthesize the speech
        response = polly.synthesize_speech(
            Text=text,
            VoiceId=voice,
            OutputFormat='mp3',
            Engine='standard'
        )

        # Save the audio stream to an MP3 file
        output_path = os.path.join('static', 'output.mp3')
        with open(output_path, 'wb') as file:
            file.write(response['AudioStream'].read())

        # Return the file URL to the frontend
        file_url = os.path.join('static', 'output.mp3')
        return jsonify({'file_url': file_url})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
