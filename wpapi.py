from flask import Flask, request, jsonify
import requests

# Initialize Flask app
app = Flask(__name__)

# Meta Cloud API credentials
ACCESS_TOKEN = "your_meta_access_token"
VERIFY_TOKEN = "your_verify_token"

# Endpoint for verifying webhook
@app.route('/webhook', methods=['GET'])
def verify_webhook():
    # Verification token from Meta Cloud API
    mode = request.args.get('hub.mode')
    token = request.args.get('hub.verify_token')
    challenge = request.args.get('hub.challenge')

    if mode == 'subscribe' and token == VERIFY_TOKEN:
        return challenge, 200
    else:
        return 'Verification failed', 403

# Endpoint for receiving messages
@app.route('/webhook', methods=['POST'])
def receive_message():
    data = request.get_json()

    # Check if the message is from WhatsApp
    if 'object' in data and data['object'] == 'whatsapp_business_account':
        for entry in data['entry']:
            for change in entry['changes']:
                if 'messages' in change['value']:
                    messages = change['value']['messages']
                    for message in messages:
                        if message['type'] == 'text':
                            sender_id = message['from']
                            text_received = message['text']['body']
                            send_message(sender_id, f"Received: {text_received}")
    return 'EVENT_RECEIVED', 200

# Function to send a reply via WhatsApp API
def send_message(recipient_id, message_text):
    url = f"https://graph.facebook.com/v15.0/{recipient_id}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": recipient_id,
        "text": {"body": message_text}
    }
    response = requests.post(url, json=payload, headers=headers)
    print(f"Message sent: {response.status_code}")

# Run the Flask app
if __name__ == '_main_':
    app.run(port=5000)
import openai

openai.api_key = "your_openai_api_key"

def generate_ai_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50
    )
    return response["choices"][0]["text"].strip()

# Modify send_message function to use AI responses
def send_message(recipient_id, user_message):
    ai_reply = generate_ai_response(user_message)
    url = f"https://graph.facebook.com/v15.0/{recipient_id}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": recipient_id,
        "text": {"body": ai_reply}
    }
    response = requests.post(url, json=payload, headers=headers)