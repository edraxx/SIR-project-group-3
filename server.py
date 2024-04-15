from sic_framework.devices.pepper import Pepper

PORT = 8080
MACHINE_IP = '10.0.0.212' # Change machine IP to your current WIFI IP address
PEPPER_IP = '10.0.0.148'
AUTOCUE_DIR = 'html_files' # Directory containing the HTML files
BASE_WEB_URL = f'http://{MACHINE_IP}:{PORT}/text.html' # Base URL to the local webpage in the autocue directory
HTML_FILE = 'text.html'

# Initialize Pepper
pepper = Pepper(ip = PEPPER_IP)
web_url = BASE_WEB_URL + HTML_FILE # Construct the complete web URL
url_message = UrlMessage(web_url) # Create a UrlMessage with the web URL
# Determine the display duration based on the file
display_duration = None # No specific delay after displaying the final message
# Send the message to Pepper's tablet display component
tts.tablet_display_url.send_message(url_message)
# Wait for the specified duration if defined
if display_duration:
    time.sleep(display_duration) # Globals
