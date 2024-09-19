import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os

def send_email_with_attachment(to_email, subject, body, file_path):
    # Replace these with your own credentials
    from_email = "krit@gmail.com"
    password = "your-email-password"

    # Set up the MIME
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    # Body and the attachments for the email
    msg.attach(MIMEText(body, 'plain'))

    # Open the file to be sent
    attachment = open(file_path, "rb")

    # Instance of MIMEBase and named as part
    part = MIMEBase('application', 'octet-stream')

    # To change the payload into encoded form
    part.set_payload((attachment).read())

    # Encode into base64
    encoders.encode_base64(part)

    part.add_header('Content-Disposition', f"attachment; filename= {os.path.basename(file_path)}")

    # Attach the instance 'part' to message instance
    msg.attach(part)

    # Create SMTP session for sending the mail
    server = smtplib.SMTP('smtp.gmail.com', 587)  # Use Gmail SMTP server
    server.starttls()  # Start TLS for security
    server.login(from_email, password)  # Authentication

    # Convert the message into a string and send it
    text = msg.as_string()
    server.sendmail(from_email, to_email, text)
    server.quit()

# Example: send an email after saving the image
if fall_image_saved:
    send_email_with_attachment(
        to_email="recipient-email@gmail.com",
        subject="Fall Detected!",
        body="A fall has been detected. Please see the attached image.",
        file_path=fall_path
    )
    print(f"Email with fall image sent to {to_email}")
