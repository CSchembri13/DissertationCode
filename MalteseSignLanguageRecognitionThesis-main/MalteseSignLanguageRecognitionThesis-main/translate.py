import cv2
import numpy as np

from os import environ

from google.cloud import translate


PROJECT_ID = environ.get("PROJECT_ID", "my-project-14198-1690529146451")
assert PROJECT_ID
PARENT = f"projects/{PROJECT_ID}"

class Translator:
    def __init__(self, source_language='mt', target_language='en'):
        self.source_language = source_language
        self.target_language = target_language

    def translate_text(text: str, target_language_code: str) -> translate.Translation:
        client = translate.TranslationServiceClient()

        response = client.translate_text(
            parent=PARENT,
            contents=[text],
            target_language_code=target_language_code,
        )

        return response.translations[0]

    def display_translation(self, text):
        translated_text = self.translate_text(text)

        # Create a black image
        image = np.zeros((200, 600, 3), dtype=np.uint8)
        
        # Set some properties for the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (255, 255, 255)
        line_type = 2

        # Text to be displayed
        display_text = f"{self.source_language} → {self.target_language} : {translated_text}"

        # Get the text size to center the text
        text_size = cv2.getTextSize(display_text, font, font_scale, line_type)[0]
        text_x = (image.shape[1] - text_size[0]) // 2
        text_y = (image.shape[0] + text_size[1]) // 2

        # Put the text on the image
        cv2.putText(image, display_text, (text_x, text_y), font, font_scale, font_color, line_type)

        # Display the image in a window
        cv2.imshow('Translation', image)
        
        # Wait for any key press to close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Create an instance of the Translator class
    translator = Translator()
    
    # Example text to translate
    example_text = "Eżempju ta' test biex jiġi tradott"
    translator.display_translation(example_text)
