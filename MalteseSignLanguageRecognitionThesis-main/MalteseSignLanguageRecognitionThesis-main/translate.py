def translateToEnglish(text):
    target_languages = ["en"]
    sentence = ' '.join(text)
    print(f" {sentence} ".center(50, "-"))
    for target_language in target_languages:
        translation = translate_text(sentence, target_language)
        source_language = translation.detected_language_code
        translated_text = translation.translated_text
        print(f"{source_language} → {target_language} : {translated_text}")

        # Set properties for the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (255, 255, 255)
        line_type = 2
        
        # Create an image canvas
        canvas_height = 400
        image_width = 200
        canvas = np.zeros((canvas_height, 600, 3), dtype=np.uint8)
        
        # Display translated text
        display_text = translated_text
        text_size = cv2.getTextSize(display_text, font, font_scale, line_type)[0]
        text_x = (canvas.shape[1] - text_size[0]) // 2
        text_y = 50  # Display the text at y=50 px
        cv2.putText(canvas, display_text, (text_x, text_y), font, font_scale, font_color, line_type)
        
        images = []  # List of images
        words = text
        for word in words:
            image_path = f"MalteseSignLanguageRecognitionThesis-main/images/{word}.jpg"  # Assuming images are named after the words
            img = cv2.imread(image_path)
            if img is not None:
                img = cv2.resize(img, (image_width, image_width))  # Resize to a fixed sizeq
                images.append(img)

        # Positioning images
        start_x = 100  # Starting x position to draw images
        for img in images:
            canvas[text_y + 30:text_y + 30 + image_width, start_x:start_x + image_width] = img
            start_x += image_width + 10  # Move to the right for the next image

        # Display the image in a window
        cv2.imshow('Translation and Images', canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()