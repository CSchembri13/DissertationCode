from os import environ

from google.cloud import translate


PROJECT_ID = environ.get("PROJECT_ID", "my-project-14198-1690529146451")
assert PROJECT_ID
PARENT = f"projects/{PROJECT_ID}"


def translate_text(text: str, target_language_code: str) -> translate.Translation:
    client = translate.TranslationServiceClient()

    response = client.translate_text(
        parent=PARENT,
        contents=[text],
        target_language_code=target_language_code,
    )

    return response.translations[0]
    

text = "Hello Kif Intom"
target_languages = ["en"]

print(f" {text} ".center(50, "-"))
for target_language in target_languages:
    translation = translate_text(text, target_language)
    source_language = translation.detected_language_code
    translated_text = translation.translated_text
    print(f"{source_language} → {target_language} : {translated_text}")
    