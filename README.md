# Image-captioning
Multilingual assistive model that helps visually impaired users by describing images in multiple Indian languages and narrating the descriptions via audio.

https://github.com/user-attachments/assets/e3de1cdc-ee37-46d3-8683-b83234dcd727

## `caption_translate_speech` Function

The `caption_translate_speech` function performs three key tasks: generating a caption for an image, translating that caption into a specified language, and converting the translated caption into speech. Below is a detailed description of the function:

### Function Definition

```python
def caption_translate_speech(image_path, language='hin_Deva', speech_lang='hi'):
```

**Arguments:**
- `image_path`: The file path or URL of the image for which the caption is to be generated.
- `language`: The target language into which the caption will be translated. The default value is Hindi in Devanagari script (`hin_Deva`).
- `speech_lang`: The language for the text-to-speech output. The default value is Hindi (`hi`).

### Steps Performed by the Function

1. **Display Image:**
    ```python
    display(load_image(image_path))
    ```
    - This line loads and displays the image at the specified path or URL. The `load_image` function is used to handle both local files and URLs.

2. **Generate Caption:**
    ```python
    caption = get_caption(model, image_processor, tokenizer, image_path)
    print("Generated Caption:", caption)
    ```
    - The image is processed by a Vision Transformer (ViT) model to generate a descriptive caption. The caption is then printed.

3. **Translate Caption:**
    ```python
    translation = translate_text(caption, language)
    print("Translated Caption:", translation)
    ```
    - The generated caption is translated into the specified target language using the IndicTrans2 model. The translated caption is printed.

4. **Convert Text to Speech:**
    ```python
    return text_to_speech_colab(translation, lang=speech_lang)
    ```
    - The translated caption is converted into speech using the `gTTS` library. The function `text_to_speech_colab` generates and plays an audio file with the translated text in the selected speech language.

### Purpose

This function is designed to:
1. Load and display an image.
2. Generate a caption for the image using a pre-trained deep learning model.
3. Translate the generated caption into a specified Indian language.
4. Provide audio output of the translated caption, making it accessible to users with visual impairments.

## Project Structure

```
Image-Captioning/
├── app.py                     # Flask web application for serving the model and handling API requests
├── requirements.txt           # List of dependencies required for the project
├── Multilingual Image Captioning and Speech Synthesis.ipynb  # Jupyter notebook for testing and experimenting with models
├── static/                    # Folder for static assets
│   ├── uploads/               # Folder where uploaded images are stored
│   └── audio/                 # Folder where generated audio files are stored
└── templates/                 # Folder containing HTML templates
    └── index.html             # Main landing page HTML file
```

## Installation

To set up the project locally, follow these steps:

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/shivamlth27/Image-captioning.git
   cd Image-Captioning
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python3 -m venv aiml
   source aiml/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Run the Flask app:
   ```bash
   python app.py
   ```

   The application will start running locally on `http://127.0.0.1:5000/`.

2. Open a web browser and visit the application to interact with the API.

## Usage

1. **Upload an Image**: Use the web interface to upload an image file or provide an image URL.
2. **Generate Caption**: After uploading the image, the model will generate a caption for the image.
3. **Translate the Caption**: You can select a target language, and the caption will be translated into that language.
4. **Generate Speech**: The translated caption will be converted into speech, and an audio file will be provided for download.

## Models Used

- **Image Captioning**: `VisionEncoderDecoderModel` from Hugging Face (ViT + GPT-2)
- **Translation**: IndicTrans (for multilingual translation)
- **Speech Synthesis**: Google Text-to-Speech (gTTS)


>>>>>>> a4ab653 (add)
