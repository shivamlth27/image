from flask import Flask, render_template, request, jsonify, send_file
import os
from PIL import Image
import requests
import torch
from transformers import (
    VisionEncoderDecoderModel, 
    GPT2TokenizerFast, 
    ViTImageProcessor,
    AutoModelForSeq2SeqLM, 
    AutoTokenizer
)
import urllib.parse as parse
import io
from gtts import gTTS
from IndicTransToolkit import IndicProcessor
from flask_caching import Cache

app = Flask(__name__)

# Configure folders
UPLOAD_FOLDER = 'static/uploads'
AUDIO_FOLDER = 'static/audio'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

for folder in [UPLOAD_FOLDER, AUDIO_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['AUDIO_FOLDER'] = AUDIO_FOLDER

# Caching configuration
app.config['CACHE_TYPE'] = 'simple'
cache = Cache(app)

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1

# Initialize IndicProcessor for translation
ip = IndicProcessor(inference=True)

# Language mapping for gTTS
LANG_TO_GTTS = {
    'hin_Deva': 'hi',  # Hindi
    'ben_Beng': 'bn',  # Bengali
    'tam_Taml': 'ta',  # Tamil
    'tel_Telu': 'te',  # Telugu
    'mal_Mlym': 'ml',  # Malayalam
    'kan_Knda': 'kn',  # Kannada
    'guj_Gujr': 'gu',  # Gujarati
    'mar_Deva': 'mr',  # Marathi
    'pan_Guru': 'pa',  # Punjabi
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_url(string):
    try:
        result = parse.urlparse(string)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False

def load_image(image_path):
    try:
        if check_url(image_path):
            response = requests.get(image_path, stream=True)
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
        elif os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
        
        max_size = 1000
        ratio = max_size / max(image.size)
        if ratio < 1:
            new_size = tuple([int(dim * ratio) for dim in image.size])
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    except Exception as e:
        raise Exception(f"Error loading image: {str(e)}")

@cache.cached(timeout=60, key_prefix='caption')
def get_caption(image_path):
    try:
        image = load_image(image_path)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Lazy loading model
        caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(DEVICE).half()
        tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        
        pixel_values = image_processor(
            images=image,
            return_tensors="pt",
            padding=True,
            do_resize=True,
            do_rescale=True,
            size={"height": 224, "width": 224}
        ).pixel_values.to(DEVICE)
        
        with torch.no_grad():
            output_ids = caption_model.generate(
                pixel_values,
                max_length=50,
                num_beams=4,
                return_dict_in_generate=True,
                early_stopping=True
            ).sequences
        
        caption = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return caption.strip()
    
    except Exception as e:
        raise Exception(f"Error generating caption: {str(e)}")

def batch_translate(input_sentences, src_lang, tgt_lang):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]
        
        # Lazy load translation model
        en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"
        translation_tokenizer = AutoTokenizer.from_pretrained(en_indic_ckpt_dir, trust_remote_code=True)
        translation_model = AutoModelForSeq2SeqLM.from_pretrained(
            en_indic_ckpt_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(DEVICE).half()
        translation_model.eval()

        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)
        
        inputs = translation_tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)
        
        with torch.no_grad():
            generated_tokens = translation_model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )
        
        with translation_tokenizer.as_target_tokenizer():
            generated_tokens = translation_tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)
        
        del inputs
        torch.cuda.empty_cache()
    
    return translations

def translate_text(text, tgt_lang):
    return batch_translate([text], "eng_Latn", tgt_lang)[0]

def generate_audio(text, lang_code):
    try:
        gtts_lang = LANG_TO_GTTS.get(lang_code, 'en')
        tts = gTTS(text=text, lang=gtts_lang)
        audio_path = os.path.join(app.config['AUDIO_FOLDER'], f'speech_{hash(text)}.mp3')
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        raise Exception(f"Error generating audio: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/caption', methods=['POST'])
def caption_image():
    try:
        target_lang = request.form.get('language', 'hin_Deva')
        
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filename)
                image_path = filename
        elif 'url' in request.form:
            url = request.form['url']
            if check_url(url):
                image_path = url
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid input. Please provide either a file or a valid URL.'
            })
        
        # Generate caption
        caption = get_caption(image_path)
        
        # Translate caption
        translation = translate_text(caption, target_lang)
        
        # Generate audio
        audio_path = generate_audio(translation, target_lang)
        
        return jsonify({
            'success': True,
            'caption': caption,
            'translation': translation,
            'image_path': image_path,
            'audio_path': audio_path
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_file(os.path.join(app.config['AUDIO_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
