from deep_translator import GoogleTranslator

_translator = None

def get_translator():
    global _translator
    if _translator is None:
        _translator = Translator()
    return _translator

def translate_text(text, target_lang):
    if target_lang == 'en':
        return text
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception as e:
        print(f"[Translation] Error: {e}")
        return f"(Translation unavailable) {text}"