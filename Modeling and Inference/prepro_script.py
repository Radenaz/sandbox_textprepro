import re
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from googletrans import Translator
import contractions

# Initialize necessary components
stpwds_id = stopwords.words('indonesian')
factory = StemmerFactory()
stemmer_id = factory.create_stemmer()
translator = Translator()

# Load the slang dictionary
with open('JSONs/slang_bank.json', 'r') as file:
    slang_dict = json.load(file)

with open('JSONs/exception_words.json', 'r') as file:
    exception_data = json.load(file)

exception_words = set(exception_data["exception_words"])

# Load additional stopwords
with open('JSONs/expand_stopwords.json', 'r') as f:
    data = json.load(f)

custom_stopwords = data.get("expand_stopwords", [])
if not isinstance(custom_stopwords, list):
    raise ValueError("The 'expand_stopwords' key must contain a list.")

stpwds_id = set(stpwds_id)
stpwds_id.update(custom_stopwords)

# Define the preprocessing function
async def text_preprocessing_id(text):
    try:
        # Translate to Bahasa Indonesia
        translated = await translator.translate(text, src='auto', dest='id')
        text = translated.text
    except Exception as e:
        print(f"Translation failed: {e}")
        return None  # Drop text if translation fails
    
    # Expand contractions
    text = contractions.fix(text)

    # Case folding
    text = text.lower()

    # Mention removal
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)

    # Hashtag removal
    text = re.sub(r"#[A-Za-z0-9_]+", " ", text)

    # Newline removal (\n)
    text = re.sub(r"\\n", " ", text)

    # Whitespace removal
    text = text.strip()

    # URL removal
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www.\S+", " ", text)

    # Non-letter removal (retain apostrophes)
    text = re.sub(r"[^A-Za-z\s']", " ", text)
    
    # Repeat letter removal
    text = re.sub(r'(.)\1{2,}', r'\1', text)

    # Tokenization
    tokens = word_tokenize(text)
    
    # Slang words replacement
    tokens = [slang_dict[word] if word in slang_dict else word for word in tokens]

    # Stopwords removal
    tokens = [word for word in tokens if word not in stpwds_id or word in exception_words]

    # Stemming
    tokens = [stemmer_id.stem(word) for word in tokens]

    # Combine tokens
    text = ' '.join(tokens)

    return text
