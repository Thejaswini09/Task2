import spacy
import re
import pickle
from nltk.stem import WordNetLemmatizer
import contractions
import logging 

logging.basicConfig(filename='textPreprocessing.log', level=logging.INFO, format='%(asctime)s %(message)s')

class NLPProcessor:

    #Initialise Spacy Model for the Pipeline, Load the dictionary of Emojies, Initialise the default stopwords and adding custom if any     
    def __init__(self, custom_stopwords=None):
        self.custom_stopwords = custom_stopwords if custom_stopwords else []
        self.nlp  = spacy.load("en_core_web_sm")
        with open ('Emoji_Dict.p', 'rb') as fp:
            self.Emoji_Dict = pickle.load(fp)
        self.Emoji_Dict = {v: k for k, v in self.Emoji_Dict.items()}
        self.nlp.Defaults.stop_words |= set(self.custom_stopwords)  

    #'Tokenising the text, Usage of Spacy [chosen for better effeciency and speed for large text processing]
    def tokenize_spacy(self, doc):
        doc = self.nlp(doc)
        tokens = [token.text for token in doc]
        logging.info("Tokenisation Initiated")
        return tokens
    
    # lemmatisation offers better and meaningful lemmas over stemming considerering the performance in different use cases
    def lemmatization(self, tokens):
        wnl = WordNetLemmatizer()
        lemmatized_tokens = [wnl.lemmatize(token) for token in tokens]
        logging.info("lemmatisation initiated")
        return lemmatized_tokens
    
    # Special Characters are removed and the emoji are replaced with emoji description as in the loaded dictionary 
    def handle_special_characters(self, text):
        for emot in self.Emoji_Dict:
            text = re.sub(r'('+emot+')', "_".join(self.Emoji_Dict[emot].replace(",","").replace(":","").split()), text)
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        logging.info("special characters have been handled")
        return text
    
    # Contractions are handled by expanding the words 
    def handle_contractions(self, text):
        expanded_text = contractions.fix(text)
        logging.info("Contracted words have been de-contracted")
        return expanded_text
    
    # Negated words are prefixed with NOT to identify during Feature Extraction 
    def handle_negations(self, text):
        negations = {"not", "no", "never", "none", "neither", "nor", "cannot", "can't", "n't"}
        logging.info("The negated words to be identified:", negations)
        result_words = []
        doc = self.nlp(text)
        i = 0
        while i < len(doc):
            token = doc[i]
            if token.lower_ in negations:
                result_words.append(token.text)
                i += 1
                if i < len(doc): 
                    next_token = doc[i]
                    # Append the next token with NOT
                    result_words.append("NOT_" + next_token.text)
                    i += 1
            else:
                result_words.append(token.text)
                i += 1
        return " ".join(result_words)

    # To Customise stopwords which is extended on default stop_words if user prompts to add, removed from default(if present) if user prompts to remove 
    def customize_stopwords(self, add_words=None, remove_words=None):
        if add_words:
            logging.info("Added the requested stopwords:", add_words)
            self.custom_stopwords.extend(add_words)
        if remove_words:
            logging.info("Removed the requested stopwords:", remove_words)
            for word in remove_words:
                if word in self.custom_stopwords:
                    self.custom_stopwords.remove(word)
        self.nlp.Defaults.stop_words |= set(self.custom_stopwords)

    # Remove stopwords     
    def remove_stopwords(self, tokens):
        tokens = [w for w in tokens if not w.lower() in self.nlp.Defaults.stop_words]
        return tokens 

    # Linker function - Pipeline to read the Input and write to the Output 
    def preprocess_file(self, input_file, output_file = 'output.txt'):
        import os
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()

        text = self.handle_contractions(text)
        text = self.handle_special_characters(text)
        text = self.handle_negations(text)
        tokens = self.tokenize_spacy(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatization(tokens)
        processed_text = " ".join(tokens)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(processed_text)


# Main function 
if __name__ == "__main__":
    processor = NLPProcessor()
    processor.customize_stopwords(add_words=['no', 'ok', 'common'])
    processor.preprocess_file('input.txt')
