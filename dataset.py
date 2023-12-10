 
import re
import nltk
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import os 

nltk.download('punkt')
nltk.download('stopwords')

'''
The dataset used is MSMARCO, which is a large scale dataset for machine reading comprehension and question answering.
The dataset contains a total 1,010,916 questions and answers from real anonymized user queries from the Bing search engine.
We are going to use the MSMARCO dataset to train our LSA, LDA, and BERT models.
'''
class MSMARCO:
    def __init__(self, model=None):
        self.dataset = None
        self.documents = None
        self.model = model
    
    def save_preprocessed_data(self, preprocessed_data):
        """Save the preprocessed data."""
        with open(f"{self.model}_preprocessed_data.txt", 'w') as file:
            for doc in preprocessed_data:
                file.write(f"{doc}\n")

    def check_preprocessed_data(self):
        """Check if preprocessed data exists."""
        return os.path.exists(f"{self.model}_preprocessed_data.txt")

    def load_data(self):
        """Load the MS MARCO dataset."""
        try:
            if self.check_preprocessed_data():
                with open(f"{self.model}_preprocessed_data.txt", 'r') as file:
                    self.documents = file.readlines()
                return self.documents
            else:
                print("Preprocessed data not found. Loading raw data. Run get_preprocessed_data to preprocess the data.")            
                self.dataset = load_dataset('ms_marco', 'v2.1')
                self.documents = [passage['passage_text'] for passage in self.dataset['train']['passages'][:10000]]
                
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
        
        return self.documents

    def preprocess(self, text_list):
        """ Preprocess the text based on the model. """
        docs = ' '.join(text_list)
        docs = docs.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', docs)

        if self.model in ['lda', 'lsa']:
            tokens = word_tokenize(text)
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(word) for word in tokens]
            return ' '.join(tokens)
        elif self.model == 'bert':
            # For BERT, return the minimally processed text
            # Actual tokenization should be done using BERT's tokenizer
            return text
        else:
            raise ValueError("Unsupported model type specified")

    def get_preprocessed_data(self):
        """ Get the preprocessed data. """
        loaded_data = self.load_data()
        data_loader = tqdm(loaded_data, desc=f"Preprocessing {self.model} data.")
        preprocessed_data = [self.preprocess(doc) for doc in data_loader]
        self.save_preprocessed_data(preprocessed_data)
        if loaded_data is None:
            return None  # Or handle the None case as required
        return preprocessed_data



