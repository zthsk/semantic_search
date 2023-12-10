from lsa import LSAmodel
from lda import LDAModel
from dataset import MSMARCO
from sbert import SBERTSearchEngine
from sentence_transformers import SentenceTransformer
from relevant_docs import RelevanceScorer
from argparse import ArgumentParser


def get_datasets():
    # initialize the dataset
    lsa_lda_marco = MSMARCO(model='lsa') # LSA and LDA require the same preprocessing
    bert_marco = MSMARCO(model='bert')

    # load the data
    marco_data = lsa_lda_marco.load_data()
    bert_marco_data =  bert_marco.load_data()

    # preprocess the data if not already preprocessed
    if not lsa_lda_marco.check_preprocessed_data():
        marco_data = lsa_lda_marco.get_preprocessed_data()
    elif not bert_marco.check_preprocessed_data():
        bert_marco_data = bert_marco.get_preprocessed_data()

    return marco_data, bert_marco_data

def main(args):
    # get the datasets
    marco_data, bert_marco_data = get_datasets()

    # train the models and save them to the specified path
    if args.lsa:
        print("LSA model path provided:", args.lsa)
        lsa = LSAmodel(n_components=300)
        lsa.fit(marco_data)
        lsa.save_model(args.lsa)

    elif args.lda:
        print("LDA model path provided:", args.lda)
        lda = LDAModel(num_topics=300, chunksize=5000)
        lda.train(marco_data)
        lda.save_model(args.lda)

    elif args.bert:
        print("BERT model path provided:", args.bert)
        # save the embeddings for the BERT model
        bert = SBERTSearchEngine()
        bert.save_embeddings(bert_marco_data, args.bert)

    elif args.glove:
        print("GloVe model path provided:", args.glove)
        #save the glove embeddings for the relavance scorer
        glove_file = 'glove.840B.300d.txt'
        relevance_scorer = RelevanceScorer(glove_file=glove_file, threshold=0.7)
        relevance_scorer.train(marco_data, args.glove)



#Defining the initial function which trains all the models and saves them
if __name__ == '__main__':

    parser = ArgumentParser()

    # Create a mutually exclusive group
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--lsa', help='Path to save the LSA model')
    group.add_argument('--lda', help='Path to save the LDA model')
    group.add_argument('--bert', help='Path to save the BERT model')
    group.add_argument('--glove', help='Path to save the GloVe model')

    # Parse arguments
    args = parser.parse_args()

    print(args)


    main(args)




        