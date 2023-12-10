from lsa import LSAmodel
from lda import LDAModel
from dataset import MSMARCO
from bert import BERTSearchEngine
from relevant_docs import RelevanceScorer
from argparse import ArgumentParser
from transformers import BertTokenizer, BertModel
from dataset import MSMARCO


def calculate_average_precision(retrieved_doc_indices, relevant_doc_indices):
    # Calculate the average precision for a single query
    relevant_count = 0
    precision_sum = 0
    num_relevant_retrieved = 0

    # Iterate over each retrieved document
    for i, doc_id in enumerate(retrieved_doc_indices):
        if doc_id in relevant_doc_indices:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precision_sum += precision_at_i
            num_relevant_retrieved += 1

    # Calculate the average precision
    if num_relevant_retrieved == 0:
        return 0
    average_precision = precision_sum / num_relevant_retrieved
    return average_precision


def calculate_performance_metrics(relevant_pairs, results):
    # print the top 10 relevant documents with their scores
    print(f'No of relevant documents found: {len(results)}')
    print(f"Top 10 relevant documents: ")
    format_results(results[:10])

    # get the doc ids from the results and relevant pairs
    doc_ids_found = get_docs_index(results)
    doc_ids_actual = get_docs_index(relevant_pairs)

    # Convert lists to sets for easy computation
    relevant_docs_found_set = set(doc_ids_found)
    actual_relevant_docs_set = set(doc_ids_actual)

    # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
    TP = len(relevant_docs_found_set.intersection(actual_relevant_docs_set))
    FP = len(relevant_docs_found_set) - TP
    FN = len(actual_relevant_docs_set) - TP

    # Calculate Precision, Recall, and F1 Score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    avg_precision = calculate_average_precision(doc_ids_found, doc_ids_actual)

    print(f"Average Precision: {round(avg_precision, 4)}")
    print(f"Precision: {round(precision, 4)}")
    print(f"Recall: {round(recall, 4)}")
    print(f"F1 Score: {round(f1_score, 4)}")

def format_results(results):
    # format the results
    for i, (doc_id, score) in enumerate(results):
        print(f"{i+1}. Doc ID: {doc_id}, Score: {score}")

def get_docs_index(results):
    # get the doc ids from the results
    doc_ids = [doc_id for doc_id, score in results]
    return doc_ids

def query_lsa(query):
    lsa = LSAmodel()
    lsa.load_model('lsa_model.pny')
    results = lsa.query(query)
    relevant_pairs = get_relevant_docs(query)
    calculate_performance_metrics(relevant_pairs, results)

def query_lda(query):
    lda = LDAModel()
    lda.load_model('lda_model.pny')
    results = lda.query(query)
    relevant_pairs = get_relevant_docs(query)
    calculate_performance_metrics(relevant_pairs, results)

def query_bert(query):
    # initialize the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    # save the embeddings for the BERT model
    bert = BERTSearchEngine(tokenizer, model)
    bert.load_embeddings("bert_embeddings.npy")
    results = bert.query(query)
    relevant_pairs = get_relevant_docs(query)
    calculate_performance_metrics(relevant_pairs, results)

def get_relevant_docs(query):
    # get the no of relevant docs for the query
    # if query mode is selected, load the embeddings
    relevance_scorer = RelevanceScorer(mode='query', threshold=0.5)
    relevance_scorer.load_embeddings('glove_embeddings.npy')
    results = relevance_scorer.query(query)
    return results

def main(args, marcodata):

    # preprocess the query
    query = args.query
    query = marcodata.preprocess([query])

    # query the model based on the model selected
    if args.model == 'lsa':
        query_lsa(query)
    elif args.model == 'lda':
        query_lda(query)
    elif args.model == 'bert':
        query_bert(query)


if __name__ == '__main__':

    parser = ArgumentParser()

    # define arguments for the parser
    parser.add_argument('--model', type=str, help='Model to use for query processing', required=True)
    parser.add_argument('--query', type=str, help='Query to search for', required=True)
    parser.add_argument('--similarity', type=str, default = 'cosine',
                        choices= ['cosine', 'dot_product', 'jensen_shannon', 'matrix_similarity'], 
                        help='Similarity metric to use')
    
    # parse and print the arguments
    args = parser.parse_args()
    print(args)

    # load the dataset based on the model
    if args.model == 'lsa':
        marcodata = MSMARCO(model='lsa')
    elif args.model == 'lda':
        marcodata = MSMARCO(model='lsa')
    elif args.model == 'bert':
        marcodata = MSMARCO(model='bert')

    # run the main function
    main(args, marcodata)