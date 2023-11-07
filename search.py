import os
import sys
import pickle
from index import MyIndex
from nltk.tokenize import word_tokenize
from itertools import product

MAX_VALUE = 100000

'''
found_doc = {'doc_id': {
    'distance': MAX_VALUE, 
    'order': 0, 
    'postings': {'line': {'token': ['pos']}}
    }
}
'''

def get_search_result(query_words, index):
    # get the rough search result of all query words
    whole_index = index.get_whole_index() # TODO

    intersection = filter_intersection(query_words, whole_index)
    
    if intersection == []:
        return {}
    
    found_doc = {}
    for word in query_words:
        for doc_id in whole_index[word]:
            # word found in inverted index
            if len(query_words) > 1 and int(doc_id) not in intersection:
                continue
            if doc_id not in found_doc:
                # document id not yet added
                found_doc[doc_id] = {'distance': MAX_VALUE, 'order': 0, 'postings': {}}
            for line, position_list in whole_index[word][doc_id].items():
                if line not in found_doc[doc_id]['postings']:
                    # line not yet added
                    found_doc[doc_id]['postings'][line] = {}
                    for i in query_words:
                        # initialise each word with empty array for each line
                        found_doc[doc_id]['postings'][line][i] = []
                for i in position_list:
                    # add each position for line containg word
                    found_doc[doc_id]['postings'][line][word].append(i)
    return found_doc

def filter_intersection(query_words, whole_index):
    # find intersection between all query words on each document found
    found = {}
    for word in query_words:
        if word not in whole_index:
            # token not found in inverted index
            return []
        found[word] = (whole_index[word])
    intersection = []
    for _, value in found.items():
        # intersection between documents
        line = []
        for doc_id, _ in value.items():
            line.append(int(doc_id))
        if intersection != []:
            intersection = list(set(line) & set(intersection))
        else:
            intersection = line

    return intersection

def get_distance(postings_list):
    # calculate minimum proximity distance between query words
    cartesian_products = []
    for word in postings_list:
        if postings_list[word]:
            cartesian_products.append(postings_list[word])
    # get all cartesian_products of query words positions
    cartesian_products = list(product(*cartesian_products))
    
    min_distance = MAX_VALUE
    min_positions = None
    for combination in cartesian_products:
        # calculate the proximity distance for each combination
        # to find the closest distance combination
        distance = 0
        for i in range(len(combination) - 1):
            distance += abs(combination[i] - combination[i + 1])
        if distance < min_distance:
            min_distance = distance
            min_positions = combination
    
    return min_distance, min_positions

def get_words_order(found_doc, query_words):
    # get the number of words appear in order
    
    # get all positions for each query word in document, 
    # including words from different lines
    all_positions = {}
    for word in query_words:
        all_positions[word] = []
    for postings in found_doc.values():
        for _, words in postings['postings'].items():
            for word, positions in words.items():
                all_positions[word].extend(positions)
    
    # sort positions
    for word in all_positions:
        all_positions[word].sort()
    
    # find the number of words appear in order
    order = 0
    prev_position = -1
    for word in query_words:
        for position in all_positions[word]:
            if position > prev_position:
                order += 1
                prev_position = position
                break # TODO
    
    return order

def sort_output(found_doc, query_words):
    results = {}
    for doc_id, doc in found_doc.items():
        # get all positions and line id of query words in document
        all_positions = {}
        line_positions = {}
        for word in query_words:
            all_positions[word] = []
            line_positions[word] = {}
        for line_id, postings in doc['postings'].items():
            for word, positions in postings.items():
                all_positions[word].extend(positions)
                for position in positions:
                    line_positions[word][position] = line_id
        
        # get minimum proximity distance between query words and their positions
        if all(all_positions.values()):
            distance, best_position = get_distance(all_positions)
        else:
            distance, best_position = MAX_VALUE, []
        
        # get the number of words appear in order
        order = 0
        if distance != MAX_VALUE:
            order = get_words_order({doc_id: doc}, query_words)
        
        # get the position of the best query words combination
        best_lines = set()
        if best_position:
            for word, position in zip(query_words, best_position):
                best_lines.add(line_positions[word][position])
        best_lines = sorted(best_lines)
        results[doc_id] = {'distance': distance, 'order': order, 'lines': best_lines}
    
    # sort by proximity distance, then query term order, last by document id
    sorted_list = sorted(results.items(), key=lambda i : (i[1]['distance'], i[1]['order'], int(i[0])))
    output = []
    for i in sorted_list:
        output.append((i[0], i[1]['lines']))
        
    return output

def search(query, index):
    special = False
    if query[:2] == "> ":
        # TODO
        # displaying lines containing matching terms
        special = True
        query = query[2:]
    
    # perform same preprocessing as tokens stored in inverted index
    query_words = index.preprocessing(query)
    
    # get rough search results
    found_doc = get_search_result(query_words, index)
    
    results = sort_output(found_doc, query_words)
    
    output = []
    input_path = index.get_input_path()
    if special:
        # output doc_id and lines where best combination query words found
        for doc_id, line_list in results:
            output.append(f'> {doc_id}')
            with open(os.path.join(input_path, doc_id), 'r') as f:
                line_num = 0
                for line in f:
                    if len(line_list) > 0 and line_num == line_list[0]:
                        output.append(line.rstrip('\n'))
                        line_list.pop(0)
                    if len(line_list) == 0:
                        break
                    line_num += 1
    else:
        # output doc_id only
        for result in results:
            output.append(result[0])
        

    return output

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Error: python3 search.py [folder-of-indexes]")

    index_path = sys.argv[1]
    if not os.path.isdir(index_path):
        sys.exit(f"Error: {index_path} is not a directory")
    
    for filename in os.listdir(index_path):
        # open as read-only
        with open(os.path.join(index_path, filename), 'rb') as f:
            # TODO: make sure only one index file
            inverted_index = pickle.load(f)
            break

    if sys.stdin.isatty():
        # input from stdin
        while True:
            try:
                query = input()
                result = search(query, inverted_index)
                for i in result:
                    print(i)
            except EOFError:
                break
    else:
        # input redirected from file
        for query in sys.stdin:
            try:
                query = query.strip()
                result = search(query, inverted_index)
                for i in result:
                    print(i)
            except EOFError:
                break
    