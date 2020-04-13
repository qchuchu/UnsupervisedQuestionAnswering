def create_graph_of_words(window, tokens):
    n = len(tokens)
    graph_of_words = {}
    for i in range(n):
        current_word = tokens[i]
        for j in range(i + 1, min(i + window, n)):
            observed_word = tokens[j]
            if observed_word in graph_of_words:
                graph_of_words[observed_word].update([current_word])
            else:
                graph_of_words[observed_word] = {current_word}
    return graph_of_words


def merge_and_postings_list(posting_term1, posting_term2):
    result = []
    index1 = 0
    index2 = 0
    n1 = len(posting_term1)
    n2 = len(posting_term2)
    while index1 < n1 and index2 < n2:
        if posting_term1[index1] == posting_term2[index2]:
            result.append(posting_term1[index1])
            index1 += 1
            index2 += 1
        elif posting_term1[index1] > posting_term2[index2]:
            index2 += 1
        elif posting_term1[index1] < posting_term2[index2]:
            index1 += 1
    return result


def merge_or_postings_list(posting_term1, posting_term2):
    result = set(posting_term1)
    result.update(posting_term2)
    return sorted(list(result))
