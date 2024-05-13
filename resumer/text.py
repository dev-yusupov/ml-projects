import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist


def text_summarization(text, num_sentences=3):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Tokenize the text into words
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Calculate word frequency
    word_freq = FreqDist(filtered_words)

    # Assign a score to each sentence based on word frequency
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                if len(sentence.split(' ')) < 30:  # Limiting sentence length
                    if sentence not in sentence_scores:
                        sentence_scores[sentence] = word_freq[word]
                    else:
                        sentence_scores[sentence] += word_freq[word]

    # Get the top 'num_sentences' sentences with highest scores
    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]

    # Join the sentences to form the summary
    summary = ' '.join(summary_sentences)

    return summary

# Example text for summarization
example_text = """
Natural language processing (NLP) is a subfield of linguistics, computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data.
Text summarization is the process of distilling the most important information from a source (or sources) to produce an abridged version for a particular user (or users) and task (or tasks).
There are two main approaches to automatic text summarization: extractive and abstractive. Extractive methods involve selecting a subset of existing words, phrases, or sentences in the original text to form the summary. Abstractive methods generate new sentences from the original text to capture the key points.
"""

# Perform text summarization
summary = text_summarization(example_text)
print("Summary:")
print(summary)
