import regex as re


underline_pattern = re.compile(r'\b([a-zA-Z]+(?:_[a-zA-Z]+)*)\b')
camel_case_pattern = re.compile(r'\b(?:[a-z]+[A-Z]|[A-Z]+[a-z])[a-zA-Z]*\b')

def deal_underline(sentence):
    def replace(match):
        return match.group(0).replace('_', ' ')
    
    return underline_pattern.sub(replace, sentence)

def deal_camel_case(sentence):
    def replace(match):
        return re.sub(r'([a-z])([A-Z])', r'\1 \2', match.group(0))
    
    return camel_case_pattern.sub(replace, sentence)

def deal_sentence(sentence):
    # check underline
    sentence = deal_underline(sentence)
    # check camel
    sentence = deal_camel_case(sentence)
    return sentence

from multiprocessing import Pool


def deal_sentence_multiprocess(sentence, num_parts=5):
    sentences = sentence.split()
    part_length = len(sentences) // num_parts
    sentence_parts = [' '.join(sentences[i*part_length:(i+1)*part_length]) for i in range(num_parts)]
    sentence_parts[-1] += ' '.join(sentence[num_parts*part_length:])

    with Pool(num_parts) as pool:
        processed_parts = pool.map(deal_sentence, sentence_parts)

    return ' '.join(processed_parts)

def deal_iri(label):
    if label.startswith("http"):
        l = label.split('/')
        return l[-1]
    return label