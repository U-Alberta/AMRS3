import spacy
import pysbd

@spacy.Language.component("pysbd")
def pysbd_sentence_boundaries(doc):
    seg = pysbd.Segmenter(language="en", clean=False, char_span=True)
    sents_char_spans = seg.segment(doc.text)
    char_spans = [doc.char_span(sent_span.start, sent_span.end) for sent_span in sents_char_spans]
    start_token_ids = [span[0].idx for span in char_spans if span is not None]
    for token in doc:
        token.is_sent_start = True if token.idx in start_token_ids else False
    return doc

nlp = spacy.blank("en")
nlp.add_pipe('pysbd')

text = "This is a sentence. This is another sentence. This is a third sentence. This is the forth."

seg = pysbd.Segmenter(language="en", clean=False)
print('sent_id', 'sentence', sep='\t|\t')
for sent_id, sent in enumerate(seg.segment(text), start=1):
    print(sent_id, sent, sep='\t|\t')