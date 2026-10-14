"""backend/text_sim.py -- tiny pure-Python text similarity (no numpy / sklearn needed)."""
import math
import re
from collections import Counter


def tokenize(text):
    return re.findall(r"[a-z]+|\d+(?:\.\d+)?", (text or "").lower())


def tfidf_scores(model_text, student_texts):
    """Cosine similarity of every student text to the model answer (TF-IDF)."""
    docs = [tokenize(model_text)] + [tokenize(t) for t in student_texts]
    n = len(docs)
    df = Counter()
    for d in docs:
        df.update(set(d))
    idf = {w: math.log((1 + n) / (1 + c)) + 1.0 for w, c in df.items()}

    def vec(d):
        tf = Counter(d)
        v = {w: (1 + math.log(c)) * idf[w] for w, c in tf.items()}
        return v, math.sqrt(sum(x * x for x in v.values()))

    vm, nm = vec(docs[0])
    out = []
    for d in docs[1:]:
        vs, ns = vec(d)
        if not d or nm == 0 or ns == 0:
            out.append(0.0)
            continue
        dot = sum(x * vm.get(w, 0.0) for w, x in vs.items())
        out.append(dot / (nm * ns))
    return out


def cosine_dense(a, b):
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)
