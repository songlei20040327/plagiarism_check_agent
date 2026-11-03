"""Offline lexical retrieval and rank fusion over one reference snapshot.

Scores are retrieval signals, never probabilities of plagiarism. Character
bigrams avoid an extra tokenizer dependency for the initial Chinese baseline.
"""
import math
import re
import unicodedata
from collections import Counter, defaultdict


def normalize_text(text: str) -> str:
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text)).casefold()


def character_ngrams(text: str, size: int) -> set[str]:
    return {text[i:i + size] for i in range(len(text) - size + 1)}


def reciprocal_rank_fusion(rankings: dict[str, list[str]], rank_constant: int = 60) -> dict[str, float]:
    if rank_constant < 1:
        raise ValueError('RRF rank_constant must be positive.')
    scores = defaultdict(float)
    for ranking in rankings.values():
        seen = set()
        rank = 0
        for chunk_id in ranking:
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            rank += 1
            scores[chunk_id] += 1 / (rank_constant + rank)
    return dict(scores)


class LexicalIndex:
    def __init__(self, chunks: dict[str, str]):
        self.texts = {key: normalize_text(text) for key, text in chunks.items()}
        self.lengths = {}
        self.postings = defaultdict(dict)
        self.gram_postings = defaultdict(set)
        self.long_grams = {}
        for chunk_id, text in self.texts.items():
            tokens = [text[i:i + 2] for i in range(len(text) - 1)]
            self.lengths[chunk_id] = max(len(tokens), 1)
            for token, count in Counter(tokens).items():
                self.postings[token][chunk_id] = count
            for gram in character_ngrams(text, 5):
                self.gram_postings[gram].add(chunk_id)
            self.long_grams[chunk_id] = character_ngrams(text, 13)
        self.count = len(chunks)
        self.average_length = sum(self.lengths.values()) / max(self.count, 1)

    def bm25(self, query: str, limit: int) -> list[tuple[str, float]]:
        text = normalize_text(query)
        tokens = {text[i:i + 2] for i in range(len(text) - 1)}
        scores = defaultdict(float)
        k1, b = 1.5, .75
        for token in tokens:
            posting = self.postings.get(token, {})
            idf = math.log(1 + (self.count - len(posting) + .5) / (len(posting) + .5))
            for chunk_id, frequency in posting.items():
                norm = k1 * (1 - b + b * self.lengths[chunk_id] / self.average_length)
                scores[chunk_id] += idf * frequency * (k1 + 1) / (frequency + norm)
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]

    def literal(self, query: str, limit: int, threshold: float) -> list[tuple[str, float]]:
        grams = character_ngrams(normalize_text(query), 5)
        if not grams:
            return []
        hits = Counter()
        for gram in grams:
            hits.update(self.gram_postings.get(gram, ()))
        scores = [(key, count / len(grams)) for key, count in hits.items()
                  if count / len(grams) >= threshold]
        return sorted(scores, key=lambda item: (-item[1], item[0]))[:limit]

    def long_literal_score(self, query: str, chunk_id: str) -> float:
        grams = character_ngrams(normalize_text(query), 13)
        return len(grams & self.long_grams[chunk_id]) / len(grams) if grams else 0.0
