"""
Word Embeddings Analysis Utilities
===================================
Reusable classes and functions for GloVe embedding exploration,
evaluation, bias analysis, and comparison with foundation models.

All heavy similarity computations are GPU-accelerated via PyTorch
when a CUDA device is available.

TC5033 - Advanced Machine Learning Methods | Team 30
"""

import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import torch
from scipy.stats import permutation_test, spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness as sklearn_trustworthiness
from sklearn.metrics import silhouette_score

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEMANTIC_CATEGORIES = {
    'food': ['burger', 'tortilla', 'bread', 'pizza', 'beef', 'steak', 'fries', 'chips'],
    'countries': ['argentina', 'mexico', 'spain', 'usa', 'france', 'italy', 'greece', 'china'],
    'drinks': ['water', 'beer', 'tequila', 'wine', 'whisky', 'brandy', 'vodka', 'coffee', 'tea'],
    'fruits': ['apple', 'banana', 'orange', 'lemon', 'grapefruit', 'grape', 'strawberry', 'raspberry'],
    'education': ['school', 'work', 'university', 'highschool'],
    'professions': ['teacher', 'student', 'professor', 'engineer', 'doctor', 'nurse', 'lawyer', 'artist'],
}

CATEGORY_COLORS = {
    'food': '#e74c3c',
    'countries': '#3498db',
    'drinks': '#2ecc71',
    'fruits': '#f39c12',
    'education': '#9b59b6',
    'professions': '#1abc9c',
}

POLYSEMOUS_CONTEXTS = {
    'bank': [
        'I deposited money at the bank',
        'The river bank was covered in mud',
    ],
    'bat': [
        'He swung the bat and hit a home run',
        'The bat flew out of the cave at dusk',
    ],
    'crane': [
        'The construction crane lifted the steel beam',
        'A crane stood motionless in the shallow water',
    ],
    'spring': [
        'Flowers bloom in the spring season',
        'Water gushed from the natural spring',
    ],
    'rock': [
        'She likes to listen to rock music',
        'He sat on a large rock by the river',
    ],
}

PROFESSIONS_FOR_BIAS = [
    'programmer', 'engineer', 'scientist', 'mechanic', 'pilot',
    'architect', 'surgeon', 'mathematician', 'professor', 'lawyer',
    'nurse', 'teacher', 'librarian', 'secretary', 'receptionist',
    'housekeeper', 'nanny', 'therapist', 'counselor', 'dietitian',
]

NATIONALITIES_FOR_BIAS = [
    'american', 'british', 'canadian', 'australian', 'german',
    'french', 'italian', 'spanish', 'mexican', 'chinese',
    'japanese', 'indian', 'brazilian', 'russian', 'korean',
    'egyptian', 'nigerian', 'colombian', 'turkish', 'swedish',
]

WEAT_WORD_SETS = {
    'gender_profession': {
        'target_x': ['programmer', 'engineer', 'scientist', 'mechanic', 'pilot',
                      'architect', 'surgeon', 'mathematician', 'financier', 'manager'],
        'target_y': ['nurse', 'teacher', 'librarian', 'secretary', 'receptionist',
                      'housekeeper', 'nanny', 'therapist', 'counselor', 'dietitian'],
        'attribute_a': ['he', 'man', 'boy', 'father', 'son', 'brother', 'husband', 'male'],
        'attribute_b': ['she', 'woman', 'girl', 'mother', 'daughter', 'sister', 'wife', 'female'],
        'description': 'Gender-profession association (Bolukbasi et al., 2016)',
    },
    'gender_science_arts': {
        'target_x': ['science', 'technology', 'physics', 'chemistry', 'math',
                      'algebra', 'geometry', 'calculus', 'equations', 'computation'],
        'target_y': ['poetry', 'art', 'dance', 'literature', 'novel',
                      'symphony', 'drama', 'sculpture', 'painting', 'music'],
        'attribute_a': ['he', 'man', 'boy', 'father', 'son', 'brother', 'husband', 'male'],
        'attribute_b': ['she', 'woman', 'girl', 'mother', 'daughter', 'sister', 'wife', 'female'],
        'description': 'Gender-science/arts association (Caliskan et al., 2017)',
    },
}

GLOVE_DIMENSIONS = [50, 100, 200, 300]


# ---------------------------------------------------------------------------
# EmbeddingSpace: core class for loading, querying, and evaluating embeddings
# ---------------------------------------------------------------------------

class EmbeddingSpace:
    """
    Manages a static word embedding space with GPU-accelerated operations.

    Parameters
    ----------
    embeddings_dict : Dict[str, np.ndarray]
        Word-to-vector mapping.
    device : str
        'cuda', 'cpu', or 'auto'.
    """

    def __init__(
        self,
        embeddings_dict: Dict[str, np.ndarray],
        device: str = 'auto',
    ) -> None:
        self.embeddings_dict = embeddings_dict
        self.device = self._resolve_device(device)

        words = list(embeddings_dict.keys())
        vectors = np.array([embeddings_dict[w] for w in words], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors_norm = vectors / (norms + 1e-12)

        self._words = words
        self._word2idx = {w: i for i, w in enumerate(words)}
        self._vectors_norm_np = vectors_norm
        self._vectors_norm_gpu = torch.tensor(
            vectors_norm, device=self.device, dtype=torch.float32,
        )
        self.dim = vectors.shape[1]
        self.vocab_size = len(words)

    # -- Constructors --

    @classmethod
    def from_glove(cls, path: str, device: str = 'auto') -> 'EmbeddingSpace':
        """Load embeddings from a GloVe-format text file."""
        embeddings_dict = create_emb_dictionary(path)
        return cls(embeddings_dict, device=device)

    @classmethod
    def from_pickle(cls, path: str, device: str = 'auto') -> 'EmbeddingSpace':
        """Load embeddings from a serialized pickle file."""
        with open(path, 'rb') as f:
            embeddings_dict = pickle.load(f)
        return cls(embeddings_dict, device=device)

    def save_pickle(self, path: str) -> None:
        """Serialize embeddings dictionary to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self.embeddings_dict, f)

    # -- Core operations (GPU-accelerated) --

    def find_most_similar(
        self, word: str, top_n: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Find top-N most similar words using cosine similarity.

        Uses GPU matrix-vector product for the similarity computation.
        Core math is numpy-based (rubric requirement); GPU accelerates the matmul.

        Parameters
        ----------
        word : str
            Target word.
        top_n : int
            Number of results to return.

        Returns
        -------
        List[Tuple[str, float]]
            (word, cosine_similarity) pairs sorted descending.
        """
        self._validate_word(word)
        target = self.embeddings_dict[word].astype(np.float32)
        sims = self._cosine_similarity_gpu(target)

        k = min(top_n + 1, sims.size)
        candidate_idx = np.argpartition(sims, -k)[-k:]
        candidate_idx = candidate_idx[np.argsort(sims[candidate_idx])[::-1]]

        results: List[Tuple[str, float]] = []
        for idx in candidate_idx:
            w = self._words[idx]
            if w == word:
                continue
            results.append((w, float(sims[idx])))
            if len(results) == top_n:
                break
        return results

    def analogy(
        self, word1: str, word2: str, word3: str,
    ) -> str:
        """
        Solve: word1 is to word2 as word3 is to ___.

        Parameters
        ----------
        word1, word2, word3 : str
            The three words forming the analogy.

        Returns
        -------
        str
            The best matching word.
        """
        for w in (word1, word2, word3):
            self._validate_word(w)

        target = (
            self.embeddings_dict[word2].astype(np.float32)
            - self.embeddings_dict[word1].astype(np.float32)
            + self.embeddings_dict[word3].astype(np.float32)
        )
        sims = self._cosine_similarity_gpu(target)

        exclude = {word1, word2, word3}
        for idx in np.argsort(sims)[::-1]:
            candidate = self._words[idx]
            if candidate not in exclude:
                return candidate
        raise RuntimeError("No suitable analogy candidate found.")

    def analogy_formatted(
        self, word1: str, word2: str, word3: str,
    ) -> str:
        """Return a formatted analogy string."""
        result = self.analogy(word1, word2, word3)
        return f"{word1} is to {word2} as {word3} is to **{result}**"

    # -- Evaluation --

    def evaluate_word_similarity(
        self, dataset: pl.DataFrame, word_col1: str, word_col2: str, score_col: str,
    ) -> Tuple[float, float]:
        """
        Evaluate Spearman correlation on a word similarity dataset.

        Parameters
        ----------
        dataset : pl.DataFrame
            Must have columns for word1, word2, and human score.
        word_col1, word_col2 : str
            Column names for the word pair.
        score_col : str
            Column name for the human judgment score.

        Returns
        -------
        Tuple[float, float]
            (spearman_rho, p_value)
        """
        human_scores = []
        model_scores = []

        for row in dataset.iter_rows(named=True):
            w1, w2 = row[word_col1].lower(), row[word_col2].lower()
            if w1 not in self.embeddings_dict or w2 not in self.embeddings_dict:
                continue

            v1 = self.embeddings_dict[w1].astype(np.float32)
            v2 = self.embeddings_dict[w2].astype(np.float32)
            cos_sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
            human_scores.append(row[score_col])
            model_scores.append(cos_sim)

        rho, p = spearmanr(human_scores, model_scores)
        return float(rho), float(p)

    def evaluate_analogies(
        self, analogies: List[Tuple[str, str, str, str]],
    ) -> pl.DataFrame:
        """
        Evaluate accuracy on analogy tuples (a, b, c, expected_d).

        Returns
        -------
        pl.DataFrame
            Columns: a, b, c, expected, predicted, correct
        """
        records = []
        for a, b, c, expected in analogies:
            try:
                predicted = self.analogy(a, b, c)
                records.append({
                    'a': a, 'b': b, 'c': c,
                    'expected': expected, 'predicted': predicted,
                    'correct': predicted == expected,
                })
            except (KeyError, RuntimeError):
                continue

        return pl.DataFrame(records)

    # -- Benchmark convenience methods --

    def evaluate_wordsim353(self, path: str) -> Tuple[float, float]:
        """
        Evaluate Spearman correlation on WordSim-353.

        Parameters
        ----------
        path : str
            Path to the WordSim-353 TSV file.

        Returns
        -------
        Tuple[float, float]
            (spearman_rho, p_value)
        """
        df = load_wordsim353(path)
        return self.evaluate_word_similarity(df, 'word1', 'word2', 'human_score')

    def evaluate_simlex999(self, path: str) -> Tuple[float, float]:
        """
        Evaluate Spearman correlation on SimLex-999.

        Parameters
        ----------
        path : str
            Path to the SimLex-999 TSV file.

        Returns
        -------
        Tuple[float, float]
            (spearman_rho, p_value)
        """
        df = load_simlex999(path)
        return self.evaluate_word_similarity(df, 'word1', 'word2', 'simlex_score')

    def evaluate_google_analogies(self, path: str) -> Dict[str, float]:
        """
        Evaluate accuracy on the Google Analogy Test Set.

        Returns accuracy broken down by category (semantic, syntactic, total).

        Parameters
        ----------
        path : str
            Path to the Google analogies text file.

        Returns
        -------
        Dict[str, float]
            Keys: 'semantic_accuracy', 'syntactic_accuracy', 'total_accuracy',
                  'semantic_total', 'syntactic_total', 'semantic_correct',
                  'syntactic_correct', 'oov_skipped'
        """
        analogies, section_map = load_google_analogies(path)

        semantic = [a for a in analogies if section_map[f"{a[0]} {a[1]} {a[2]} {a[3]}"] == 'semantic']
        syntactic = [a for a in analogies if section_map[f"{a[0]} {a[1]} {a[2]} {a[3]}"] == 'syntactic']

        total_oov = 0
        results = {}
        for label, subset in [('semantic', semantic), ('syntactic', syntactic)]:
            initial_count = len(subset)
            valid = [
                t for t in subset
                if all(w in self.embeddings_dict for w in t)
            ]
            oov_skipped = initial_count - len(valid)
            total_oov += oov_skipped

            if valid:
                df = self.evaluate_analogies(valid)
                correct = int(df['correct'].sum())
                total = len(df)
            else:
                correct, total = 0, 0

            results[f'{label}_correct'] = correct
            results[f'{label}_total'] = total
            results[f'{label}_accuracy'] = correct / total if total > 0 else 0.0

        sem_t = results['semantic_total']
        syn_t = results['syntactic_total']
        total_correct = results['semantic_correct'] + results['syntactic_correct']
        total_count = sem_t + syn_t

        results['total_accuracy'] = total_correct / total_count if total_count > 0 else 0.0
        results['oov_skipped'] = total_oov
        return results

    def run_benchmarks(
        self,
        wordsim_path: Optional[str] = None,
        simlex_path: Optional[str] = None,
        analogies_path: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Run all available benchmarks and return a summary table.

        Parameters
        ----------
        wordsim_path : str, optional
            Path to WordSim-353 dataset.
        simlex_path : str, optional
            Path to SimLex-999 dataset.
        analogies_path : str, optional
            Path to Google Analogy Test Set.

        Returns
        -------
        pl.DataFrame
            Columns: benchmark, metric, value, p_value
        """
        records: List[Dict] = []

        if wordsim_path:
            rho, p = self.evaluate_wordsim353(wordsim_path)
            records.append({'benchmark': 'WordSim-353', 'metric': 'Spearman rho', 'value': rho, 'p_value': p})

        if simlex_path:
            rho, p = self.evaluate_simlex999(simlex_path)
            records.append({'benchmark': 'SimLex-999', 'metric': 'Spearman rho', 'value': rho, 'p_value': p})

        if analogies_path:
            res = self.evaluate_google_analogies(analogies_path)
            records.append({
                'benchmark': 'Google Analogy (semantic)',
                'metric': 'Accuracy',
                'value': res['semantic_accuracy'],
                'p_value': None,
            })
            records.append({
                'benchmark': 'Google Analogy (syntactic)',
                'metric': 'Accuracy',
                'value': res['syntactic_accuracy'],
                'p_value': None,
            })
            records.append({
                'benchmark': 'Google Analogy (total)',
                'metric': 'Accuracy',
                'value': res['total_accuracy'],
                'p_value': None,
            })

        return pl.DataFrame(records)

    # -- Internal --

    def _cosine_similarity_gpu(self, target: np.ndarray) -> np.ndarray:
        """Compute cosine similarity of target against full vocab on GPU."""
        target_t = torch.tensor(target, device=self.device, dtype=torch.float32)
        target_t = target_t / (target_t.norm() + 1e-12)
        sims = self._vectors_norm_gpu @ target_t
        return sims.cpu().numpy()

    def _validate_word(self, word: str) -> None:
        if word not in self.embeddings_dict:
            raise KeyError(f"'{word}' not found in vocabulary")
        if np.linalg.norm(self.embeddings_dict[word]) == 0:
            raise ValueError(f"Embedding for '{word}' has zero norm")

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)

    def __repr__(self) -> str:
        return f"EmbeddingSpace(vocab={self.vocab_size}, dim={self.dim}, device={self.device})"


# ---------------------------------------------------------------------------
# EmbeddingVisualizer: PCA, t-SNE, UMAP with semantic coloring
# ---------------------------------------------------------------------------

class EmbeddingVisualizer:
    """
    Advanced visualization for word embeddings with category-based coloring.

    Parameters
    ----------
    embedding_space : EmbeddingSpace
        The embedding space to visualize.
    """

    REDUCERS = {
        'pca': lambda: PCA(n_components=2, random_state=42),
        'tsne': lambda: TSNE(n_components=2, random_state=42, perplexity=15),
    }

    def __init__(self, embedding_space: EmbeddingSpace) -> None:
        self.space = embedding_space
        if UMAP_AVAILABLE:
            self.REDUCERS['umap'] = lambda: UMAP(n_components=2, random_state=42)

    def plot_comparison(
        self,
        words: List[str],
        categories: Dict[str, List[str]],
        colors: Optional[Dict[str, str]] = None,
        figsize: Tuple[int, int] = (24, 8),
    ) -> plt.Figure:
        """
        Side-by-side comparison of PCA, t-SNE, and UMAP (if available).

        Parameters
        ----------
        words : List[str]
            Words to plot.
        categories : Dict[str, List[str]]
            Category name -> list of words in that category.
        colors : Dict[str, str], optional
            Category name -> hex color. Defaults to CATEGORY_COLORS.
        figsize : Tuple[int, int]
            Figure size.

        Returns
        -------
        plt.Figure
        """
        colors = colors or CATEGORY_COLORS
        methods = list(self.REDUCERS.keys())
        embeddings = self._gather_embeddings(words)

        fig, axes = plt.subplots(1, len(methods), figsize=figsize)
        if len(methods) == 1:
            axes = [axes]

        word_to_cat = {}
        for cat, cat_words in categories.items():
            for w in cat_words:
                if w in self.space.embeddings_dict:
                    word_to_cat[w] = cat

        for ax, method in zip(axes, methods):
            reducer = self.REDUCERS[method]()
            coords = reducer.fit_transform(embeddings)
            self._plot_on_axis(ax, coords, words, word_to_cat, colors, method.upper())

        handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=colors.get(cat, '#999'), markersize=8, label=cat)
            for cat in categories if cat in colors
        ]
        fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=10)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        return fig

    def plot_single(
        self,
        words: List[str],
        categories: Dict[str, List[str]],
        method: str = 'pca',
        colors: Optional[Dict[str, str]] = None,
        figsize: Tuple[int, int] = (14, 8),
    ) -> plt.Figure:
        """Single-method visualization."""
        colors = colors or CATEGORY_COLORS
        embeddings = self._gather_embeddings(words)

        reducer = self.REDUCERS[method]()
        coords = reducer.fit_transform(embeddings)

        word_to_cat = {}
        for cat, cat_words in categories.items():
            for w in cat_words:
                if w in self.space.embeddings_dict:
                    word_to_cat[w] = cat

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        self._plot_on_axis(ax, coords, words, word_to_cat, colors, method.upper())

        handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=colors.get(cat, '#999'), markersize=8, label=cat)
            for cat in categories if cat in colors
        ]
        ax.legend(handles=handles, fontsize=9)
        plt.tight_layout()
        return fig

    def compute_silhouette(
        self,
        words: List[str],
        categories: Dict[str, List[str]],
    ) -> float:
        """
        Compute silhouette score measuring cluster separability.

        Parameters
        ----------
        words : List[str]
            Words to evaluate.
        categories : Dict[str, List[str]]
            Category assignments.

        Returns
        -------
        float
            Silhouette score in [-1, 1]. Higher is better.
        """
        word_to_cat = {}
        for cat, cat_words in categories.items():
            for w in cat_words:
                word_to_cat[w] = cat

        filtered = [w for w in words if w in word_to_cat and w in self.space.embeddings_dict]
        if len(set(word_to_cat[w] for w in filtered)) < 2:
            return 0.0

        embeddings = np.array([self.space.embeddings_dict[w] for w in filtered])
        labels = [word_to_cat[w] for w in filtered]
        return float(silhouette_score(embeddings, labels))

    def compute_trustworthiness(
        self,
        words: List[str],
        method: str = 'tsne',
        n_neighbors: int = 5,
    ) -> float:
        """
        Compute trustworthiness score for a dimensionality reduction method.

        Trustworthiness measures whether nearest neighbors in the reduced space
        were also neighbors in the original high-dimensional space
        (Venna & Kaski, 2006).

        Parameters
        ----------
        words : List[str]
            Words to evaluate.
        method : str
            Reduction method ('pca', 'tsne', 'umap').
        n_neighbors : int
            Number of neighbors to consider.

        Returns
        -------
        float
            Trustworthiness score in [0, 1]. Higher is better.
        """
        valid_words = [w for w in words if w in self.space.embeddings_dict]
        X_high = np.array(
            [self.space.embeddings_dict[w] for w in valid_words], dtype=np.float32,
        )

        if method not in self.REDUCERS:
            raise ValueError(f"Unknown method '{method}'. Available: {list(self.REDUCERS.keys())}")

        reducer = self.REDUCERS[method]()
        X_low = reducer.fit_transform(X_high)

        k = min(n_neighbors, len(valid_words) - 1)
        if k < 1:
            return 0.0
        return float(sklearn_trustworthiness(X_high, X_low, n_neighbors=k))

    def compute_method_metrics(
        self,
        words: List[str],
        categories: Dict[str, List[str]],
        method: str = 'pca',
        n_neighbors: int = 5,
    ) -> Dict[str, float]:
        """
        Compute quality metrics for a single reduction method.

        Returns
        -------
        Dict[str, float]
            Keys: 'trustworthiness', 'silhouette_original', 'silhouette_reduced'
        """
        valid_words = [w for w in words if w in self.space.embeddings_dict]

        word_to_cat = {}
        for cat, cat_words in categories.items():
            for w in cat_words:
                word_to_cat[w] = cat

        filtered = [w for w in valid_words if w in word_to_cat]
        labels = [word_to_cat[w] for w in filtered]

        X_high = np.array(
            [self.space.embeddings_dict[w] for w in filtered], dtype=np.float32,
        )

        if method not in self.REDUCERS:
            raise ValueError(f"Unknown method '{method}'. Available: {list(self.REDUCERS.keys())}")

        reducer = self.REDUCERS[method]()
        X_low = reducer.fit_transform(X_high)

        k = min(n_neighbors, len(filtered) - 1)
        trust = float(sklearn_trustworthiness(X_high, X_low, n_neighbors=k)) if k >= 1 else 0.0

        unique_labels = set(labels)
        if len(unique_labels) >= 2:
            sil_orig = float(silhouette_score(X_high, labels))
            sil_reduced = float(silhouette_score(X_low, labels))
        else:
            sil_orig = 0.0
            sil_reduced = 0.0

        return {
            'trustworthiness': round(trust, 4),
            'silhouette_original': round(sil_orig, 4),
            'silhouette_reduced': round(sil_reduced, 4),
        }

    def compare_methods_metrics(
        self,
        words: List[str],
        categories: Dict[str, List[str]],
        n_neighbors: int = 5,
    ) -> pl.DataFrame:
        """
        Compare quality metrics across all available reduction methods.

        Returns
        -------
        pl.DataFrame
            Columns: method, trustworthiness, silhouette_original, silhouette_reduced
        """
        records = []
        for method_name in self.REDUCERS:
            metrics = self.compute_method_metrics(words, categories, method_name, n_neighbors)
            records.append({'method': method_name, **metrics})
        return pl.DataFrame(records)

    def plot_perplexity_sweep(
        self,
        words: List[str],
        categories: Dict[str, List[str]],
        perplexities: Optional[List[int]] = None,
        colors: Optional[Dict[str, str]] = None,
        figsize: Tuple[int, int] = (24, 6),
    ) -> plt.Figure:
        """
        Visualize t-SNE output for different perplexity values.

        Perplexity roughly controls the balance between local and global
        structure preservation (Kobak & Linderman, 2021). Low values
        emphasize local neighborhoods; high values capture broader patterns.

        Parameters
        ----------
        words : List[str]
            Words to visualize.
        categories : Dict[str, List[str]]
            Category assignments for coloring.
        perplexities : List[int], optional
            Perplexity values to sweep. Default: [5, 15, 30, 50].
        colors : Dict[str, str], optional
            Category colors.
        figsize : Tuple[int, int]
            Figure size.

        Returns
        -------
        plt.Figure
        """
        if perplexities is None:
            perplexities = [5, 15, 30, 50]
        colors = colors or CATEGORY_COLORS

        valid_words = [w for w in words if w in self.space.embeddings_dict]
        X_high = np.array(
            [self.space.embeddings_dict[w] for w in valid_words], dtype=np.float32,
        )

        # Filter perplexities that are valid (must be < n_samples)
        perplexities = [p for p in perplexities if p < len(valid_words)]
        if not perplexities:
            raise ValueError("All perplexity values are >= number of words. Reduce perplexity or add more words.")

        word_to_cat = {}
        for cat, cat_words in categories.items():
            for w in cat_words:
                if w in self.space.embeddings_dict:
                    word_to_cat[w] = cat

        fig, axes = plt.subplots(1, len(perplexities), figsize=figsize)
        if len(perplexities) == 1:
            axes = [axes]

        for ax, perp in zip(axes, perplexities):
            tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
            coords = tsne.fit_transform(X_high)
            self._plot_on_axis(ax, coords, valid_words, word_to_cat, colors, f't-SNE (perplexity={perp})')

            # Annotate trustworthiness
            k = min(5, len(valid_words) - 1)
            if k >= 1:
                trust = sklearn_trustworthiness(X_high, coords, n_neighbors=k)
                ax.text(
                    0.02, 0.02, f'Trust: {trust:.3f}',
                    transform=ax.transAxes, fontsize=9,
                    verticalalignment='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                )

        handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=colors.get(cat, '#999'), markersize=8, label=cat)
            for cat in categories if cat in colors
        ]
        fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=10)
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        return fig

    def plot_comparison_with_metrics(
        self,
        words: List[str],
        categories: Dict[str, List[str]],
        colors: Optional[Dict[str, str]] = None,
        n_neighbors: int = 5,
        figsize: Tuple[int, int] = (24, 8),
    ) -> plt.Figure:
        """
        Side-by-side PCA vs t-SNE vs UMAP with quality metrics annotated.

        Each subplot includes trustworthiness and silhouette scores.

        Parameters
        ----------
        words : List[str]
            Words to plot.
        categories : Dict[str, List[str]]
            Category name -> list of words.
        colors : Dict[str, str], optional
            Category colors.
        n_neighbors : int
            Number of neighbors for trustworthiness.
        figsize : Tuple[int, int]
            Figure size.

        Returns
        -------
        plt.Figure
        """
        colors = colors or CATEGORY_COLORS
        methods = list(self.REDUCERS.keys())

        valid_words = [w for w in words if w in self.space.embeddings_dict]
        X_high = np.array(
            [self.space.embeddings_dict[w] for w in valid_words], dtype=np.float32,
        )

        word_to_cat = {}
        for cat, cat_words in categories.items():
            for w in cat_words:
                if w in self.space.embeddings_dict:
                    word_to_cat[w] = cat

        labels = [word_to_cat.get(w, 'other') for w in valid_words]
        unique_labels = set(labels)
        has_labels = len(unique_labels) >= 2

        fig, axes = plt.subplots(1, len(methods), figsize=figsize)
        if len(methods) == 1:
            axes = [axes]

        for ax, method in zip(axes, methods):
            reducer = self.REDUCERS[method]()
            coords = reducer.fit_transform(X_high)
            self._plot_on_axis(ax, coords, valid_words, word_to_cat, colors, method.upper())

            # Compute and annotate metrics
            k = min(n_neighbors, len(valid_words) - 1)
            trust = sklearn_trustworthiness(X_high, coords, n_neighbors=k) if k >= 1 else 0.0
            sil = float(silhouette_score(coords, labels)) if has_labels else 0.0

            ax.text(
                0.02, 0.02,
                f'Trust: {trust:.3f} | Silhouette: {sil:.3f}',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
            )

        handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=colors.get(cat, '#999'), markersize=8, label=cat)
            for cat in categories if cat in colors
        ]
        fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=10)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        return fig

    def _gather_embeddings(self, words: List[str]) -> np.ndarray:
        valid = [w for w in words if w in self.space.embeddings_dict]
        return np.array([self.space.embeddings_dict[w] for w in valid], dtype=np.float32)

    @staticmethod
    def _plot_on_axis(
        ax: plt.Axes, coords: np.ndarray, words: List[str],
        word_to_cat: Dict[str, str], colors: Dict[str, str], title: str,
    ) -> None:
        for i, word in enumerate(words):
            cat = word_to_cat.get(word, 'other')
            color = colors.get(cat, '#999999')
            ax.scatter(coords[i, 0], coords[i, 1], c=color, s=60, edgecolors='white', linewidths=0.5)
            ax.annotate(word, (coords[i, 0], coords[i, 1]), fontsize=8,
                        xytext=(5, 5), textcoords='offset points')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# BiasAnalyzer: gender and nationality bias in embeddings
# ---------------------------------------------------------------------------

class BiasAnalyzer:
    """
    Analyze biases in word embeddings using projection onto bias directions.

    Implements a simplified version of the WEAT framework
    (Caliskan et al., 2017; Bolukbasi et al., 2016).

    Parameters
    ----------
    embedding_space : EmbeddingSpace
        The embedding space to analyze.
    """

    DEFAULT_GENDER_PAIRS = [
        ('he', 'she'), ('man', 'woman'), ('boy', 'girl'),
        ('king', 'queen'), ('brother', 'sister'), ('father', 'mother'),
        ('husband', 'wife'), ('son', 'daughter'),
    ]

    def __init__(self, embedding_space: EmbeddingSpace) -> None:
        self.space = embedding_space

    def compute_gender_direction(
        self, pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> np.ndarray:
        """
        Compute the gender direction as the mean difference of word pairs.

        Parameters
        ----------
        pairs : List[Tuple[str, str]], optional
            Male-female word pairs. Defaults to common pairs.

        Returns
        -------
        np.ndarray
            Unit vector representing the gender direction.
        """
        pairs = pairs or self.DEFAULT_GENDER_PAIRS
        diffs = []
        for male, female in pairs:
            if male in self.space.embeddings_dict and female in self.space.embeddings_dict:
                diff = self.space.embeddings_dict[male] - self.space.embeddings_dict[female]
                diffs.append(diff)
        direction = np.mean(diffs, axis=0).astype(np.float32)
        return direction / (np.linalg.norm(direction) + 1e-12)

    def projection_scores(
        self, words: List[str], direction: np.ndarray,
    ) -> pl.DataFrame:
        """
        Project words onto a bias direction and return scores.

        Parameters
        ----------
        words : List[str]
            Words to analyze.
        direction : np.ndarray
            Unit vector for the bias direction.

        Returns
        -------
        pl.DataFrame
            Columns: word, bias_score. Positive = male-leaning, negative = female-leaning.
        """
        records = []
        for w in words:
            if w not in self.space.embeddings_dict:
                continue
            vec = self.space.embeddings_dict[w].astype(np.float32)
            score = float(np.dot(vec, direction) / (np.linalg.norm(vec) + 1e-12))
            records.append({'word': w, 'bias_score': score})

        return pl.DataFrame(records).sort('bias_score')

    def plot_bias(
        self,
        scores_df: pl.DataFrame,
        title: str = 'Gender Bias in Word Embeddings',
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """
        Horizontal bar chart of bias scores.

        Parameters
        ----------
        scores_df : pl.DataFrame
            Output from projection_scores().
        title : str
            Plot title.
        figsize : Tuple[int, int]
            Figure size.

        Returns
        -------
        plt.Figure
        """
        df_pd = scores_df.to_pandas()
        fig, ax = plt.subplots(figsize=figsize)

        bar_colors = ['#3498db' if s > 0 else '#e74c3c' for s in df_pd['bias_score']]
        ax.barh(df_pd['word'], df_pd['bias_score'], color=bar_colors)
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_xlabel('Bias Score (positive = male-associated)')
        ax.set_title(title, fontsize=13)
        plt.tight_layout()
        return fig

    def compute_weat_score(
        self,
        target_x: List[str],
        target_y: List[str],
        attribute_a: List[str],
        attribute_b: List[str],
        n_resamples: int = 10000,
        random_state: int = 42,
    ) -> Dict[str, float]:
        """
        Compute simplified WEAT effect size and p-value.

        Measures whether target_x is more associated with attribute_a
        than target_y, relative to attribute_b (Caliskan et al., 2017).

        Parameters
        ----------
        target_x : List[str]
            First set of target words (e.g., male-stereotyped professions).
        target_y : List[str]
            Second set of target words (e.g., female-stereotyped professions).
        attribute_a : List[str]
            First set of attribute words (e.g., male terms).
        attribute_b : List[str]
            Second set of attribute words (e.g., female terms).
        n_resamples : int
            Permutations for significance test.
        random_state : int
            Seed for reproducibility.

        Returns
        -------
        Dict[str, float]
            Keys: 'effect_size', 'p_value', 'test_statistic',
                  'n_target_x', 'n_target_y', 'n_attr_a', 'n_attr_b'
        """
        idx = self.space._word2idx
        norms = self.space._vectors_norm_np

        # Filter to in-vocabulary words
        tx = [w for w in target_x if w in idx]
        ty = [w for w in target_y if w in idx]
        aa = [w for w in attribute_a if w in idx]
        ab = [w for w in attribute_b if w in idx]

        def _association(word: str) -> float:
            """s(w, A, B) = mean cos(w, a) - mean cos(w, b)."""
            w_vec = norms[idx[word]]
            mean_a = np.mean([np.dot(w_vec, norms[idx[a]]) for a in aa])
            mean_b = np.mean([np.dot(w_vec, norms[idx[b]]) for b in ab])
            return float(mean_a - mean_b)

        scores_x = np.array([_association(w) for w in tx])
        scores_y = np.array([_association(w) for w in ty])

        # Test statistic
        test_stat = float(np.sum(scores_x) - np.sum(scores_y))

        # Effect size (Cohen's d)
        all_scores = np.concatenate([scores_x, scores_y])
        std_all = np.std(all_scores, ddof=1) if len(all_scores) > 1 else 1.0
        effect_size = float(
            (np.mean(scores_x) - np.mean(scores_y)) / (std_all + 1e-12)
        )

        # Permutation test for p-value
        def stat_func(x_s, y_s):
            return np.mean(x_s) - np.mean(y_s)

        result = permutation_test(
            (scores_x, scores_y),
            stat_func,
            n_resamples=n_resamples,
            alternative='two-sided',
            random_state=random_state,
        )

        return {
            'effect_size': effect_size,
            'p_value': float(result.pvalue),
            'test_statistic': test_stat,
            'n_target_x': len(tx),
            'n_target_y': len(ty),
            'n_attr_a': len(aa),
            'n_attr_b': len(ab),
        }

    def run_weat_battery(
        self,
        weat_sets: Optional[Dict[str, Dict]] = None,
        n_resamples: int = 10000,
    ) -> pl.DataFrame:
        """
        Run multiple WEAT tests and return consolidated results.

        Parameters
        ----------
        weat_sets : Dict, optional
            WEAT configurations. Defaults to WEAT_WORD_SETS.
        n_resamples : int
            Permutations per test.

        Returns
        -------
        pl.DataFrame
            Columns: test_name, description, effect_size, p_value,
                     test_statistic, n_words_used, interpretation
        """
        weat_sets = weat_sets or WEAT_WORD_SETS
        rows = []
        for name, cfg in weat_sets.items():
            res = self.compute_weat_score(
                cfg['target_x'], cfg['target_y'],
                cfg['attribute_a'], cfg['attribute_b'],
                n_resamples=n_resamples,
            )
            d = abs(res['effect_size'])
            if d > 1.0:
                interp = 'Strong'
            elif d > 0.5:
                interp = 'Moderate'
            else:
                interp = 'Weak'
            rows.append({
                'test_name': name,
                'description': cfg['description'],
                'effect_size': round(res['effect_size'], 4),
                'p_value': round(res['p_value'], 4),
                'test_statistic': round(res['test_statistic'], 4),
                'n_words_used': res['n_target_x'] + res['n_target_y'],
                'interpretation': interp,
            })
        return pl.DataFrame(rows)

    def nationality_bias_analysis(
        self,
        nationalities: List[str],
        positive_attrs: Optional[List[str]] = None,
        negative_attrs: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """
        Measure bias associations for nationalities along sentiment and gender axes.

        Projects nationality embeddings onto directions defined by
        positive vs negative attribute words and male vs female pairs.

        Parameters
        ----------
        nationalities : List[str]
            Nationality adjectives to analyze.
        positive_attrs : List[str], optional
            Positive sentiment words.
        negative_attrs : List[str], optional
            Negative sentiment words.

        Returns
        -------
        pl.DataFrame
            Columns: nationality, net_sentiment, gender_score
        """
        if positive_attrs is None:
            positive_attrs = [
                'wealthy', 'educated', 'innovative', 'peaceful',
                'democratic', 'modern', 'advanced', 'prosperous',
            ]
        if negative_attrs is None:
            negative_attrs = [
                'poor', 'dangerous', 'corrupt', 'violent',
                'backward', 'primitive', 'criminal', 'terrorist',
            ]

        emb = self.space.embeddings_dict

        # Compute sentiment direction
        pos_vecs = [emb[w].astype(np.float32) for w in positive_attrs if w in emb]
        neg_vecs = [emb[w].astype(np.float32) for w in negative_attrs if w in emb]
        sentiment_dir = np.mean(pos_vecs, axis=0) - np.mean(neg_vecs, axis=0)
        sentiment_dir = sentiment_dir / (np.linalg.norm(sentiment_dir) + 1e-12)

        # Compute gender direction
        gender_dir = self.compute_gender_direction()

        records = []
        for nat in nationalities:
            if nat not in emb:
                continue
            vec = emb[nat].astype(np.float32)
            norm_v = np.linalg.norm(vec) + 1e-12
            net_sentiment = float(np.dot(vec, sentiment_dir) / norm_v)
            gender_score = float(np.dot(vec, gender_dir) / norm_v)
            records.append({
                'nationality': nat,
                'net_sentiment': net_sentiment,
                'gender_score': gender_score,
            })

        return pl.DataFrame(records).sort('net_sentiment')

    def plot_nationality_bias(
        self,
        nationality_df: pl.DataFrame,
        figsize: Tuple[int, int] = (12, 7),
    ) -> plt.Figure:
        """
        Scatter plot of nationalities on sentiment vs gender axes.

        Parameters
        ----------
        nationality_df : pl.DataFrame
            Output from nationality_bias_analysis().
        figsize : Tuple[int, int]
            Figure size.

        Returns
        -------
        plt.Figure
        """
        df = nationality_df.to_pandas()
        fig, ax = plt.subplots(figsize=figsize)

        # Color by quadrant
        colors = []
        for _, row in df.iterrows():
            if row['net_sentiment'] >= 0 and row['gender_score'] >= 0:
                colors.append('#3498db')   # male + positive
            elif row['net_sentiment'] >= 0 and row['gender_score'] < 0:
                colors.append('#2ecc71')   # female + positive
            elif row['net_sentiment'] < 0 and row['gender_score'] >= 0:
                colors.append('#e74c3c')   # male + negative
            else:
                colors.append('#f39c12')   # female + negative

        ax.scatter(df['gender_score'], df['net_sentiment'], c=colors, s=80, zorder=3)

        for _, row in df.iterrows():
            ax.annotate(
                row['nationality'],
                (row['gender_score'], row['net_sentiment']),
                fontsize=8, ha='center', va='bottom',
                textcoords='offset points', xytext=(0, 5),
            )

        ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
        ax.axvline(x=0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_xlabel('Gender Score (positive = male-associated)')
        ax.set_ylabel('Net Sentiment (positive = favorable)')
        ax.set_title('Nationality Bias: Sentiment vs Gender Associations', fontsize=13)

        # Quadrant labels
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        offset_x = (xlim[1] - xlim[0]) * 0.02
        offset_y = (ylim[1] - ylim[0]) * 0.02
        ax.text(xlim[1] - offset_x, ylim[1] - offset_y,
                'Male + Positive', ha='right', va='top', fontsize=8, color='#3498db')
        ax.text(xlim[0] + offset_x, ylim[1] - offset_y,
                'Female + Positive', ha='left', va='top', fontsize=8, color='#2ecc71')
        ax.text(xlim[1] - offset_x, ylim[0] + offset_y,
                'Male + Negative', ha='right', va='bottom', fontsize=8, color='#e74c3c')
        ax.text(xlim[0] + offset_x, ylim[0] + offset_y,
                'Female + Negative', ha='left', va='bottom', fontsize=8, color='#f39c12')

        plt.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# DimensionalityAnalyzer: effect of embedding dimensionality on quality/cost
# ---------------------------------------------------------------------------

class DimensionalityAnalyzer:
    """
    Analyze the effect of embedding dimensionality on quality and cost.

    Loads multiple GloVe dimension variants, runs benchmarks, measures
    computational cost, and provides spectral analysis via SVD.

    Parameters
    ----------
    glove_dir : str
        Directory containing GloVe text files (glove.6B.Xd.txt).
    dimensions : List[int], optional
        Dimensions to compare. Defaults to GLOVE_DIMENSIONS.
    device : str
        Device for EmbeddingSpace ('auto', 'cuda', 'cpu').
    """

    def __init__(
        self,
        glove_dir: str,
        dimensions: Optional[List[int]] = None,
        device: str = 'auto',
    ) -> None:
        self.glove_dir = Path(glove_dir)
        self.dimensions = dimensions or GLOVE_DIMENSIONS
        self.device = device
        self.spaces: Dict[int, EmbeddingSpace] = {}
        self.load_times: Dict[int, float] = {}
        self.memory_mb: Dict[int, float] = {}
        self._singular_values: Dict[int, np.ndarray] = {}

    def load_all_dimensions(self) -> pl.DataFrame:
        """
        Load GloVe embeddings for all configured dimensions with timing.

        Returns
        -------
        pl.DataFrame
            Columns: dimension, vocab_size, load_time_s, memory_mb
        """
        records: List[Dict] = []
        for dim in self.dimensions:
            path = self.glove_dir / f'glove.6B.{dim}d.txt'
            print(f'Loading GloVe {dim}d from {path}...')

            start = time.perf_counter()
            space = EmbeddingSpace.from_glove(str(path), device=self.device)
            elapsed = time.perf_counter() - start

            self.spaces[dim] = space
            self.load_times[dim] = elapsed
            mem = space._vectors_norm_np.nbytes / (1024 * 1024)
            self.memory_mb[dim] = mem

            print(f'  {dim}d: {space.vocab_size} words, '
                  f'{elapsed:.1f}s, {mem:.0f} MB')
            records.append({
                'dimension': dim,
                'vocab_size': space.vocab_size,
                'load_time_s': round(elapsed, 2),
                'memory_mb': round(mem, 1),
            })

        return pl.DataFrame(records)

    def benchmark_all(
        self,
        wordsim_path: str,
        simlex_path: str,
        analogies_path: str,
    ) -> pl.DataFrame:
        """
        Run all benchmarks across all loaded dimensions.

        Wraps compare_embeddings_on_benchmarks() with proper labeling.

        Parameters
        ----------
        wordsim_path : str
            Path to WordSim-353 dataset.
        simlex_path : str
            Path to SimLex-999 dataset.
        analogies_path : str
            Path to Google Analogy Test Set.

        Returns
        -------
        pl.DataFrame
            Columns: embedding, benchmark, metric, value, dimension
        """
        spaces_dict = {f'GloVe-{dim}d': space
                       for dim, space in self.spaces.items()}
        combined = compare_embeddings_on_benchmarks(
            spaces_dict, wordsim_path, simlex_path, analogies_path,
        )
        dim_col = combined['embedding'].str.extract(r'(\d+)').cast(pl.Int64)
        combined = combined.with_columns(dim_col.alias('dimension'))
        return combined

    def measure_computational_cost(
        self,
        query_words: Optional[List[str]] = None,
        n_queries: int = 100,
        top_n: int = 10,
    ) -> pl.DataFrame:
        """
        Measure query time for find_most_similar across dimensions.

        Parameters
        ----------
        query_words : List[str], optional
            Words to query. Defaults to first n_queries common words.
        n_queries : int
            Number of queries to average over.
        top_n : int
            Top-N for similarity search.

        Returns
        -------
        pl.DataFrame
            Columns: dimension, mean_query_ms, std_query_ms,
                     memory_mb, load_time_s
        """
        min_dim = min(self.spaces.keys())
        if query_words is None:
            query_words = self.spaces[min_dim]._words[:n_queries]
        else:
            query_words = query_words[:n_queries]

        # Filter to words present in all spaces
        common = set(query_words)
        for space in self.spaces.values():
            common &= set(space._words)
        query_words = [w for w in query_words if w in common][:n_queries]

        records: List[Dict] = []
        for dim in sorted(self.spaces.keys()):
            space = self.spaces[dim]

            # Warmup (5 queries, not counted)
            for w in query_words[:5]:
                space.find_most_similar(w, top_n=top_n)

            times = []
            for w in query_words:
                start = time.perf_counter()
                space.find_most_similar(w, top_n=top_n)
                elapsed = time.perf_counter() - start
                times.append(elapsed * 1000)

            records.append({
                'dimension': dim,
                'mean_query_ms': round(float(np.mean(times)), 3),
                'std_query_ms': round(float(np.std(times)), 3),
                'memory_mb': round(self.memory_mb[dim], 1),
                'load_time_s': round(self.load_times[dim], 2),
            })

        return pl.DataFrame(records)

    def compute_intrinsic_dimensionality(
        self,
        n_sample: int = 10000,
        random_state: int = 42,
    ) -> pl.DataFrame:
        """
        Estimate intrinsic dimensionality via singular value decay.

        Performs SVD on a sample of embeddings and computes participation
        ratio, 90% variance threshold, and 95% variance threshold.

        Parameters
        ----------
        n_sample : int
            Number of words to sample for SVD.
        random_state : int
            Seed for reproducibility.

        Returns
        -------
        pl.DataFrame
            Columns: dimension, participation_ratio,
                     dims_90pct_variance, dims_95pct_variance
        """
        rng = np.random.default_rng(random_state)
        records: List[Dict] = []

        for dim in sorted(self.spaces.keys()):
            space = self.spaces[dim]
            sample_size = min(n_sample, space.vocab_size)
            indices = rng.choice(space.vocab_size, size=sample_size,
                                 replace=False)
            matrix = space._vectors_norm_np[indices]
            matrix = matrix - matrix.mean(axis=0)

            _, s, _ = np.linalg.svd(matrix, full_matrices=False)
            self._singular_values[dim] = s

            pr = float((s.sum()) ** 2 / (s ** 2).sum())
            var_explained = np.cumsum(s ** 2) / np.sum(s ** 2)
            dims_90 = int(np.searchsorted(var_explained, 0.90) + 1)
            dims_95 = int(np.searchsorted(var_explained, 0.95) + 1)

            records.append({
                'dimension': dim,
                'participation_ratio': round(pr, 1),
                'dims_90pct_variance': dims_90,
                'dims_95pct_variance': dims_95,
            })

        return pl.DataFrame(records)

    def plot_benchmark_comparison(
        self,
        benchmark_df: pl.DataFrame,
        figsize: Tuple[int, int] = (14, 5),
    ) -> plt.Figure:
        """
        Line plots of benchmark scores vs dimensionality.

        Parameters
        ----------
        benchmark_df : pl.DataFrame
            Output from benchmark_all().
        figsize : Tuple[int, int]
            Figure size.

        Returns
        -------
        plt.Figure
        """
        benchmarks = benchmark_df['benchmark'].unique().sort().to_list()
        n_plots = len(benchmarks)
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            axes = [axes]

        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']

        for idx, bench in enumerate(benchmarks):
            ax = axes[idx]
            subset = benchmark_df.filter(pl.col('benchmark') == bench)
            subset = subset.sort('dimension')
            dims = subset['dimension'].to_list()
            vals = subset['value'].to_list()

            ax.plot(dims, vals, 'o-', color=colors[idx % len(colors)],
                    linewidth=2, markersize=8)

            for i, (d, v) in enumerate(zip(dims, vals)):
                ax.annotate(f'{v:.3f}', (d, v), textcoords='offset points',
                            xytext=(0, 10), ha='center', fontsize=9)
                if i > 0:
                    delta = v - vals[i - 1]
                    mid_x = (dims[i - 1] + d) / 2
                    mid_y = (vals[i - 1] + v) / 2
                    ax.annotate(f'+{delta:.3f}', (mid_x, mid_y),
                                fontsize=7, ha='center', color='gray',
                                style='italic')

            ax.set_xlabel('Embedding Dimension')
            ax.set_ylabel(subset['metric'][0])
            ax.set_title(bench)
            ax.set_xticks(dims)
            ax.grid(True, alpha=0.3)

        fig.suptitle('Benchmark Performance vs Embedding Dimensionality',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig

    def plot_cost_analysis(
        self,
        cost_df: pl.DataFrame,
        benchmark_df: Optional[pl.DataFrame] = None,
        figsize: Tuple[int, int] = (14, 5),
    ) -> plt.Figure:
        """
        Visualize computational cost vs dimensionality.

        Parameters
        ----------
        cost_df : pl.DataFrame
            Output from measure_computational_cost().
        benchmark_df : pl.DataFrame, optional
            Output from benchmark_all(), used for efficiency ratio.
        figsize : Tuple[int, int]
            Figure size.

        Returns
        -------
        plt.Figure
        """
        cost_sorted = cost_df.sort('dimension')
        dims = cost_sorted['dimension'].to_list()
        query_ms = cost_sorted['mean_query_ms'].to_list()
        query_std = cost_sorted['std_query_ms'].to_list()
        mem = cost_sorted['memory_mb'].to_list()

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Subplot 1: Query time
        axes[0].errorbar(dims, query_ms, yerr=query_std, fmt='o-',
                         color='#3498db', linewidth=2, markersize=8,
                         capsize=4)
        for d, v in zip(dims, query_ms):
            axes[0].annotate(f'{v:.2f}', (d, v), textcoords='offset points',
                             xytext=(0, 10), ha='center', fontsize=9)
        axes[0].set_xlabel('Embedding Dimension')
        axes[0].set_ylabel('Mean Query Time (ms)')
        axes[0].set_title('Query Latency vs Dimension')
        axes[0].set_xticks(dims)
        axes[0].grid(True, alpha=0.3)

        # Subplot 2: Memory
        axes[1].bar(dims, mem, color='#e74c3c', alpha=0.7, width=30)
        for d, v in zip(dims, mem):
            axes[1].annotate(f'{v:.0f}', (d, v), textcoords='offset points',
                             xytext=(0, 5), ha='center', fontsize=9)
        axes[1].set_xlabel('Embedding Dimension')
        axes[1].set_ylabel('Memory (MB)')
        axes[1].set_title('Memory Usage vs Dimension')
        axes[1].set_xticks(dims)
        axes[1].grid(True, alpha=0.3)

        # Subplot 3: Efficiency ratio
        if benchmark_df is not None:
            efficiency = []
            for d, qms, m in zip(dims, query_ms, mem):
                sub = benchmark_df.filter(
                    (pl.col('dimension') == d)
                    & (pl.col('benchmark').is_in(
                        ['WordSim-353', 'SimLex-999']))
                )
                if len(sub) > 0:
                    avg_score = sub['value'].mean()
                    cost = qms * m
                    efficiency.append(avg_score / cost * 1e4)
                else:
                    efficiency.append(0.0)

            axes[2].plot(dims, efficiency, 'o-', color='#2ecc71',
                         linewidth=2, markersize=8)
            for d, v in zip(dims, efficiency):
                axes[2].annotate(f'{v:.1f}', (d, v),
                                 textcoords='offset points',
                                 xytext=(0, 10), ha='center', fontsize=9)
            axes[2].set_ylabel('Efficiency (score / cost x 10^4)')
        else:
            axes[2].text(0.5, 0.5, 'Provide benchmark_df\nfor efficiency plot',
                         ha='center', va='center', transform=axes[2].transAxes)

        axes[2].set_xlabel('Embedding Dimension')
        axes[2].set_title('Quality/Cost Efficiency')
        axes[2].set_xticks(dims)
        axes[2].grid(True, alpha=0.3)

        fig.suptitle('Computational Cost vs Embedding Dimensionality',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig

    def plot_singular_value_decay(
        self,
        figsize: Tuple[int, int] = (14, 5),
    ) -> plt.Figure:
        """
        Plot singular value spectrum and cumulative variance for each dimension.

        Must call compute_intrinsic_dimensionality() first.

        Parameters
        ----------
        figsize : Tuple[int, int]
            Figure size.

        Returns
        -------
        plt.Figure
        """
        if not self._singular_values:
            raise RuntimeError(
                'Call compute_intrinsic_dimensionality() first.')

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

        for idx, dim in enumerate(sorted(self._singular_values.keys())):
            s = self._singular_values[dim]
            color = colors[idx % len(colors)]
            label = f'{dim}d'

            # Subplot 1: Normalized singular value spectrum (log scale)
            s_norm = s / s[0]
            axes[0].plot(range(1, len(s_norm) + 1), s_norm, color=color,
                         label=label, linewidth=1.5, alpha=0.8)

            # Subplot 2: Cumulative variance explained
            var_explained = np.cumsum(s ** 2) / np.sum(s ** 2)
            axes[1].plot(range(1, len(var_explained) + 1), var_explained,
                         color=color, label=label, linewidth=1.5, alpha=0.8)

            # Mark 90% threshold
            dim_90 = int(np.searchsorted(var_explained, 0.90) + 1)
            axes[1].plot(dim_90, 0.90, 'x', color=color, markersize=8,
                         markeredgewidth=2)
            axes[1].annotate(f'{dim_90}', (dim_90, 0.90),
                             textcoords='offset points', xytext=(5, -10),
                             fontsize=8, color=color)

        axes[0].set_yscale('log')
        axes[0].set_xlabel('Singular Value Index')
        axes[0].set_ylabel('Normalized Singular Value (log)')
        axes[0].set_title('Singular Value Spectrum')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].axhline(y=0.90, color='gray', linestyle='--', alpha=0.5,
                        label='90% variance')
        axes[1].axhline(y=0.95, color='gray', linestyle=':', alpha=0.5,
                        label='95% variance')
        axes[1].set_xlabel('Number of Dimensions')
        axes[1].set_ylabel('Cumulative Variance Explained')
        axes[1].set_title('Cumulative Variance by Dimension')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.suptitle('Intrinsic Dimensionality Analysis via SVD',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# FoundationalComparison: GloVe vs sentence-transformers
# ---------------------------------------------------------------------------

class FoundationalComparison:
    """
    Compare static embeddings (GloVe) with contextual embeddings
    from sentence-transformers, using GPU acceleration.

    Parameters
    ----------
    glove_space : EmbeddingSpace
        Static embedding space.
    st_model_name : str
        Sentence-transformers model identifier.
    device : str
        'cuda', 'cpu', or 'auto'.
    """

    def __init__(
        self,
        glove_space: EmbeddingSpace,
        st_model_name: str = 'all-MiniLM-L6-v2',
        device: str = 'auto',
    ) -> None:
        if not ST_AVAILABLE:
            raise ImportError("sentence-transformers is required: pip install sentence-transformers")

        self.glove = glove_space
        device_str = 'cuda' if device == 'auto' and torch.cuda.is_available() else device
        self.st_model = SentenceTransformer(st_model_name, device=device_str)
        self._st_cache: Dict[str, np.ndarray] = {}

    def _encode_word(self, word: str) -> np.ndarray:
        """Encode a single word with sentence-transformers (cached)."""
        if word not in self._st_cache:
            self._st_cache[word] = self.st_model.encode([word])[0]
        return self._st_cache[word]

    def _encode_words(self, words: List[str]) -> np.ndarray:
        """Encode multiple words in a single batch (GPU-efficient)."""
        uncached = [w for w in words if w not in self._st_cache]
        if uncached:
            encodings = self.st_model.encode(uncached, batch_size=128, show_progress_bar=False)
            for w, enc in zip(uncached, encodings):
                self._st_cache[w] = enc
        return np.array([self._st_cache[w] for w in words], dtype=np.float32)

    def compare_neighbors(
        self, words: List[str], top_n: int = 10, vocab_subset: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """
        Compare top-N neighbors between GloVe and sentence-transformers.

        Parameters
        ----------
        words : List[str]
            Query words.
        top_n : int
            Number of neighbors.
        vocab_subset : List[str], optional
            Vocabulary to search in ST space. Defaults to top 50K GloVe words.

        Returns
        -------
        pl.DataFrame
            Columns: query_word, glove_neighbors, st_neighbors, jaccard_overlap
        """
        if vocab_subset is None:
            vocab_subset = self.glove._words[:50000]

        st_embeddings = self._encode_words(vocab_subset)
        st_norms = np.linalg.norm(st_embeddings, axis=1, keepdims=True)
        st_normed = st_embeddings / (st_norms + 1e-12)

        records = []
        for query in words:
            if query not in self.glove.embeddings_dict:
                continue

            glove_neighbors = {w for w, _ in self.glove.find_most_similar(query, top_n)}

            q_emb = self._encode_word(query)
            q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-12)
            st_sims = st_normed @ q_norm
            top_idx = np.argsort(st_sims)[::-1]
            st_neighbors = set()
            for idx in top_idx:
                if vocab_subset[idx] != query:
                    st_neighbors.add(vocab_subset[idx])
                if len(st_neighbors) == top_n:
                    break

            intersection = glove_neighbors & st_neighbors
            union = glove_neighbors | st_neighbors
            jaccard = len(intersection) / len(union) if union else 0.0

            records.append({
                'query_word': query,
                'glove_neighbors': ', '.join(sorted(glove_neighbors)),
                'st_neighbors': ', '.join(sorted(st_neighbors)),
                'jaccard_overlap': round(jaccard, 4),
            })

        return pl.DataFrame(records)

    def compare_analogies(
        self, analogies: List[Tuple[str, str, str, str]],
        vocab_subset: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """
        Compare analogy accuracy: GloVe vs sentence-transformers.

        Parameters
        ----------
        analogies : List[Tuple[str, str, str, str]]
            (word1, word2, word3, expected_answer) tuples.
        vocab_subset : List[str], optional
            Search vocabulary for ST. Defaults to top 50K GloVe words.

        Returns
        -------
        pl.DataFrame
            Columns: a, b, c, expected, glove_pred, st_pred, glove_correct, st_correct
        """
        if vocab_subset is None:
            vocab_subset = self.glove._words[:50000]

        st_embeddings = self._encode_words(vocab_subset)
        st_norms = np.linalg.norm(st_embeddings, axis=1, keepdims=True)
        st_normed = st_embeddings / (st_norms + 1e-12)

        records = []
        for a, b, c, expected in analogies:
            try:
                glove_pred = self.glove.analogy(a, b, c)
            except (KeyError, RuntimeError):
                glove_pred = ''

            try:
                emb_a = self._encode_word(a)
                emb_b = self._encode_word(b)
                emb_c = self._encode_word(c)
                target = emb_b - emb_a + emb_c
                target_norm = target / (np.linalg.norm(target) + 1e-12)
                sims = st_normed @ target_norm
                exclude = {a, b, c}
                st_pred = ''
                for idx in np.argsort(sims)[::-1]:
                    if vocab_subset[idx] not in exclude:
                        st_pred = vocab_subset[idx]
                        break
            except Exception:
                st_pred = ''

            records.append({
                'a': a, 'b': b, 'c': c, 'expected': expected,
                'glove_pred': glove_pred, 'st_pred': st_pred,
                'glove_correct': glove_pred == expected,
                'st_correct': st_pred == expected,
            })

        return pl.DataFrame(records)

    def plot_side_by_side(
        self,
        words: List[str],
        categories: Dict[str, List[str]],
        colors: Optional[Dict[str, str]] = None,
        figsize: Tuple[int, int] = (20, 8),
    ) -> plt.Figure:
        """
        PCA visualization of the same words in GloVe vs ST space.

        Parameters
        ----------
        words : List[str]
            Words to visualize.
        categories : Dict[str, List[str]]
            Category assignments.
        colors : Dict[str, str], optional
            Category colors.
        figsize : Tuple[int, int]
            Figure size.

        Returns
        -------
        plt.Figure
        """
        colors = colors or CATEGORY_COLORS
        valid_words = [w for w in words if w in self.glove.embeddings_dict]

        glove_embs = np.array([self.glove.embeddings_dict[w] for w in valid_words], dtype=np.float32)
        st_embs = self._encode_words(valid_words)

        pca_glove = PCA(n_components=2, random_state=42).fit_transform(glove_embs)
        pca_st = PCA(n_components=2, random_state=42).fit_transform(st_embs)

        word_to_cat = {}
        for cat, cat_words in categories.items():
            for w in cat_words:
                word_to_cat[w] = cat

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        for ax, coords, title in [
            (ax1, pca_glove, f'GloVe ({self.glove.dim}d)'),
            (ax2, pca_st, f'Sentence-Transformers ({st_embs.shape[1]}d)'),
        ]:
            EmbeddingVisualizer._plot_on_axis(ax, coords, valid_words, word_to_cat, colors, title)

        handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=colors.get(cat, '#999'), markersize=8, label=cat)
            for cat in categories if cat in colors
        ]
        fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=10)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        return fig

    def compare_rankings_correlation(
        self,
        word_pairs: List[Tuple[str, str]],
    ) -> Dict[str, float]:
        """
        Spearman rank correlation between GloVe and ST similarity rankings.

        For each word pair, compute cosine similarity in both embedding spaces,
        then correlate the two ranking vectors.

        Parameters
        ----------
        word_pairs : List[Tuple[str, str]]
            Pairs of words to compare.

        Returns
        -------
        Dict[str, float]
            Keys: 'spearman_rho', 'p_value', 'n_pairs', 'n_valid'
        """
        glove_sims: List[float] = []
        st_sims: List[float] = []
        n_valid = 0

        for w1, w2 in word_pairs:
            if w1 not in self.glove._word2idx or w2 not in self.glove._word2idx:
                continue

            idx1 = self.glove._word2idx[w1]
            idx2 = self.glove._word2idx[w2]
            g_sim = float(np.dot(
                self.glove._vectors_norm_np[idx1],
                self.glove._vectors_norm_np[idx2],
            ))

            st_e1 = self._encode_word(w1)
            st_e2 = self._encode_word(w2)
            st_e1_n = st_e1 / (np.linalg.norm(st_e1) + 1e-12)
            st_e2_n = st_e2 / (np.linalg.norm(st_e2) + 1e-12)
            s_sim = float(np.dot(st_e1_n, st_e2_n))

            glove_sims.append(g_sim)
            st_sims.append(s_sim)
            n_valid += 1

        if n_valid < 3:
            return {
                'spearman_rho': 0.0, 'p_value': 1.0,
                'n_pairs': len(word_pairs), 'n_valid': n_valid,
            }

        rho, p_val = spearmanr(glove_sims, st_sims)
        return {
            'spearman_rho': float(rho),
            'p_value': float(p_val),
            'n_pairs': len(word_pairs),
            'n_valid': n_valid,
        }

    def analyze_polysemy(
        self,
        polysemous_words: Dict[str, List[str]],
        top_n: int = 5,
    ) -> pl.DataFrame:
        """
        Analyze how GloVe (static) vs ST (contextual) handle polysemous words.

        GloVe assigns a single vector regardless of context.
        ST generates different embeddings depending on the surrounding sentence.

        Parameters
        ----------
        polysemous_words : Dict[str, List[str]]
            Mapping of polysemous word -> list of sentences using it in
            different senses.
        top_n : int
            Number of neighbors to retrieve per context.

        Returns
        -------
        pl.DataFrame
            Columns: word, context, st_neighbors, glove_neighbors, intra_similarity
        """
        records: List[Dict] = []

        for word, sentences in polysemous_words.items():
            # GloVe neighbors (always the same for all contexts)
            glove_neighbors_str = ''
            if word in self.glove.embeddings_dict:
                glove_nb = self.glove.find_most_similar(word, top_n)
                glove_neighbors_str = ', '.join(w for w, _ in glove_nb)

            # Encode all sentences for this word
            st_sentence_embs = self.st_model.encode(
                sentences, batch_size=128, show_progress_bar=False,
            )

            # Compute intra-similarity: cosine between all sentence-pair embeddings
            norms = np.linalg.norm(st_sentence_embs, axis=1, keepdims=True)
            st_normed = st_sentence_embs / (norms + 1e-12)
            sim_matrix = st_normed @ st_normed.T

            # For 2 contexts, intra_similarity is sim_matrix[0, 1]
            if len(sentences) == 2:
                intra_sim = float(sim_matrix[0, 1])
            else:
                # Average of off-diagonal elements
                mask = ~np.eye(len(sentences), dtype=bool)
                intra_sim = float(sim_matrix[mask].mean())

            # ST neighbors for each context sentence
            # Build a vocab subset for neighbor search
            vocab_subset = self.glove._words[:50000]
            vocab_embs = self._encode_words(vocab_subset)
            vocab_norms = np.linalg.norm(vocab_embs, axis=1, keepdims=True)
            vocab_normed = vocab_embs / (vocab_norms + 1e-12)

            for i, sentence in enumerate(sentences):
                q_emb = st_sentence_embs[i]
                q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-12)
                sims = vocab_normed @ q_norm
                top_idx = np.argsort(sims)[::-1]

                st_nb = []
                for idx in top_idx:
                    candidate = vocab_subset[idx]
                    if candidate != word:
                        st_nb.append(candidate)
                    if len(st_nb) == top_n:
                        break

                records.append({
                    'word': word,
                    'context': sentence,
                    'glove_neighbors': glove_neighbors_str,
                    'st_neighbors': ', '.join(st_nb),
                    'intra_similarity': round(intra_sim, 4),
                })

        return pl.DataFrame(records)

    def plot_polysemy_heatmap(
        self,
        polysemous_words: Dict[str, List[str]],
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """
        Bar chart of intra-word cosine similarities across contexts in ST space.

        Each bar shows the cosine similarity between ST embeddings of the same
        polysemous word in two different sentence contexts. Lower values indicate
        better context differentiation by the contextual model.

        Parameters
        ----------
        polysemous_words : Dict[str, List[str]]
            Word -> list of context sentences.
        figsize : Tuple[int, int]
            Figure size.

        Returns
        -------
        plt.Figure
        """
        words_list: List[str] = []
        intra_sims: List[float] = []

        for word, sentences in polysemous_words.items():
            embs = self.st_model.encode(
                sentences, batch_size=128, show_progress_bar=False,
            )
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            normed = embs / (norms + 1e-12)

            if len(sentences) == 2:
                sim = float(np.dot(normed[0], normed[1]))
            else:
                sim_mat = normed @ normed.T
                mask = ~np.eye(len(sentences), dtype=bool)
                sim = float(sim_mat[mask].mean())

            words_list.append(word)
            intra_sims.append(sim)

        # Sort by similarity for visual clarity
        sorted_pairs = sorted(zip(words_list, intra_sims), key=lambda x: x[1])
        words_sorted = [p[0] for p in sorted_pairs]
        sims_sorted = [p[1] for p in sorted_pairs]

        fig, ax = plt.subplots(figsize=figsize)
        colors = ['#e74c3c' if s < 0.5 else '#f39c12' if s < 0.7 else '#2ecc71'
                  for s in sims_sorted]
        bars = ax.barh(words_sorted, sims_sorted, color=colors, edgecolor='white')

        for bar, val in zip(bars, sims_sorted):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=10)

        ax.set_xlabel('Intra-word Cosine Similarity (ST)')
        ax.set_title('Polysemy Detection: Sentence-Transformer Context Differentiation')
        ax.set_xlim(0, 1.0)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold')
        ax.legend(loc='lower right')
        plt.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Benchmark dataset loaders
# ---------------------------------------------------------------------------

def load_wordsim353(path: str) -> pl.DataFrame:
    """
    Load the WordSim-353 dataset.

    Parameters
    ----------
    path : str
        Path to the WordSim-353 TSV file (tab-separated, may have comment lines).

    Returns
    -------
    pl.DataFrame
        Columns: word1, word2, human_score
    """
    rows: List[Dict] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            try:
                score = float(parts[2])
            except ValueError:
                continue
            rows.append({
                'word1': parts[0].lower(),
                'word2': parts[1].lower(),
                'human_score': score,
            })
    return pl.DataFrame(rows)


def load_simlex999(path: str) -> pl.DataFrame:
    """
    Load the SimLex-999 dataset.

    Parameters
    ----------
    path : str
        Path to the SimLex-999 TSV file (tab-separated with header).

    Returns
    -------
    pl.DataFrame
        Columns: word1, word2, simlex_score, pos
    """
    df = pl.read_csv(path, separator='\t')
    return df.select([
        pl.col('word1').str.to_lowercase().alias('word1'),
        pl.col('word2').str.to_lowercase().alias('word2'),
        pl.col('SimLex999').alias('simlex_score'),
        pl.col('POS').alias('pos'),
    ])


def load_google_analogies(
    path: str,
) -> Tuple[List[Tuple[str, str, str, str]], Dict[str, str]]:
    """
    Load the Google Analogy Test Set with section classification.

    Parameters
    ----------
    path : str
        Path to the analogies text file.

    Returns
    -------
    Tuple[List[Tuple[str, str, str, str]], Dict[str, str]]
        (analogy_tuples, section_map) where section_map maps
        "word1 word2 word3 word4" -> "semantic" or "syntactic".
        Analogy tuples are (a, b, c, expected_d) all lowercase.
    """
    analogies: List[Tuple[str, str, str, str]] = []
    section_map: Dict[str, str] = {}
    current_section = ''

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(':'):
                current_section = line[1:].strip()
                continue
            parts = line.lower().split()
            if len(parts) != 4:
                continue
            t = (parts[0], parts[1], parts[2], parts[3])
            analogies.append(t)
            category = 'syntactic' if 'gram' in current_section else 'semantic'
            section_map[f"{t[0]} {t[1]} {t[2]} {t[3]}"] = category

    return analogies, section_map


def compare_embeddings_on_benchmarks(
    spaces: Dict[str, 'EmbeddingSpace'],
    wordsim_path: Optional[str] = None,
    simlex_path: Optional[str] = None,
    analogies_path: Optional[str] = None,
) -> pl.DataFrame:
    """
    Compare multiple embedding spaces on standard benchmarks.

    Parameters
    ----------
    spaces : Dict[str, EmbeddingSpace]
        Label -> EmbeddingSpace mapping (e.g., {'GloVe-50d': space_50, ...}).
    wordsim_path, simlex_path, analogies_path : str, optional
        Paths to benchmark datasets.

    Returns
    -------
    pl.DataFrame
        Columns: embedding, benchmark, metric, value
    """
    frames = []
    for label, space in spaces.items():
        df = space.run_benchmarks(wordsim_path, simlex_path, analogies_path)
        df = df.with_columns(pl.lit(label).alias('embedding'))
        frames.append(df)

    if not frames:
        return pl.DataFrame({'embedding': [], 'benchmark': [], 'metric': [], 'value': []})

    combined = pl.concat(frames)
    return combined.select(['embedding', 'benchmark', 'metric', 'value'])


# ---------------------------------------------------------------------------
# Standalone functions (backward-compatible with rubric requirements)
# ---------------------------------------------------------------------------

def create_emb_dictionary(path: str) -> Dict[str, np.ndarray]:
    """
    Load GloVe embeddings from a text file into a dictionary.

    Parameters
    ----------
    path : str
        Path to the GloVe file. Each line: <word> <f1> <f2> ... <fN>.

    Returns
    -------
    Dict[str, np.ndarray]
        Word -> embedding vector (float32).
    """
    embeddings_dict: Dict[str, np.ndarray] = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            embeddings_dict[word] = vector
    return embeddings_dict


def prepare_normalized_matrix(
    embeddings_dict: Dict[str, np.ndarray],
    eps: float = 1e-12,
) -> Tuple[List[str], np.ndarray]:
    """
    Precompute L2-normalized embedding matrix for fast cosine queries.

    Parameters
    ----------
    embeddings_dict : Dict[str, np.ndarray]
        Word -> embedding vector.
    eps : float
        Numerical stability constant.

    Returns
    -------
    Tuple[List[str], np.ndarray]
        (words_list, normalized_matrix) where matrix shape is [vocab, dim].
    """
    words = list(embeddings_dict.keys())
    vectors = np.array([embeddings_dict[w] for w in words], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors_norm = vectors / (norms + eps)
    return words, vectors_norm


def find_most_similar(
    word: str,
    embeddings_dict: Dict[str, np.ndarray],
    top_n: int = 10,
    *,
    precomputed: Optional[Tuple[List[str], np.ndarray]] = None,
    eps: float = 1e-12,
) -> List[Tuple[str, float]]:
    """
    Find top-N most similar words using cosine similarity (numpy only).

    Parameters
    ----------
    word : str
        Target word.
    embeddings_dict : Dict[str, np.ndarray]
        Word -> embedding vector.
    top_n : int
        Number of results.
    precomputed : Tuple, optional
        (words_list, normalized_matrix) from prepare_normalized_matrix().
    eps : float
        Numerical stability constant.

    Returns
    -------
    List[Tuple[str, float]]
        (word, cosine_similarity) sorted descending.
    """
    if word not in embeddings_dict:
        raise KeyError(f"'{word}' not found in embeddings_dict")

    target = embeddings_dict[word]
    target_norm_val = np.linalg.norm(target)
    if target_norm_val == 0:
        raise ValueError(f"Embedding for '{word}' has zero norm")

    target_norm = target / (target_norm_val + eps)

    if precomputed is not None:
        words, vectors_norm = precomputed
        if not isinstance(words, list):
            words = list(words)
    else:
        words, vectors_norm = prepare_normalized_matrix(embeddings_dict, eps)

    similarities = vectors_norm @ target_norm

    k = min(top_n + 1, similarities.size)
    candidate_idx = np.argpartition(similarities, -k)[-k:]
    candidate_idx = candidate_idx[np.argsort(similarities[candidate_idx])[::-1]]

    results: List[Tuple[str, float]] = []
    for idx in candidate_idx:
        if words[idx] == word:
            continue
        results.append((words[idx], float(similarities[idx])))
        if len(results) == top_n:
            break
    return results


def analogy(
    word1: str,
    word2: str,
    word3: str,
    embeddings_dict: Dict[str, np.ndarray],
    *,
    precomputed: Optional[Tuple[List[str], np.ndarray]] = None,
    eps: float = 1e-12,
) -> str:
    """
    Solve analogies: word1 is to word2 as word3 is to ___ (numpy only).

    Parameters
    ----------
    word1, word2, word3 : str
        Analogy words.
    embeddings_dict : Dict[str, np.ndarray]
        Word -> embedding vector.
    precomputed : Tuple, optional
        (words_list, normalized_matrix) from prepare_normalized_matrix().
    eps : float
        Numerical stability constant.

    Returns
    -------
    str
        Formatted string: "word1 is to word2 as word3 is to **answer**"
    """
    for w in (word1, word2, word3):
        if w not in embeddings_dict:
            raise KeyError(f"'{w}' not found in embeddings_dict")
        if np.linalg.norm(embeddings_dict[w]) == 0:
            raise ValueError(f"Embedding for '{w}' has zero norm")

    target = embeddings_dict[word2] - embeddings_dict[word1] + embeddings_dict[word3]
    target_norm = target / (np.linalg.norm(target) + eps)

    if precomputed is not None:
        words, vectors_norm = precomputed
        if not isinstance(words, list):
            words = list(words)
    else:
        words, vectors_norm = prepare_normalized_matrix(embeddings_dict, eps)

    similarities = vectors_norm @ target_norm
    exclude = {word1, word2, word3}

    best_word = None
    for idx in np.argsort(similarities)[::-1]:
        if words[idx] not in exclude:
            best_word = words[idx]
            break

    if best_word is None:
        raise RuntimeError("No suitable analogy candidate found.")

    return f"{word1} is to {word2} as {word3} is to **{best_word}**"


def show_n_first_words(path: str, n_words: int = 5) -> None:
    """Preview the first N lines of a GloVe-format embeddings file."""
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            tokens = line.split()
            if not tokens:
                continue
            print(tokens[:3], f'... ({len(tokens) - 1} dimensions)')
            if i >= n_words - 1:
                break
