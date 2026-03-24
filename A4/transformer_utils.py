import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Callable
from collections import Counter, defaultdict
import math
from sacrebleu import corpus_bleu, corpus_chrf, sentence_bleu


class TranslationMetrics:
    """
    Computes standard translation evaluation metrics (BLEU, chrF++, perplexity).

    Uses sacrebleu for corpus and sentence-level BLEU/chrF++ computation.
    Perplexity is computed from model cross-entropy loss on a dataset.
    """

    @staticmethod
    def compute_bleu(
        references: List[str],
        hypotheses: List[str]
    ) -> Dict[str, float]:
        """
        Computes corpus-level BLEU scores (BLEU-1 to BLEU-4).

        Args:
            references: List of reference translations (one per sample).
            hypotheses: List of predicted translations (one per sample).

        Returns:
            Dict with keys 'bleu1', 'bleu2', 'bleu3', 'bleu4' and 'bleu' (corpus BLEU).
        """
        bleu = corpus_bleu(hypotheses, [references])
        scores = bleu.precisions
        return {
            'bleu1': scores[0] if len(scores) > 0 else 0.0,
            'bleu2': scores[1] if len(scores) > 1 else 0.0,
            'bleu3': scores[2] if len(scores) > 2 else 0.0,
            'bleu4': scores[3] if len(scores) > 3 else 0.0,
            'bleu': bleu.score
        }

    @staticmethod
    def compute_chrf(
        references: List[str],
        hypotheses: List[str]
    ) -> float:
        """
        Computes corpus-level chrF++ score.

        Args:
            references: List of reference translations.
            hypotheses: List of predicted translations.

        Returns:
            chrF++ score (0-100).
        """
        chrf = corpus_chrf(hypotheses, [references])
        return chrf.score

    @staticmethod
    def compute_sentence_bleu(
        reference: str,
        hypothesis: str
    ) -> float:
        """
        Computes sentence-level BLEU score.

        Args:
            reference: Single reference translation.
            hypothesis: Single predicted translation.

        Returns:
            Sentence BLEU score (0-100).
        """
        bleu = sentence_bleu(hypothesis, [reference])
        return bleu.score

    @staticmethod
    def compute_perplexity(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        vocab_size: int,
        device: torch.device
    ) -> float:
        """
        Computes perplexity on a dataset.

        Perplexity = exp(average_cross_entropy_loss).
        Computed over the target sequence cross-entropy loss.

        Args:
            model: Transformer model with .eval() method.
            dataloader: DataLoader yielding (source_tensor, target_tensor) batches.
            vocab_size: Size of target vocabulary.
            device: Device to compute on.

        Returns:
            Perplexity score (scalar).
        """
        model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for src_batch, tgt_batch in dataloader:
                src_batch = src_batch.to(device)
                tgt_batch = tgt_batch.to(device)

                logits = model(src_batch, tgt_batch[:, :-1])
                logits_flat = logits.reshape(-1, vocab_size)
                targets_flat = tgt_batch[:, 1:].reshape(-1)

                loss = F.cross_entropy(logits_flat, targets_flat, reduction='sum')
                total_loss += loss.item()
                total_tokens += targets_flat.numel()

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(avg_loss)
        return perplexity

    @staticmethod
    def full_evaluation(
        references: List[str],
        hypotheses: List[str],
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        vocab_size: int,
        device: torch.device
    ) -> Dict[str, float]:
        """
        Computes all translation metrics in one call.

        Args:
            references: List of reference translations.
            hypotheses: List of predicted translations.
            model: Transformer model.
            dataloader: DataLoader for perplexity computation.
            vocab_size: Target vocabulary size.
            device: Device to compute on.

        Returns:
            Dict with keys: 'bleu1', 'bleu2', 'bleu3', 'bleu4', 'bleu', 'chrf', 'perplexity'.
        """
        bleu_scores = TranslationMetrics.compute_bleu(references, hypotheses)
        chrf_score = TranslationMetrics.compute_chrf(references, hypotheses)
        perplexity = TranslationMetrics.compute_perplexity(model, dataloader, vocab_size, device)

        return {
            **bleu_scores,
            'chrf': chrf_score,
            'perplexity': perplexity
        }


class TrainingMonitor:
    """
    Tracks training and validation losses with early stopping and visualization.
    """

    def __init__(self) -> None:
        """Initialize monitor with empty loss lists."""
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def update(
        self,
        train_loss: float,
        val_loss: float
    ) -> None:
        """
        Record losses for current epoch.

        Args:
            train_loss: Training loss for epoch.
            val_loss: Validation loss for epoch.
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

    def plot_learning_curves(
        self,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot training and validation loss curves.

        Args:
            save_path: If provided, save figure to this path. Otherwise show.
        """
        if not self.train_losses or not self.val_losses:
            print("No loss data to plot.")
            return

        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def check_early_stopping(
        self,
        patience: int = 5
    ) -> bool:
        """
        Check if validation loss has not improved for patience epochs.

        Args:
            patience: Number of epochs without improvement before stopping.

        Returns:
            True if should stop training, False otherwise.
        """
        if len(self.val_losses) < patience + 1:
            return False

        best_val_loss = min(self.val_losses[:-patience])
        current_val_loss = self.val_losses[-1]

        return current_val_loss > best_val_loss

    def get_best_epoch(self) -> int:
        """
        Get epoch index (1-indexed) with lowest validation loss.

        Returns:
            Epoch number with best validation loss.
        """
        if not self.val_losses:
            return 0
        return int(np.argmin(self.val_losses)) + 1


class AttentionVisualizer:
    """
    Visualizes attention weights from Transformer models.
    Supports single-head, multi-head, and cross-layer comparisons.
    """

    @staticmethod
    def plot_attention_heatmap(
        attention_weights: torch.Tensor,
        src_tokens: List[str],
        tgt_tokens: List[str],
        head: int = 0,
        layer: int = -1,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot attention weights for a single head as a heatmap.

        attention_weights shape: (num_layers, batch_size, num_heads, tgt_len, src_len)
        or (num_layers, num_heads, tgt_len, src_len) for single-sample inference.

        Args:
            attention_weights: Attention tensor from model.
            src_tokens: List of source tokens.
            tgt_tokens: List of target tokens.
            head: Head index to visualize (default last head).
            layer: Layer index to visualize (default last layer, -1).
            save_path: Path to save figure. If None, show plot.
        """
        weights = attention_weights
        if weights.dim() == 5:
            weights = weights[layer, 0, head, :, :]
        elif weights.dim() == 4:
            weights = weights[layer, head, :, :]
        else:
            weights = weights[layer, :, :]

        weights = weights.detach().cpu().numpy()

        plt.figure(figsize=(10, 8))
        sns.heatmap(weights, xticklabels=src_tokens, yticklabels=tgt_tokens,
                    cmap='viridis', cbar_kws={'label': 'Attention Weight'})
        plt.xlabel('Source Tokens', fontsize=12)
        plt.ylabel('Target Tokens', fontsize=12)
        plt.title(f'Attention Weights (Layer {layer}, Head {head})', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_multi_head_attention(
        attention_weights: torch.Tensor,
        src_tokens: List[str],
        tgt_tokens: List[str],
        layer: int = -1,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot attention for all heads in a layer (2x4 grid for 8 heads).

        Args:
            attention_weights: Attention tensor from model.
            src_tokens: List of source tokens.
            tgt_tokens: List of target tokens.
            layer: Layer index to visualize.
            save_path: Path to save figure.
        """
        weights = attention_weights
        if weights.dim() == 5:
            weights = weights[layer, 0, :, :, :]
        elif weights.dim() == 4:
            weights = weights[layer, :, :, :]

        num_heads = weights.shape[0]
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for head_idx in range(min(num_heads, 8)):
            head_weights = weights[head_idx, :, :].detach().cpu().numpy()
            sns.heatmap(head_weights, ax=axes[head_idx], cmap='viridis',
                       xticklabels=src_tokens, yticklabels=tgt_tokens,
                       cbar_kws={'label': 'Weight'})
            axes[head_idx].set_title(f'Head {head_idx}', fontsize=10)
            axes[head_idx].set_xlabel('Source', fontsize=9)
            axes[head_idx].set_ylabel('Target', fontsize=9)

        for idx in range(num_heads, 8):
            fig.delaxes(axes[idx])

        plt.suptitle(f'Multi-Head Attention (Layer {layer})', fontsize=14, y=1.00)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_layer_comparison(
        attention_weights_by_layer: List[torch.Tensor],
        src_tokens: List[str],
        tgt_tokens: List[str],
        head: int = 0,
        save_path: Optional[str] = None
    ) -> None:
        """
        Compare attention across layers for a single head.

        Args:
            attention_weights_by_layer: List of attention tensors, one per layer.
            src_tokens: List of source tokens.
            tgt_tokens: List of target tokens.
            head: Head index to visualize.
            save_path: Path to save figure.
        """
        num_layers = len(attention_weights_by_layer)
        fig, axes = plt.subplots(1, num_layers, figsize=(4*num_layers, 4))

        if num_layers == 1:
            axes = [axes]

        for layer_idx, attn in enumerate(attention_weights_by_layer):
            if attn.dim() == 4:
                weights = attn[head, :, :].detach().cpu().numpy()
            elif attn.dim() == 3:
                weights = attn[:, :].detach().cpu().numpy()
            else:
                weights = attn.detach().cpu().numpy()

            sns.heatmap(weights, ax=axes[layer_idx], cmap='viridis',
                       xticklabels=src_tokens, yticklabels=tgt_tokens,
                       cbar_kws={'label': 'Weight'})
            axes[layer_idx].set_title(f'Layer {layer_idx}', fontsize=11)
            axes[layer_idx].set_xlabel('Source', fontsize=10)
            axes[layer_idx].set_ylabel('Target', fontsize=10)

        plt.suptitle(f'Attention Across Layers (Head {head})', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class BeamSearchDecoder:
    """
    Beam search decoder for sequence-to-sequence translation.
    Maintains top-k hypotheses and uses length normalization.
    """

    def __init__(
        self,
        model: nn.Module,
        beam_width: int = 5,
        max_len: int = 128,
        device: torch.device = None
    ) -> None:
        """
        Initialize beam search decoder.

        Args:
            model: Transformer model with forward(source, target) interface.
            beam_width: Number of top hypotheses to track.
            max_len: Maximum target sequence length.
            device: Device to run on. If None, auto-detect.
        """
        self.model = model
        self.beam_width = beam_width
        self.max_len = max_len
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def decode(
        self,
        src_tensor: torch.Tensor,
        sos_idx: int,
        eos_idx: int
    ) -> List[Tuple[List[int], float]]:
        """
        Perform beam search decoding on a source tensor.

        Args:
            src_tensor: Source token tensor, shape (1, src_len).
            sos_idx: Index of start-of-sequence token.
            eos_idx: Index of end-of-sequence token.

        Returns:
            List of (sequence, score) tuples, sorted by score (descending).
            Scores are length-normalized log probabilities.
        """
        src_tensor = src_tensor.to(self.device)
        batch_size = 1

        hypotheses = [([sos_idx], 0.0)]

        for step in range(self.max_len):
            all_candidates = []

            for seq, score in hypotheses:
                if seq[-1] == eos_idx:
                    all_candidates.append((seq, score))
                    continue

                tgt_tensor = torch.tensor([seq], dtype=torch.long, device=self.device)

                with torch.no_grad():
                    logits = self.model(src_tensor, tgt_tensor)

                log_probs = F.log_softmax(logits[0, -1, :], dim=0)

                top_k_log_probs, top_k_indices = torch.topk(log_probs, self.beam_width)

                for idx, log_prob in zip(top_k_indices, top_k_log_probs):
                    new_seq = seq + [idx.item()]
                    new_score = score + log_prob.item()
                    all_candidates.append((new_seq, new_score))

            scored_candidates = []
            for seq, score in all_candidates:
                if seq[-1] == eos_idx:
                    alpha = 0.6
                    normalized_score = score / (len(seq) ** alpha)
                    scored_candidates.append((seq, normalized_score))
                else:
                    scored_candidates.append((seq, score / (len(seq) ** 0.6)))

            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            hypotheses = scored_candidates[:self.beam_width]

            if all(seq[-1] == eos_idx for seq, _ in hypotheses):
                break

        final_results = []
        for seq, score in hypotheses:
            alpha = 0.6
            normalized = score / (len(seq) ** alpha)
            final_results.append((seq, normalized))

        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results

    def translate(
        self,
        sentence: str,
        eng_word2idx: Dict[str, int],
        spa_word2idx: Dict[str, int],
        spa_idx2word: Dict[int, str],
        preprocess_fn: Callable,
        beam_width: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Translate a sentence using beam search.

        Args:
            sentence: Input sentence in English.
            eng_word2idx: English word-to-index mapping.
            spa_word2idx: Spanish word-to-index mapping (for sos/eos tokens).
            spa_idx2word: Spanish index-to-word mapping.
            preprocess_fn: Preprocessing function (e.g., preprocess_sentence).
            beam_width: Override beam width for this translation.

        Returns:
            List of (translation, score) tuples, best first.
        """
        processed = preprocess_fn(sentence)
        indices = [eng_word2idx.get(word, eng_word2idx.get('<unk>', 1)) for word in processed.split()]

        src_tensor = torch.tensor([indices], dtype=torch.long, device=self.device)

        sos_idx = spa_word2idx.get('<sos>', 2)
        eos_idx = spa_word2idx.get('<eos>', 3)

        old_beam_width = self.beam_width
        self.beam_width = beam_width

        sequences = self.decode(src_tensor, sos_idx, eos_idx)

        self.beam_width = old_beam_width

        translations = []
        for seq, score in sequences:
            words = []
            for idx in seq:
                word = spa_idx2word.get(idx, '<unk>')
                if word in ('<sos>', '<eos>', '<pad>'):
                    continue
                words.append(word)
            translation = ' '.join(words)
            translations.append((translation, float(score)))

        return translations


class ErrorAnalyzer:
    """
    Analyzes translation errors by type, length, and patterns.
    """

    @staticmethod
    def categorize_errors(
        reference: str,
        hypothesis: str
    ) -> Dict[str, int]:
        """
        Categorize errors in a single translation.

        Categories: omission (missing words), substitution (wrong words),
        addition (extra words), reordering (word order issues).

        Args:
            reference: Reference translation.
            hypothesis: Predicted translation.

        Returns:
            Dict with error type counts.
        """
        ref_words = reference.split()
        hyp_words = hypothesis.split()

        errors = {
            'omission': 0,
            'substitution': 0,
            'addition': 0,
            'reordering': 0
        }

        ref_counter = Counter(ref_words)
        hyp_counter = Counter(hyp_words)

        for word, count in ref_counter.items():
            if word not in hyp_counter:
                errors['omission'] += count
            elif hyp_counter[word] < count:
                errors['omission'] += count - hyp_counter[word]

        for word, count in hyp_counter.items():
            if word not in ref_counter:
                errors['addition'] += count
            elif hyp_counter[word] > ref_counter[word]:
                errors['addition'] += count - ref_counter[word]

        common_words = set(ref_words) & set(hyp_words)
        if common_words:
            ref_order = [w for w in ref_words if w in common_words]
            hyp_order = [w for w in hyp_words if w in common_words]
            if ref_order != hyp_order:
                errors['reordering'] = len(ref_order) // 3 + 1

        errors['substitution'] = max(0, len(hyp_words) - len(ref_words) - errors['addition'])

        return errors

    @staticmethod
    def analyze_by_length(
        references: List[str],
        hypotheses: List[str],
        sources: List[str]
    ) -> pd.DataFrame:
        """
        Compute metrics grouped by source sentence length.

        Args:
            references: List of reference translations.
            hypotheses: List of predicted translations.
            sources: List of source sentences.

        Returns:
            DataFrame with columns: length_bucket, count, avg_bleu, avg_chrf.
        """
        from sacrebleu import sentence_bleu, sentence_chrf

        length_buckets = defaultdict(lambda: {'bleu': [], 'chrf': []})

        for src, ref, hyp in zip(sources, references, hypotheses):
            src_len = len(src.split())
            bucket = f"{(src_len // 5) * 5}-{(src_len // 5) * 5 + 4}"

            bleu = sentence_bleu(hyp, [ref]).score
            chrf = sentence_chrf(hyp, [ref]).score

            length_buckets[bucket]['bleu'].append(bleu)
            length_buckets[bucket]['chrf'].append(chrf)

        rows = []
        for bucket in sorted(length_buckets.keys()):
            data = length_buckets[bucket]
            rows.append({
                'length_bucket': bucket,
                'count': len(data['bleu']),
                'avg_bleu': np.mean(data['bleu']),
                'avg_chrf': np.mean(data['chrf'])
            })

        return pd.DataFrame(rows)

    @staticmethod
    def analyze_common_errors(
        references: List[str],
        hypotheses: List[str]
    ) -> pd.DataFrame:
        """
        Aggregate error analysis across all translations.

        Args:
            references: List of reference translations.
            hypotheses: List of predicted translations.

        Returns:
            DataFrame with error type totals and percentages.
        """
        error_totals = {'omission': 0, 'substitution': 0, 'addition': 0, 'reordering': 0}

        for ref, hyp in zip(references, hypotheses):
            errors = ErrorAnalyzer.categorize_errors(ref, hyp)
            for error_type, count in errors.items():
                error_totals[error_type] += count

        total_errors = sum(error_totals.values())
        percentages = {k: (v / total_errors * 100) if total_errors > 0 else 0
                      for k, v in error_totals.items()}

        return pd.DataFrame({
            'error_type': list(error_totals.keys()),
            'count': list(error_totals.values()),
            'percentage': list(percentages.values())
        })

    @staticmethod
    def plot_bleu_vs_length(
        references: List[str],
        hypotheses: List[str],
        sources: List[str],
        save_path: Optional[str] = None
    ) -> None:
        """
        Scatter plot: source length vs BLEU score.

        Args:
            references: List of reference translations.
            hypotheses: List of predicted translations.
            sources: List of source sentences.
            save_path: Path to save figure.
        """
        from sacrebleu import sentence_bleu

        lengths = [len(src.split()) for src in sources]
        bleus = [sentence_bleu(hyp, [ref]).score for ref, hyp in zip(references, hypotheses)]

        plt.figure(figsize=(10, 6))
        plt.scatter(lengths, bleus, alpha=0.6, s=50)
        plt.xlabel('Source Sentence Length (tokens)', fontsize=12)
        plt.ylabel('BLEU Score', fontsize=12)
        plt.title('BLEU Score vs Source Length', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_error_distribution(
        references: List[str],
        hypotheses: List[str],
        save_path: Optional[str] = None
    ) -> None:
        """
        Bar chart of error type distribution.

        Args:
            references: List of reference translations.
            hypotheses: List of predicted translations.
            save_path: Path to save figure.
        """
        error_df = ErrorAnalyzer.analyze_common_errors(references, hypotheses)

        plt.figure(figsize=(10, 6))
        plt.bar(error_df['error_type'], error_df['count'], color='steelblue', alpha=0.8)
        plt.xlabel('Error Type', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Translation Error Distribution', fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class WarmupCosineScheduler:
    """
    Learning rate scheduler with warmup and cosine annealing.

    Implements Vaswani et al. formula from "Attention is All You Need":
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int = 4000
    ) -> None:
        """
        Initialize scheduler.

        Args:
            optimizer: PyTorch optimizer to schedule.
            d_model: Model dimension (embedding size).
            warmup_steps: Number of warmup steps.
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_count = 0

    def step(self) -> None:
        """Increment step counter and update learning rate."""
        self.step_count += 1
        lr = self.get_lr()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        """
        Compute current learning rate using Vaswani formula.

        Returns:
            Current learning rate (scalar).
        """
        d_inv_sqrt = self.d_model ** (-0.5)
        step_inv_sqrt = self.step_count ** (-0.5)
        warmup_inv_sqrt = self.warmup_steps ** (-1.5)

        lr = d_inv_sqrt * min(step_inv_sqrt, self.step_count * warmup_inv_sqrt)
        return lr
