"""Training-only local teacher alignment; no native model forward is replaced."""
from __future__ import annotations
import math
import weakref
import torch
from torch import nn
from torch.nn import functional as F


def require(condition, message):
    if not condition:
        raise ValueError(message)


class LocalEvidenceHead(nn.Module):
    """96 -> 3 affine head initialized without touching global RNG streams."""
    def __init__(self, rank=96, *, seed, device=None):
        super().__init__()
        require(rank > 0, "Rank must be positive")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 1000003)
        weights = torch.randn((3, rank), generator=generator, dtype=torch.float32) * .01
        self.weight = nn.Parameter(weights.to(device=device))
        self.bias = nn.Parameter(torch.zeros(3, dtype=torch.float32, device=device))

    def forward(self, messages):
        require(messages.ndim == 2 and messages.shape[1] == self.weight.shape[1],
                "Expected image-by-rank messages")
        return F.linear(messages, self.weight, self.bias)


def local_teacher_kl(head, messages, probabilities, *, detach_messages):
    """Mean-image KL(teacher || student), temperature one, no confidence weights."""
    require(messages.ndim == 2 and messages.shape[0] > 0, "Expected nonempty messages")
    require(probabilities.shape == (messages.shape[0], 3), "Teacher/image order or shape mismatch")
    require(not probabilities.requires_grad and bool(torch.isfinite(probabilities).all())
            and bool((probabilities >= 0).all()), "Teacher probabilities must be frozen and valid")
    require(torch.allclose(probabilities.sum(-1), torch.ones_like(probabilities[:, 0]),
                           rtol=0, atol=2e-6), "Teacher probabilities do not sum to one")
    logits = head(messages.detach() if detach_messages else messages)
    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    loss = F.kl_div(log_probabilities, probabilities.to(log_probabilities), reduction="batchmean")
    return loss, logits


def last_prompt_messages(branch, hidden, image_memory, image_ends, query_positions,
                         language_mask, *, prompt_length, total_length, expected_images):
    """Exact rank-space formula at prompt_length-1 with a live autograd graph.

    One-row projection can select a different arithmetic kernel from the full
    prefill; equivalence here is mathematical, without modifying count logits.
    """
    require(branch.mode == "all" and branch.merge == "sum", "V6 requires active native SUM")
    require(hidden.ndim == 3 and hidden.shape == (1, total_length, branch.hidden_size),
            "Unexpected training hidden shape")
    require(0 < prompt_length <= total_length, "Original prompt index is out of bounds")
    require(query_positions.shape == (total_length,)
            and torch.equal(query_positions, torch.arange(total_length, device=hidden.device)),
            "Auxiliary observer requires an uncached contiguous training prefill")
    require(language_mask.shape == (total_length,) and language_mask.dtype == torch.bool,
            "Invalid language mask")
    index = prompt_length - 1
    require(bool(language_mask[index]), "Original final prompt token must be a language query")
    require(expected_images > 0 and len(image_memory) == expected_images
            and image_ends.shape == (expected_images,), "Image count or visibility metadata changed")
    require(bool((query_positions[index] > image_ends).all()),
            "Every teacher image must be complete before the original prompt query")
    query = branch.query(branch.rms(hidden[0, index:index + 1]))
    messages = []
    for memory in image_memory:
        require(memory.ndim == 2 and memory.shape[0] > 0 and memory.shape[1] == branch.hidden_size,
                "Invalid ordinary visual memory")
        kv = branch.memory(branch.rms(memory))
        weights = torch.softmax((query @ kv.transpose(0, 1)) / math.sqrt(branch.rank), dim=-1)
        read = weights @ kv
        messages.append((F.silu(branch.read(read) + query) - F.silu(branch.read.bias + query))[0])
    return torch.stack(messages)


class LastPromptMessageObserver:
    """One-shot input observer; consume() releases its retained graph reference."""
    def __init__(self, branch):
        self.branch_ref = weakref.ref(branch)
        self.expected = self.messages = None
        self.calls = 0
        self.handle = branch.register_forward_pre_hook(self._before_branch)

    def begin(self, *, prompt_length, total_length, expected_images):
        require(self.handle is not None and self.expected is None and self.messages is None,
                "Observer removed or previous training call unconsumed")
        require(torch.is_grad_enabled(), "Training observer requires autograd")
        self.expected = dict(prompt_length=int(prompt_length), total_length=int(total_length),
                             expected_images=int(expected_images))
        self.calls = 0

    def _before_branch(self, branch, args):
        if self.expected is None:
            return None
        require(len(args) == 5 and self.calls == 0, "Expected exactly one ordinary branch call")
        self.calls += 1
        self.messages = last_prompt_messages(branch, *args, **self.expected)
        return None

    def consume(self):
        require(self.expected is not None and self.calls == 1 and self.messages is not None,
                "Observer did not capture exactly one original prompt query")
        messages, metadata = self.messages, dict(self.expected, branch_calls=self.calls)
        self.cancel()
        return messages, metadata

    def cancel(self):
        self.expected = self.messages = None
        self.calls = 0

    def remove(self):
        self.cancel()
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
