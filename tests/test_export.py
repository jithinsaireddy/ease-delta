"""The edge model written as a plain transformers classifier computes the same logits."""

import pytest
import torch

from ease.export import LABELS, to_sequence_classifier


def tiny_edge(hidden=256, msg=128, seed=0):
    from transformers import ModernBertConfig, ModernBertModel

    from ease.model.edge import EdgeConfig, EdgeModel

    torch.manual_seed(seed)
    cfg = ModernBertConfig(vocab_size=120, hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=2,
                           num_attention_heads=4, max_position_embeddings=64, pad_token_id=0, bos_token_id=1,
                           eos_token_id=2, cls_token_id=1, sep_token_id=2, global_attn_every_n_layers=2,
                           local_attention=16)
    edge = EdgeModel.__new__(EdgeModel)
    torch.nn.Module.__init__(edge)
    edge.cfg = EdgeConfig(encoder_name="tiny", msg_dim=msg, pooling="cls")
    edge.encoder = ModernBertModel(cfg)
    edge.proj = torch.nn.Linear(hidden, msg)
    edge.norm = torch.nn.LayerNorm(msg)
    edge.drop = torch.nn.Dropout(0.1)
    edge.classifier = torch.nn.Linear(msg, 3)
    with torch.no_grad():  # trained-looking values: a LayerNorm with non-trivial weight and bias
        edge.norm.weight.uniform_(0.5, 1.5)
        edge.norm.bias.uniform_(-0.3, 0.3)
    return edge.eval()


@pytest.mark.parametrize("hidden", [128, 256, 384])
def test_exported_classifier_computes_the_edge_models_logits(hidden):
    edge = tiny_edge(hidden=hidden, seed=hidden)
    clf = to_sequence_classifier(edge)
    assert [clf.config.id2label[i] for i in range(3)] == list(LABELS)
    torch.manual_seed(1)
    ids = torch.randint(3, 120, (6, 20))
    ids[:, 0] = 1
    mask = torch.ones_like(ids)
    mask[3:, 15:] = 0  # some padded rows
    ids[3:, 15:] = 0
    with torch.no_grad():
        want, _ = edge(ids, mask)
        got = clf(input_ids=ids, attention_mask=mask).logits
    assert torch.allclose(want, got, atol=1e-5, rtol=1e-5), (want - got).abs().max()
    assert torch.equal(want.argmax(-1), got.argmax(-1))


def test_a_hidden_size_that_is_not_a_multiple_is_refused():
    edge = tiny_edge(hidden=256, msg=96)
    with pytest.raises(ValueError):
        to_sequence_classifier(edge)
