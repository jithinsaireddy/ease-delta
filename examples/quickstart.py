"""Five minutes with EASE-Delta: read evidence, then keep a task current as messages change.

    pip install "ease-delta @ git+https://github.com/jithinsaireddy/ease-delta"
    python examples/quickstart.py

The first run downloads the model (1.6 GB) from the Hugging Face Hub and caches it.
"""

from ease import EvidenceReader, Tracker

reader = EvidenceReader.from_pretrained()

# 1. One claim against one passage: supports, refutes, or settles nothing
claim = "The client has approved the final design."
for passage in ["Email from the client: we approve the final design, please go ahead.",
                "Email from the client: we cannot approve the design yet; the colours are wrong.",
                "The office is closed on the first Monday of the month for staff training."]:
    print(reader.read(claim, passage))

# 2. A task whose decisions follow the messages
task = Tracker.define(
    requirements={
        "approved": "Acme has approved the final design.",
        "date": "Acme has confirmed the delivery date.",
        "nda": "The non-disclosure agreement has been signed by both parties.",
    },
    actions={"send_packet": {"requires": "approved and date and nda", "description": "Send the delivery packet"}},
    reader=reader,
)
# `about` says which requirement a message concerns; leave it out and the message is read against all of them
task.add("mail-1", "Email from Acme: we approve the final design, please go ahead.", about=["approved"])
task.add("mail-2", "Email from Acme: we confirm delivery on 12 March.", about=["date"])
task.add("nda", "The NDA has been signed by Acme and countersigned by us.", about=["nda"])
print(task.status())

# READY needs 90% confidence by default, so it says which answer would get there. Ask it:
print(task.confirm("nda", by="Priya in Legal"))

# The client changes their mind
print(task.add("mail-3", "Email from Acme: we withdraw our approval; the colours are wrong.", about=["approved"]))
print(task.explain("send_packet"))
task.close()
