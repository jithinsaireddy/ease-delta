# Misreadings reported by people

`misreadings.jsonl` holds cases where a released reader read a message wrongly, as reported on GitHub with the
[Misreading form](https://github.com/jithinsaireddy/ease-delta/issues/new?template=misreading.yml). The file
appears with the first report whose author agrees to publication; until then there is nothing here to score.

It is built by `python scripts/misreadings.py sync` from the issues labelled `misreading`, and rebuilt from the
issues as they stand each time:

- only reports whose author answered "Yes" to the form's consent question are included; a report without consent is
  counted and nothing from it is stored;
- a report a maintainer labels `invalid` (the answer it expects is wrong) or `duplicate` is left out;
- the same case reported twice is kept once; a case reported with different expected answers is left out;
- an edited report replaces its row, and a withdrawn consent removes it.

One JSON object per line, sorted by issue number:

| Field | |
|---|---|
| `id`, `issue`, `url` | `gh-<n>`, the issue number and its address |
| `author`, `created` | who reported it, and the day the issue was opened |
| `claim`, `passage` | the requirement and the message, as reported |
| `reported_label` | what the reader said: `SUPPORTS`, `REFUTES` or `NOT_ENOUGH_INFO` |
| `gold_label` | what the reporter says it should have said |
| `model` | the model the reporter used, if given |
| `licence` | `CC-BY-4.0` |

`python scripts/misreadings.py score` measures both released readers on the file; `--save` also writes
`runs/results/misreadings.json` with each reader's revision, temperature and per-case answers.

**Licence.** Each case is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) by its reporter,
who is named in the `author` field and linked by `url`; attribute them when you reuse a case. The rest of this
repository's licences are unchanged: code Apache-2.0, weights CC BY-SA 4.0.

Cases from elsewhere, such as a pilot, go in through the same form, so every example has its consent on record.
