"""The demo scenario, driven by an oracle reader: checks the bookkeeping the demo is meant to show."""

from ease import demo
from ease.aggregate import RuleAggregator
from ease.labels import REFUTES, SUPPORTS
from ease.runtime import Runtime
from ease.scorer import OracleScorer

D = 8


def gold():
    s = demo.packet_schema()
    t = {p.id: p.text for p in s.predicates}
    e = {ev.record_id + str(ev.revision): ev.text for _, ev in demo.scenario() if ev is not None and ev.text}
    return {(t["nda_signed"], e["nda1"]): SUPPORTS,
            (t["design_approved"], e["approval1"]): SUPPORTS,
            (t["design_approved"], e["approval2"]): REFUTES,
            (t["design_approved"], e["approval3"]): SUPPORTS,
            (t["date_confirmed"], e["date1"]): SUPPORTS,
            (t["date_confirmed"], e["date2"]): REFUTES,
            (t["date_confirmed"], e["date3"]): SUPPORTS,
            (t["invoice_paid"], e["payment1"]): SUPPORTS}


def test_demo_bookkeeping(tmp_path):
    sc = OracleScorer(gold(), margin=14.0, msg_dim=D)
    sc.cache = {}
    rt = Runtime(tmp_path / "d", sc, RuleAggregator(msg_dim=D), tau=0.9)
    lines = []
    res = demo.run(rt, out=lines.append)
    text = "\n".join(lines)
    a = res["state"]["assessments"]
    assert a["send_packet"]["disposition"] == "READY" and a["send_packet"]["needs_approval"]
    assert a["send_payment_reminder"]["disposition"] == "NEEDS_INFO", "payment record withdrawn: unknown again"
    assert res["verify"]["not_bitwise_equal"] == 0
    assert res["totals"]["unchanged"] == 2, "the duplicate and the late archive copy change nothing"
    assert "send_packet: READY -> BLOCKED" in text and "send_packet: NEEDS_INFO -> READY" in text or \
           "send_packet: BLOCKED -> NEEDS_INFO" in text
    assert res["state"]["records"]["approval"]["revision"] == 3
    rt.close()
