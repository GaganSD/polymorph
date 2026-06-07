"""Phase 0d: the decode-time structural insurance floor."""

from polymorph_lamr.bench.methods import KeepSeverityHeuristic, RandomDropFloor
from polymorph_lamr.bench.structural import structural_keep_mask, structural_spans
from polymorph_lamr.bench.survival import answer_survives


def test_locker_catches_atoms():
    text = "GET /x 200 ok\nPOST /y HTTP status=503 from 10.0.0.5 ValueError: boom INC00042"
    spans = structural_spans(text)
    assert spans, "expected structural matches"
    _ids, _sp, force = structural_keep_mask(text)
    assert any(force), "expected some force-kept tokens"


def test_floor_never_drops_a_structural_atom_even_at_full_drop():
    # An atom on a NON-severe line — exactly what keep-severity drops.
    text = "\n".join(f"INFO heartbeat ok seq={i}" for i in range(40))
    text += "\nINFO request done from 192.168.1.250 status=502"
    floored = RandomDropFloor(floor=True)
    for r in (0.5, 0.8, 0.95):
        comp = floored.compress(text, r)
        assert answer_survives("192.168.1.250", comp), f"IP dropped at R={r}"
        assert answer_survives("502", comp), f"status dropped at R={r}"


def test_random_plus_floor_beats_keep_severity_on_nonsevere_atom():
    # The non-circular demonstration: a needle (IP) on a non-severe line. A blind
    # random ranker drops whole lines indiscriminately; keep-severity drops the
    # non-severe line entirely; only the structural floor protects the atom.
    text = "\n".join(f"INFO tick {i} ok" for i in range(60))
    text += "\nDEBUG connection from 203.0.113.77 established"
    needle = "203.0.113.77"
    r = 0.8
    assert answer_survives(needle, RandomDropFloor(floor=True).compress(text, r))
    # keep-severity has no severe line here, so the needle's line is droppable.
    assert not answer_survives(needle, KeepSeverityHeuristic().compress(text, r))


def test_floor_off_is_plain_random():
    plain = RandomDropFloor()
    assert plain.name == "random"
    floored = RandomDropFloor(floor=True)
    assert floored.name == "random+floor"
