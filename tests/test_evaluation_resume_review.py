"""Independent recovery-boundary checks using development-only synthetic graphs."""
import json

import pytest

from neuroterrarium import evaluation as ev
import test_evaluation_budget as budget_cases

completed_fixture=budget_cases.completed_fixture
evaluated_fixture=budget_cases.evaluated_fixture
resources=budget_cases.resources
evidence_bytes_and_times=budget_cases.evidence_bytes_and_times


def test_completed_resume_rejects_false_window_count_with_unchanged_numeric_evidence(completed_fixture):
    config,path=completed_fixture
    raw=path/'raw/rule--none--A_static--7101.json'
    row=json.loads(raw.read_text());row['completed_windows']=1
    raw.write_text(json.dumps(row))
    result=path/'results.json';data=json.loads(result.read_text());data['records'][0]=row
    result.write_text(json.dumps(data))
    before=evidence_bytes_and_times(path)
    with pytest.raises(ValueError,match='(duration|window|integrity)'):
        ev.evaluate(config,'unused','unused',path)
    assert evidence_bytes_and_times(path)==before


def test_completed_resume_rejects_unplanned_raw_without_deleting_it(completed_fixture):
    config,path=completed_fixture
    extra=path/'raw/unplanned-development.json';extra.write_text('{}')
    before=evidence_bytes_and_times(path)
    with pytest.raises(ValueError,match='Unexpected records'):
        ev.evaluate(config,'unused','unused',path)
    assert evidence_bytes_and_times(path)==before


def test_completed_index_is_last_commit_after_audited_budget(evaluated_fixture,monkeypatch):
    config,path=evaluated_fixture
    original=ev.atomic_json
    class AbruptExit(BaseException):
        pass
    def stop_after_final_index(file,value):
        original(file,value)
        if file.name=='results.json' and value.get('status')=='completed':
            raise AbruptExit()
    monkeypatch.setattr(ev,'atomic_json',stop_after_final_index)
    with pytest.raises(AbruptExit):ev.evaluate(config,'unused','unused',path)
    monkeypatch.setattr(ev,'atomic_json',original)
    before=evidence_bytes_and_times(path)
    assert ev.evaluate(config,'unused','unused',path)['status']=='verified'
    assert evidence_bytes_and_times(path)==before
