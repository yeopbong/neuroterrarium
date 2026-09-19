import test from 'node:test';
import assert from 'node:assert/strict';
import {checkInvocation, selectRelease} from './release_gate.mjs';
const repo = 'yeopbong/neuroterrarium';
const sha = 'a'.repeat(40);
const release = {tag_name: 'v0.1.0', draft: false, prerelease: false, published_at: '2026-09-13T12:00:00Z'};
const green = {id: 20, run_attempt: 1, head_sha: sha, repository: {full_name: repo},
  head_repository: {full_name: repo}, path: '.github/workflows/ci.yml', event: 'push',
  status: 'completed', conclusion: 'success'};
const verify = (r = release, runs = [green], commit = {sha}) => selectRelease(r, commit, runs, repo, 'v0.1.0');
test('only the exact published tag with its successful latest core CI is accepted', () => {
  assert.equal(verify().sha, sha);
  assert.throws(() => verify({...release, draft: true}));
  assert.throws(() => verify({...release, prerelease: true}));
  assert.throws(() => verify({...release, tag_name: 'v0.2.0'}));
  assert.throws(() => verify({...release, published_at: null}));
  assert.throws(() => verify(release, [green], {sha: 'b'.repeat(40)}));
});
test('pending, failed, cancelled and newer reruns cannot use an earlier green result', () => {
  for (const status of ['queued', 'in_progress', 'waiting']) {
    assert.throws(() => verify(release, [green, {...green, id: 21, status, conclusion: null}]));
  }
  for (const conclusion of ['failure', 'cancelled', 'skipped', 'neutral', 'timed_out']) {
    assert.throws(() => verify(release, [green, {...green, id: 21, conclusion}]));
  }
  assert.throws(() => verify(release, [{...green, run_attempt: 2, status: 'in_progress', conclusion: null}]));
});
test('unrelated workflow, pull request and foreign-repository runs cannot authorize deployment', () => {
  assert.throws(() => verify(release, []));
  for (const patch of [{path: '.github/workflows/pages.yml'}, {event: 'pull_request'},
    {head_repository: {full_name: 'some/fork'}}, {repository: {full_name: 'other/repo'}}]) {
    assert.throws(() => verify(release, [{...green, ...patch}]));
  }
});
test('manual dispatch cannot bypass release or trusted workflow requirements', () => {
  assert.equal(checkInvocation({}, 'workflow_dispatch', 'refs/heads/main', 'main', 'v0.1.0'), 'v0.1.0');
  assert.throws(() => checkInvocation({}, 'workflow_dispatch', 'refs/heads/feature', 'main', 'v0.1.0'));
  assert.throws(() => checkInvocation({}, 'push', 'refs/heads/main', 'main', 'v0.1.0'));
  assert.throws(() => checkInvocation({action: 'created'}, 'release', 'refs/tags/v0.1.0'));
  assert.equal(checkInvocation({action: 'published', release}, 'release', 'refs/tags/v0.1.0'), 'v0.1.0');
});
