/** Resolve an official release and require the newest CI run for its exact commit. */
import {appendFileSync, readFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';

export function selectRelease(release, commit, runs, repository, tag) {
  if (!/^v\d+\.\d+\.\d+$/.test(tag) || release.tag_name !== tag || release.draft !== false ||
      release.prerelease !== false || !release.published_at || !Number.isFinite(Date.parse(release.published_at))) {
    throw new Error('A published, stable version release is required');
  }
  const sha = commit.sha;
  if (!/^[a-f0-9]{40}$/.test(sha)) throw new Error('Release tag does not resolve to a commit');
  const matching = runs.filter(run => run.head_sha === sha &&
    run.repository?.full_name === repository && run.head_repository?.full_name === repository &&
    run.path === '.github/workflows/ci.yml' && ['push', 'workflow_dispatch'].includes(run.event));
  matching.sort((a, b) => b.id - a.id);
  const newest = matching[0];
  if (!newest || newest.status !== 'completed' || newest.conclusion !== 'success') {
    throw new Error('The newest core CI run for the exact release commit must be completed and successful');
  }
  if (!Number.isSafeInteger(newest.id) || newest.id <= 0) throw new Error('Invalid CI run identity');
  return {tag, sha, ci_run_id: newest.id, ci_run_attempt: newest.run_attempt};
}

export function checkInvocation(event, eventName, ref, defaultBranch, requestedTag) {
  if (eventName === 'release') {
    if (event.action !== 'published') throw new Error('Only published release events are allowed');
    return event.release?.tag_name;
  }
  if (eventName !== 'workflow_dispatch' || ref !== `refs/heads/${defaultBranch}`) {
    throw new Error('Manual deployment must run the workflow from the default branch');
  }
  return requestedTag;
}

async function run() {
  const repository = process.env.GITHUB_REPOSITORY;
  if (repository !== 'yeopbong/neuroterrarium') throw new Error('Unexpected publishing repository');
  const event = JSON.parse(readFileSync(process.env.GITHUB_EVENT_PATH, 'utf8'));
  const tag = checkInvocation(event, process.env.GITHUB_EVENT_NAME, process.env.GITHUB_REF,
    event.repository?.default_branch, process.env.RELEASE_TAG);
  if (typeof tag !== 'string' || !/^v\d+\.\d+\.\d+$/.test(tag)) throw new Error('Invalid release tag');
  const token = process.env.GITHUB_TOKEN;
  if (!token) throw new Error('The workflow read token is unavailable');
  async function get(path) {
    const response = await fetch(`https://api.github.com/repos/${repository}/${path}`, {
      headers: {Authorization: `Bearer ${token}`, Accept: 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2026-03-10'}, signal: AbortSignal.timeout(30000), redirect: 'error'});
    if (!response.ok) throw new Error(`Release gate API request failed (${response.status})`);
    return response.json();
  }
  const release = await get(`releases/tags/${encodeURIComponent(tag)}`);
  const commit = await get(`commits/${encodeURIComponent(tag)}`);
  if (!/^[a-f0-9]{40}$/.test(commit.sha)) throw new Error('Invalid commit returned by API');
  const runs = [];
  for (let page = 1; page <= 10; page++) {
    const result = await get(`actions/workflows/ci.yml/runs?head_sha=${commit.sha}&per_page=100&page=${page}`);
    if (!Array.isArray(result.workflow_runs) || result.total_count > 1000) throw new Error('CI run listing exceeds verification capacity');
    runs.push(...result.workflow_runs);
    if (result.workflow_runs.length < 100) break;
    if (page === 10) throw new Error('Incomplete CI run listing');
  }
  const result = selectRelease(release, commit, runs, repository, tag);
  if (process.env.EXPECTED_RELEASE_SHA && result.sha !== process.env.EXPECTED_RELEASE_SHA) {
    throw new Error('Release tag changed after the artifact was built');
  }
  if (process.env.GITHUB_OUTPUT) appendFileSync(process.env.GITHUB_OUTPUT, `sha=${result.sha}\ntag=${result.tag}\n`);
  console.log(JSON.stringify({status: 'passed', ...result}));
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  run().catch(error => { console.error(error.message); process.exitCode = 1; });
}
