/** Validate recorded evidence without loading models or executing controllers. */
import {readFileSync, statSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {resolve} from 'node:path';
export const MAX_BYTES = 32 * 2 ** 20;
export function sourceFilesFromRuntime(text) {
  const match = text.match(/BEHAVIOR_SOURCE_FILES\s*=\s*tuple\(sorted\(\(([\s\S]*?)\)\)\)/);
  if (!match) throw new Error('Missing authoritative behavior source list');
  const files = [];
  const remainder = match[1].replace(/"([a-z_]+\.py)"/g, (_, name) => { files.push(name); return ''; }).replace(/[\s,]/g, '');
  if (remainder || !files.includes('runtime.py') || files.length > 32 || new Set(files).size !== files.length) {
    throw new Error('Invalid authoritative behavior source list');
  }
  return files.sort();
}
export const SOURCE_FILES = sourceFilesFromRuntime(readFileSync(new URL('../src/neuroterrarium/runtime.py',import.meta.url),'utf8'));
const kinds = ['connectome','feedforward','recurrent','hybrid'];
const groups = ['sugar','mn9','lplc2','lc4','gf','dna01','dna02'];
export const sha256 = data => createHash('sha256').update(data).digest('hex');
const fail = message => { throw new Error(message); };
export function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.keys(value).sort().map(k => `${JSON.stringify(k)}:${canonical(value[k])}`).join(',')}}`;
  return JSON.stringify(value);
}
const same = (a,b) => canonical(a) === canonical(b);
function object(v,label) { if (!v || typeof v !== 'object' || Array.isArray(v)) fail(`Invalid ${label} object`); }
function keys(v,expected,label) { object(v,label); if (!same(Object.keys(v).sort(),[...expected].sort())) fail(`Invalid ${label} fields`); }
function number(v,lo,hi,label) { if (typeof v !== 'number' || !Number.isFinite(v) || v<lo || v>hi) fail(`Invalid ${label} number`); }
function integer(v,lo,hi,label) { number(v,lo,hi,label); if (!Number.isSafeInteger(v)) fail(`Invalid ${label} integer`); }
function flag(v,label) { if (typeof v !== 'boolean') fail(`Invalid ${label} flag`); }
function array(v,lo,hi,label) { if (!Array.isArray(v) || v.length<lo || v.length>hi) fail(`Invalid ${label} length`); }
function hash(v,label) { if (typeof v !== 'string' || !/^[a-f0-9]{64}$/.test(v)) fail(`Invalid ${label} checksum`); }
function boundedStructure(value) {
  const stack=[[value,0,'']]; let count=0;
  while (stack.length) {
    const [item,depth,key]=stack.pop();
    if (++count>5e6 || depth>32) fail('Recording structure exceeds capacity');
    if (key==='root_id' && (typeof item!=='string' || !/^[1-9][0-9]{14,19}$/.test(item))) fail('Root ID must be a decimal string');
    if (key==='root_ids' && (!Array.isArray(item) || item.some(id => typeof id!=='string' || !/^[1-9][0-9]{14,19}$/.test(id)))) fail('Root IDs must be decimal strings');
    if (typeof item==='number') number(item,-Number.MAX_SAFE_INTEGER,Number.MAX_SAFE_INTEGER,'recorded value');
    else if (typeof item==='string' && item.length>4096) fail('Recorded string exceeds capacity');
    else if (item && typeof item==='object') for (const [name,child] of Object.entries(item)) {
      if (['__proto__','prototype','constructor'].includes(name)) fail('Unsupported recorded key');
      stack.push([child,depth+1,name]);
    }
  }
}
function action(v,label='action') { array(v,4,4,label); v.forEach((x,i)=>number(x,i===1?-1:0,1,label)); }
function worldObjects(world,time) {
  keys(world,['width','height','bodies','foods','obstacles','stimuli'],'world');
  if (world.width!==80 || world.height!==56) fail('Invalid world dimensions');
  array(world.bodies,10,10,'bodies');
  for (const b of world.bodies) {
    keys(b,['x','y','heading','speed','energy','food','collisions','action'],'body');
    number(b.x,0.55,79.45,'body x'); number(b.y,0.55,55.45,'body y');
    number(b.heading,-Math.PI-1e-12,Math.PI+1e-12,'heading');
    number(b.speed,0,20.000001,'speed'); number(b.energy,0,1,'energy');
    number(b.food,0,10000,'food'); integer(b.collisions,0,1e9,'collisions'); action(b.action);
  }
  for (const [name,limit,fields] of [['foods',256,['x','y','amount','radius']],['obstacles',256,['x','y','radius']],['stimuli',32,['x','y','radius','vx','vy','growth','physical']]]) {
    array(world[name],0,limit,name);
    for (const item of world[name]) {
      keys(item,fields,name);
      if (name==='stimuli') {
        for (const key of ['vx','vy','growth']) number(item[key],-10000,10000,key);
        flag(item.physical,'physical stimulus');
        for (const [key,rate,initial] of [['x',Math.abs(item.vx),10000],['y',Math.abs(item.vy),10000],['radius',Math.max(0,item.growth),100]]) {
          const bound=initial+rate*time, allowance=(2*Math.round(time/0.02)+8)*Number.EPSILON*bound;
          number(item[key],key==='radius'?Number.MIN_VALUE:-bound-allowance,bound+allowance,key);
        }
      } else {
        number(item.x,-10000,10000,`${name} x`); number(item.y,-10000,10000,`${name} y`);
        number(item.radius,Number.MIN_VALUE,100,`${name} radius`);
        if (name==='foods') number(item.amount,0,10000,'food amount');
      }
    }
  }
}
function panel(p,world,time) {
  object(p,'inspector'); integer(p.index,0,9,'inspector index');
  keys(p,['index','label','kind','body','action','observation','observation_time',
    ...(p.kind==='connectome'?['neural_rates_hz','neural_inputs_hz']:['hidden','policy_decision']),
    ...('reflex_enabled' in p?['reflex_enabled']:[])],'inspector');
  if ('reflex_enabled' in p) {
    if (p.kind==='connectome') fail('Connectome inspector has no engineered policy reflex');
    flag(p.reflex_enabled,'policy reflex enabled');
  }
  if (!kinds.includes(p.kind) || typeof p.label!=='string' || !p.label.length || p.label.length>64) fail('Invalid inspector identity');
  if (!same(p.body,world.bodies[p.index])) fail('Inspector body does not match selected individual');
  action(p.action); if (!same(p.action,p.body.action)) fail('Inspector action does not match body action');
  array(p.observation,59,59,'observation');
  p.observation.forEach((v,i)=>number(v,(i>=32&&i<48)||i===56?-1:0,1,'observation'));
  number(p.observation_time,0,time,'observation time');
  if (Math.abs(p.observation_time-Math.max(0,time-0.02))>1e-8) fail('Observation/action clock mismatch');
  if (p.kind==='connectome') {
    for (const [key,allowed] of [['neural_rates_hz',groups],['neural_inputs_hz',['sugar','lplc2','lc4','dna02']]]) {
      object(p[key],key);
      if (Object.keys(p[key]).some(name=>!allowed.includes(name))) fail('Unregistered neural group');
      if (time>0 && !Object.keys(p[key]).length) fail('Missing recorded neural activity');
      for (const rate of Object.values(p[key])) number(rate,0,10000,'neural rate');
    }
    if ('policy_decision' in p || 'hidden' in p) fail('Incorrect connectome inspector fields');
  } else {
    const size=p.kind==='feedforward'?0:64;
    array(p.hidden,size,size,'policy hidden state'); for (const v of p.hidden) number(v,-1,1,'hidden state');
    object(p.policy_decision,'policy decision');
    if (time>0 || Object.keys(p.policy_decision).length) {
      const d=p.policy_decision;
      keys(d,['raw_action','log_probability','value','applied_action','base_action','reflex_triggered','correction_l1'],'policy decision');
      array(d.raw_action,4,4,'raw policy sample'); for (const raw of d.raw_action) number(raw,-1e6,1e6,'raw policy sample');
      for (const key of ['log_probability','value']) number(d[key],-1e9,1e9,key);
      action(d.applied_action); action(d.base_action); flag(d.reflex_triggered,'reflex'); number(d.correction_l1,0,5,'reflex correction');
      if (!same(d.applied_action,p.action)) fail('Recorded policy applied action mismatch');
      const correction=d.applied_action.reduce((sum,v,i)=>sum+Math.abs(v-d.base_action[i]),0);
      if (Math.abs(correction-d.correction_l1)>2e-6) fail('Recorded reflex correction mismatch');
      if (p.kind!=='hybrid' && (d.reflex_triggered || d.correction_l1!==0)) fail('Unexpected policy reflex');
    }
    if ('neural_rates_hz' in p || 'neural_inputs_hz' in p) fail('Incorrect learned-controller inspector fields');
  }
}
function profile(value,data) {
  object(value,'full graph profile');
  if (value.schema!=='neuroterrarium.graph-summary.v1' || value.profile!==data.profile || value.connectome_version!==data.connectome_version) fail('Recorded graph profile mismatch');
  for (const [key,count] of Object.entries(data.expected)) if (value[key]!==count) fail(`Recorded graph ${key} mismatch`);
  if (!same(value.source_sha256,Object.fromEntries(Object.entries(data.sources).map(([key,item])=>[key,item.sha256])))) fail('Recorded graph data checksum mismatch');
  integer(value.positive_pairs,0,value.directed_pairs,'positive pairs'); integer(value.negative_pairs,0,value.directed_pairs,'negative pairs');
  if (value.positive_pairs+value.negative_pairs!==value.directed_pairs || value.duplicate_pairs!==0) fail('Invalid graph edge accounting');
}
function frame(value,data,depth=0) {
  if (depth>1) fail('Nested recording forks are unsupported'); object(value,'frame');
  keys(value,['mode','ready','paused','simulation_time','wall_time','rate','compute_seconds','speed','scenario',
    'noise_std','readout_clamped','channels','world','panels','selected','events',
    ...('profile' in value?['profile']:[]),...('fork' in value?['fork']:[])],'frame');
  if (value.mode!=='Local full graph' || value.ready!==true || 'error' in value) fail('Recording does not contain a ready full-graph session');
  for (const key of ['simulation_time','wall_time','rate','compute_seconds']) number(value[key],0,2e7,key);
  if (Math.abs(value.simulation_time/0.02-Math.round(value.simulation_time/0.02))>1e-7) fail('Simulation time is off the fixed action clock');
  const rate=value.wall_time>0?value.simulation_time/value.wall_time:0;
  if (Math.abs(rate-value.rate)>1e-9*Math.max(1,rate)) fail('Recorded simulation/wall-time ratio mismatch');
  number(value.speed,0.05,4,'requested speed'); number(value.noise_std,0,0.2,'observation noise');
  flag(value.paused,'pause'); flag(value.readout_clamped,'readout clamp');
  keys(value.channels,['chemical','proximity','taste','vision'],'shared channels'); for (const v of Object.values(value.channels)) flag(v,'channel');
  if (!['open','occluded','looming'].includes(value.scenario)) fail('Unknown recorded scenario');
  worldObjects(value.world,value.simulation_time); if (depth===0 || 'profile' in value) profile(value.profile,data);
  array(value.panels,10,10,'inspectors');
  value.panels.forEach((p,i)=>{ panel(p,value.world,value.simulation_time); if (p.index!==i) fail('Inspector ordering mismatch'); });
  panel(value.selected,value.world,value.simulation_time);
  if (!same(value.selected,value.panels[value.selected.index])) fail('Selected inspector mismatch');
  if (depth===0 && !same(kinds.map(kind=>value.panels.filter(p=>p.kind===kind).length),[1,3,3,3])) fail('Recorded default controller allocation mismatch');
  array(value.events,0,4096,'events');
  for (const event of value.events) {
    object(event,'event');
    if (typeof event.type!=='string' || !event.type.length || event.type.length>64) fail('Invalid event type');
    number(event.time,0,value.simulation_time,'event time');
  }
  if ('fork' in value) { frame(value.fork,data,depth+1); if (value.fork.simulation_time!==value.simulation_time) fail('Forks do not share the same simulation clock'); }
}
export function validateRecording(recording,manifest,expected) {
  boundedStructure(recording); boundedStructure(manifest);
  keys(recording,['schema','mode','verification','frames'],'recording');
  if (manifest.schema!=='neuroterrarium.replay-manifest.v1') fail('Missing versioned replay manifest');
  for (const key of ['sha256','source_sha256','data_manifest_sha256','interface_sha256','interface_registry_sha256','models_manifest_sha256','model_set_sha256']) hash(manifest[key],key);
  keys(manifest.source_files,SOURCE_FILES,'source files');
  if (!same(manifest.source_files,expected.source_files) || manifest.source_sha256!==sha256(canonical(expected.source_files))) fail('Recording behavior source mismatch');
  for (const key of ['data_manifest_sha256','interface_sha256','interface_registry_sha256','models_manifest_sha256','model_set_sha256']) if (manifest[key]!==expected[key]) fail(`Recording provenance mismatch: ${key}`);
  if (recording.schema!=='neuroterrarium.frame-replay.v1' || recording.mode!=='Replay' || recording.verification!=='recorded event/action playback; no controller recomputation') fail('Expected explicitly labeled frame replay');
  array(recording.frames,2,3000,'recording frames'); if (manifest.frames!==recording.frames.length) fail('Recording frame count mismatch');
  let moved=false, previous;
  for (const current of recording.frames) {
    frame(current,expected.data);
    if (previous && (current.simulation_time<previous.simulation_time || current.wall_time<previous.wall_time || current.compute_seconds<previous.compute_seconds)) fail('Representative recording clock regressed');
    if (previous && !same(current.panels.map(p=>p.kind),previous.panels.map(p=>p.kind))) fail('Default controller identities changed during recording');
    if (previous && !same(current.profile,previous.profile)) fail('Graph profile changed during recording');
    current.world.bodies.forEach((body,i)=>{ const first=recording.frames[0].world.bodies[i]; moved ||= body.x!==first.x || body.y!==first.y; });
    previous=current;
  }
  if (!moved) fail('Recording contains no body motion');
  return {status:'passed',mode:'Replay',frames:recording.frames.length,sha256:manifest.sha256};
}
export function validateBytes(raw,manifest,expected) {
  if (raw.length>MAX_BYTES || manifest.bytes!==raw.length) fail('Replay byte length mismatch or capacity exceeded');
  if (sha256(raw)!==manifest.sha256) fail('Replay checksum mismatch');
  return validateRecording(JSON.parse(raw.toString('utf8')),manifest,expected);
}
function boundedRead(path,limit=MAX_BYTES) { if (statSync(path).size>limit) fail('Asset exceeds byte capacity'); return readFileSync(path); }
export function loadExpected(root) {
  const read=path=>boundedRead(resolve(root,path));
  const dataBytes=read('configs/data-v783.json'), modelsBytes=read('artifacts/models/manifest.json'), models=JSON.parse(modelsBytes);
  if (models.schema!=='neuroterrarium.model-set.v1' || models.status!=='completed' || models.models?.length!==9) fail('Completed nine-model manifest missing');
  if (new Set(models.models.map(m=>m.policy_sha256)).size!==9) fail('Model weights are not distinct');
  for (const architecture of kinds.slice(1)) { const group=models.models.filter(m=>m.architecture===architecture); if (group.length!==3 || new Set(group.map(m=>m.seed)).size!==3) fail('Expected three independent seeds for each policy architecture'); }
  for (const m of models.models) { if (m.training_status!=='completed' || !Number.isSafeInteger(m.transitions) || m.transitions<=0) fail('Incomplete trained model'); hash(m.policy_sha256,'policy'); hash(m.checkpoint_manifest_sha256,'checkpoint'); }
  return {data:JSON.parse(dataBytes),data_manifest_sha256:sha256(dataBytes),
    source_files:Object.fromEntries(SOURCE_FILES.map(name=>[name,sha256(read(`src/neuroterrarium/${name}`))])),
    interface_sha256:sha256(read('configs/brain-interface.json')),interface_registry_sha256:sha256(read('configs/interface-v783.json')),
    models_manifest_sha256:sha256(modelsBytes),model_set_sha256:sha256(canonical(models))};
}
if (process.argv[1] && import.meta.url===pathToFileURL(process.argv[1]).href) {
  try {
    const root=fileURLToPath(new URL('../',import.meta.url));
    const raw=boundedRead(resolve(root,'web/dist/representative-replay.json'));
    const manifest=JSON.parse(boundedRead(resolve(root,'web/dist/replay-manifest.json'),1024*1024));
    console.log(JSON.stringify(validateBytes(raw,manifest,loadExpected(root))));
  } catch (error) { console.error(error.message); process.exitCode=1; }
}
