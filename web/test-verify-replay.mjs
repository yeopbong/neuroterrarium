import test from 'node:test';
import assert from 'node:assert/strict';
import {MAX_BYTES,SOURCE_FILES,canonical,sha256,sourceFilesFromRuntime,validateBytes,validateRecording} from './verify-replay.mjs';

// Synthetic schema fixtures are confined to tests; they are never distributed as recordings.
function fixture() {
  const digest='a'.repeat(64);
  const expected={source_files:Object.fromEntries(SOURCE_FILES.map(name=>[name,digest])),
    data_manifest_sha256:digest,interface_sha256:digest,interface_registry_sha256:digest,
    models_manifest_sha256:digest,model_set_sha256:digest,
    data:{profile:'test-schema-full',connectome_version:'783',expected:{neurons:10,connection_rows:9,directed_pairs:9,synaptic_contacts:12},sources:{test:{sha256:digest}}}};
  const frames=[0,0.02].map(time=>{
    const bodies=Array.from({length:10},(_,i)=>({x:8+i*6+time,y:25,heading:0,speed:0,energy:1,food:0,collisions:0,action:[0,0,0,0]}));
    const panels=bodies.map((body,index)=>{
      const kind=index===0?'connectome':index<=3?'feedforward':index<=6?'recurrent':'hybrid';
      const p={index,label:`Fly ${index}`,kind,body,action:body.action,observation:Array(59).fill(0),observation_time:0};
      if (kind==='connectome') Object.assign(p,{neural_rates_hz:{gf:0,mn9:0},neural_inputs_hz:{sugar:0,dna02:15}});
      else Object.assign(p,{hidden:kind==='feedforward'?[]:Array(64).fill(0),policy_decision:time===0?{}:{raw_action:[0,0,0,0],log_probability:-2,value:0,applied_action:[0,0,0,0],base_action:[0,0,0,0],reflex_triggered:false,correction_l1:0}});
      return p;
    });
    return {mode:'Local full graph',ready:true,paused:false,simulation_time:time,wall_time:1+time,rate:time/(1+time),compute_seconds:time,
      speed:1,scenario:'open',noise_std:0,readout_clamped:false,channels:{chemical:true,proximity:true,taste:true,vision:true},
      world:{width:80,height:56,bodies,foods:[{x:20,y:16,amount:1,radius:1.2}],obstacles:[],stimuli:[]},selected:panels[0],panels,events:[],
      profile:{schema:'neuroterrarium.graph-summary.v1',profile:'test-schema-full',connectome_version:'783',...expected.data.expected,
        positive_pairs:7,negative_pairs:2,duplicate_pairs:0,source_sha256:{test:digest}}};
  });
  const recording={schema:'neuroterrarium.frame-replay.v1',mode:'Replay',verification:'recorded event/action playback; no controller recomputation',frames};
  const raw=Buffer.from(JSON.stringify(recording));
  const manifest={schema:'neuroterrarium.replay-manifest.v1',sha256:sha256(raw),bytes:raw.length,frames:frames.length,
    source_sha256:sha256(canonical(expected.source_files)),...expected};
  delete manifest.data;
  return {recording,manifest,expected,raw};
}
function rejects(mutate,pattern) {
  const {recording,manifest,expected}=fixture(); mutate(recording,manifest,expected);
  assert.throws(()=>validateRecording(recording,manifest,expected),pattern);
}

test('bounded schema fixture validates without executing any controller',()=>{
  const f=fixture(); assert.equal(validateBytes(f.raw,f.manifest,f.expected).status,'passed');
});
test('behavior source list is read from the runtime constant without evaluating code',()=>{
  assert.deepEqual(sourceFilesFromRuntime('BEHAVIOR_SOURCE_FILES = tuple(sorted(("world.py", "runtime.py")))'),['runtime.py','world.py']);
  assert.throws(()=>sourceFilesFromRuntime('BEHAVIOR_SOURCE_FILES = unknown()'));
  assert.throws(()=>sourceFilesFromRuntime('BEHAVIOR_SOURCE_FILES = tuple(sorted(("runtime.py", "runtime.py")))'));
  assert.throws(()=>sourceFilesFromRuntime('BEHAVIOR_SOURCE_FILES = tuple(sorted(("runtime.py", function())))'));
  assert.ok(SOURCE_FILES.includes('brain_cache.py'));
});
test('checksum, size, frame capacity and explicit Replay labeling are required',()=>{
  const f=fixture();
  assert.throws(()=>validateBytes(f.raw,{...f.manifest,sha256:'b'.repeat(64)},f.expected),/checksum/);
  assert.throws(()=>validateBytes(f.raw,{...f.manifest,bytes:1},f.expected),/length/);
  assert.throws(()=>validateBytes({length:MAX_BYTES+1},f.manifest,f.expected),/capacity/);
  rejects(r=>r.frames=Array(3001).fill({}),/length/);
  rejects(r=>r.mode='Live',/replay/i);
  rejects(r=>r.verification='controller recomputation',/replay/i);
  rejects((r,m)=>m.frames=3,/frame count/);
});
test('body, observations, hidden state, actions and all nested values must be finite and bounded',()=>{
  for (const mutation of [r=>r.frames[1].world.bodies[0].x=Infinity,
    r=>r.frames[1].world.bodies[0].energy=1.01,
    r=>r.frames[1].world.bodies[0].collisions=0.5,
    r=>r.frames[1].panels[2].observation[0]=-0.1,
    r=>r.frames[1].panels[2].observation[59]=0,
    r=>r.frames[1].panels[4].hidden[0]=NaN,
    r=>r.frames[1].panels[4].hidden.push(0),
    r=>r.frames[1].panels[3].action[0]=2,
    r=>r.frames[1].panels[1].policy_decision.log_probability=Infinity]) rejects(mutation);
});
test('inspector selection, body, action and policy sample semantics agree',()=>{
  rejects(r=>r.frames[1].panels[2].index=3,/Inspector body/);
  rejects(r=>r.frames[1].panels.reverse(),/ordering/);
  rejects(r=>r.frames[1].selected={...r.frames[1].selected,label:'wrong selection'},/Selected inspector/);
  rejects(r=>r.frames[1].panels[1].body={...r.frames[1].panels[1].body,x:42},/Inspector body/);
  rejects(r=>r.frames[1].panels[1].policy_decision.applied_action[0]=0.2,/applied action/);
  rejects(r=>r.frames[1].panels[7].policy_decision.correction_l1=1,/correction/);
  rejects(r=>r.frames[1].panels[1].policy_decision.reflex_triggered=true,/Unexpected policy reflex/);
  rejects(r=>r.frames[1].panels[7].reflex_enabled='false',/flag/);
  const f=fixture(); f.recording.frames[1].panels[7].reflex_enabled=false;
  assert.equal(validateRecording(f.recording,f.manifest,f.expected).status,'passed');
});
test('clock, events, full profile and neural groups are checked',()=>{
  rejects(r=>r.frames[1].simulation_time=0.021,/fixed action clock/);
  rejects(r=>r.frames[1].rate=10,/ratio/);
  rejects(r=>r.frames[1].panels[0].observation_time=0.01,/clock mismatch/);
  rejects(r=>r.frames.reverse(),/regressed/);
  rejects(r=>r.frames[1].events.push({type:'food_add',time:2}),/event time/);
  rejects(r=>r.frames[1].profile.neurons=11,/neurons/);
  rejects(r=>r.frames[1].profile.negative_pairs=1,/edge accounting/);
  rejects(r=>r.frames[1].profile.source_sha256.test='b'.repeat(64),/data checksum/);
  rejects(r=>r.frames[1].panels[0].neural_rates_hz.fear=1,/Unregistered/);
  rejects(r=>r.frames[1].mode='Derived circuit',/full-graph/);
  rejects(r=>r.frames[1].ready=false,/full-graph/);
});
test('object schemas, capacity, flags and accumulated stimulus domain are checked',()=>{
  rejects(r=>r.frames[1].world.foods[0].amount=-1,/food amount/);
  rejects(r=>r.frames[1].world.obstacles=Array(257).fill({}),/length/);
  rejects(r=>r.frames[1].world.foods[0].script='javascript:alert(1)',/fields/);
  rejects(r=>r.frames[1].channels.vision=1,/flag/);
  rejects(r=>r.frames[1].world.stimuli=[{x:1e9,y:0,radius:1,vx:1,vy:0,growth:0,physical:false}],/number/);
});
test('forks are bounded, clock matched and selected consistently',()=>{
  const f=fixture(); f.recording.frames[1].fork=structuredClone(f.recording.frames[1]);
  assert.equal(validateRecording(f.recording,f.manifest,f.expected).status,'passed');
  f.recording.frames[1].fork.fork=structuredClone(f.recording.frames[0]);
  assert.throws(()=>validateRecording(f.recording,f.manifest,f.expected),/Nested/);
  rejects(r=>r.frames[1].fork=structuredClone(r.frames[0]),/same simulation clock/);
});
test('wrong source, data, registry and model provenance cannot validate',()=>{
  for (const key of ['data_manifest_sha256','interface_sha256','interface_registry_sha256','models_manifest_sha256','model_set_sha256'])
    rejects((r,m)=>m[key]='b'.repeat(64),/provenance/);
  rejects((r,m)=>m.source_files['world.py']='b'.repeat(64),/behavior source/);
  rejects((r,m)=>m.source_sha256='b'.repeat(64),/behavior source/);
});
test('root ID precision, deep payloads and motion-free content are rejected',()=>{
  rejects(r=>r.frames[1].events.push({type:'input',time:0,root_id:720575940622838154}),/Root ID/);
  rejects(r=>r.frames[1].events.push({type:'input',time:0,root_ids:['720575940622838154',720575940622838154]}),/Root ID/);
  rejects(r=>{let v=r; for(let i=0;i<40;i++){v.extra={};v=v.extra;}},/capacity/);
  rejects(r=>{r.frames[1]=structuredClone(r.frames[0]);},/no body motion/);
});
