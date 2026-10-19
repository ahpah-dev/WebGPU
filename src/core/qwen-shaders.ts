const reduce = `
var<workgroup> sums:array<f32,64>;
fn sum(lane:u32,x:f32)->f32{
  sums[lane]=x;workgroupBarrier();
  for(var s=32u;s>0u;s/=2u){if(lane<s){sums[lane]+=sums[lane+s];}workgroupBarrier();}
  let result=sums[0];workgroupBarrier();return result;
}`;

export const splitQueryShader = `
@group(0) @binding(0) var<storage,read> qg:array<f32>;
@group(0) @binding(1) var<storage,read_write> q:array<f32>;
@group(0) @binding(2) var<storage,read_write> gate:array<f32>;
struct P{n:u32,hd:u32,pad0:u32,pad1:u32}
@group(0) @binding(3) var<uniform> p:P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>){
  let i=g.x;if(i<p.n){let source=(i/p.hd)*p.hd*2u+i%p.hd;q[i]=qg[source];gate[i]=qg[source+p.hd];}
}`;

export const sigmoidGateShader = `
@group(0) @binding(0) var<storage,read> gate:array<f32>;
@group(0) @binding(1) var<storage,read_write> y:array<f32>;
struct P{n:u32,pad0:u32,pad1:u32,pad2:u32}
@group(0) @binding(2) var<uniform> p:P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>){if(g.x<p.n){y[g.x]/=1.0+exp(-gate[g.x]);}}
`;

export const siluShader = `
@group(0) @binding(0) var<storage,read> gate:array<f32>;
@group(0) @binding(1) var<storage,read> up:array<f32>;
@group(0) @binding(2) var<storage,read_write> y:array<f32>;
struct P{n:u32,pad0:u32,pad1:u32,pad2:u32}
@group(0) @binding(3) var<uniform> p:P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>){if(g.x<p.n){let x=gate[g.x];y[g.x]=x/(1.0+exp(-x))*up[g.x];}}
`;

/** Depthwise causal convolution. Each invocation owns one channel's history. */
export const convShader = `
@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(1) var<storage,read> weight:array<f32>;
@group(0) @binding(2) var<storage,read_write> history:array<f32>;
@group(0) @binding(3) var<storage,read_write> y:array<f32>;
struct P{channels:u32,kernel:u32,batch:u32,pad0:u32}
@group(0) @binding(4) var<uniform> p:P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>){
  let c=g.x;if(c>=p.channels){return;}
  let hbase=c*(p.kernel-1u);let wbase=c*p.kernel;
  for(var t=0u;t<p.batch;t++){
    let current=x[t*p.channels+c];var value=current*weight[wbase+p.kernel-1u];
    for(var j=0u;j+1u<p.kernel;j++){value+=history[hbase+j]*weight[wbase+j];}
    for(var j=0u;j+2u<p.kernel;j++){history[hbase+j]=history[hbase+j+1u];}
    if(p.kernel>1u){history[hbase+p.kernel-2u]=current;}
    y[t*p.channels+c]=value/(1.0+exp(-value));
  }
}`;

export const l2Shader = `
@group(0) @binding(0) var<storage,read_write> qkv:array<f32>;
struct P{channels:u32,hd:u32,epsilon:f32,pad0:u32}
@group(0) @binding(1) var<uniform> p:P;
${reduce}
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) g:vec3<u32>){
  let base=g.y*p.channels+g.x*p.hd;var square=0.0;
  for(var d=lane;d<p.hd;d+=64u){square+=qkv[base+d]*qkv[base+d];}
  let inv=inverseSqrt(sum(lane,square)+p.epsilon);
  for(var d=lane;d<p.hd;d+=64u){qkv[base+d]*=inv;}
}`;

export const decayShader = `
@group(0) @binding(0) var<storage,read> alpha:array<f32>;
@group(0) @binding(1) var<storage,read> beta:array<f32>;
@group(0) @binding(2) var<storage,read> a:array<f32>;
@group(0) @binding(3) var<storage,read> dt:array<f32>;
@group(0) @binding(4) var<storage,read_write> control:array<f32>;
struct P{n:u32,heads:u32,pad0:u32,pad1:u32}
@group(0) @binding(5) var<uniform> p:P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>){
  let i=g.x;if(i<p.n){let z=alpha[i]+dt[i%p.heads];let softplus=max(z,0.0)+log(1.0+exp(-abs(z)));
    control[2u*i]=exp(a[i%p.heads]*softplus);control[2u*i+1u]=1.0/(1.0+exp(-beta[i]));}
}`;

/** Each group owns one value column of a head's recurrent matrix, across all tokens.
 * GGUF Qwen35 conversion tiles V heads; K head = V head modulo key-head count.
 * State layout is [value-head, value-dimension, key-dimension]. */
export const deltaShader = `
@group(0) @binding(0) var<storage,read> qkv:array<f32>;
@group(0) @binding(1) var<storage,read> control:array<f32>;
@group(0) @binding(2) var<storage,read_write> state:array<f32>;
@group(0) @binding(3) var<storage,read_write> y:array<f32>;
struct P{hd:u32,keyHeads:u32,valueHeads:u32,batch:u32}
@group(0) @binding(4) var<uniform> p:P;
${reduce}
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) g:vec3<u32>){
  let h=g.x/p.hd;let v=g.x%p.hd;let kh=h%p.keyHeads;
  let keys=p.keyHeads*p.hd;let values=p.valueHeads*p.hd;let channels=2u*keys+values;
  let sb=g.x*p.hd;
  for(var t=0u;t<p.batch;t++){
    let qb=t*channels+kh*p.hd;let kb=qb+keys;
    let decay=control[2u*(t*p.valueHeads+h)];let beta=control[2u*(t*p.valueHeads+h)+1u];
    var memory=0.0;
    for(var d=lane;d<p.hd;d+=64u){state[sb+d]*=decay;memory+=state[sb+d]*qkv[kb+d];}
    let correction=(qkv[t*channels+2u*keys+h*p.hd+v]-sum(lane,memory))*beta;
    var output=0.0;
    for(var d=lane;d<p.hd;d+=64u){let updated=state[sb+d]+qkv[kb+d]*correction;state[sb+d]=updated;output+=updated*qkv[qb+d];}
    let result=sum(lane,output)*inverseSqrt(f32(p.hd));
    if(lane==0u){y[t*values+h*p.hd+v]=result;}
  }
}`;

export const normGateShader = `
@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(1) var<storage,read> weight:array<f32>;
@group(0) @binding(2) var<storage,read> gate:array<f32>;
@group(0) @binding(3) var<storage,read_write> y:array<f32>;
struct P{hd:u32,epsilon:f32,pad0:u32,pad1:u32}
@group(0) @binding(4) var<uniform> p:P;
${reduce}
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) g:vec3<u32>){
  let base=g.x*p.hd;var square=0.0;for(var d=lane;d<p.hd;d+=64u){square+=x[base+d]*x[base+d];}
  let inv=inverseSqrt(sum(lane,square)/f32(p.hd)+p.epsilon);
  for(var d=lane;d<p.hd;d+=64u){let z=gate[base+d];y[base+d]=x[base+d]*inv*weight[d]*z/(1.0+exp(-z));}
}`;

export const qwenKernels = { splitQuery: splitQueryShader, sigmoidGate: sigmoidGateShader, silu: siluShader, conv: convShader, l2: l2Shader, decay: decayShader, delta: deltaShader, normGate: normGateShader };
