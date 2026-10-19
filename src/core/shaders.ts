const reduction = `
var<workgroup> sums: array<f32,64>;
fn reduceSum(lane:u32, x:f32) -> f32 {
  sums[lane]=x; workgroupBarrier();
  for(var s=32u;s>0u;s=s/2u){ if(lane<s){sums[lane]+=sums[lane+s];} workgroupBarrier(); }
  return sums[0];
}`;
const bytes = `
fn byteAt(p:u32)->u32{return (w[p/4u]>>((p%4u)*8u))&255u;}
fn halfAt(p:u32)->f32{return unpack2x16float(byteAt(p)|(byteAt(p+1u)<<8u)).x;}
fn signedByte(p:u32)->f32{return f32(i32(byteAt(p)<<24u)>>24);}
`;
export function weightFunction(type: number): string {
  const code: Record<number, string> = {
    0: 'return bitcast<f32>(w[i]);',
    1: 'return halfAt(i*2u);',
    2: 'let p=(i/32u)*18u; let j=i%32u; return halfAt(p)*f32(i32((byteAt(p+2u+j%16u)>>((j/16u)*4u))&15u)-8);',
    8: 'let p=(i/32u)*34u; return halfAt(p)*signedByte(p+2u+i%32u);',
    12: `let p=(i/256u)*144u; let j=i%256u; let s=j/32u;
      var sc=0u;var mn=0u;
      if(s<4u){sc=byteAt(p+4u+s)&63u;mn=byteAt(p+8u+s)&63u;}
      else{sc=(byteAt(p+8u+s)&15u)|((byteAt(p+s)>>6u)<<4u);mn=(byteAt(p+8u+s)>>4u)|((byteAt(p+4u+s)>>6u)<<4u);}
      let q=(byteAt(p+16u+(j/64u)*32u+j%32u)>>((s%2u)*4u))&15u;
      return halfAt(p)*f32(sc*q)-halfAt(p+2u)*f32(mn);`,
    13: `let p=(i/256u)*176u; let j=i%256u; let s=j/32u;
      var sc=0u;var mn=0u;
      if(s<4u){sc=byteAt(p+4u+s)&63u;mn=byteAt(p+8u+s)&63u;}
      else{sc=(byteAt(p+8u+s)&15u)|((byteAt(p+s)>>6u)<<4u);mn=(byteAt(p+8u+s)>>4u)|((byteAt(p+4u+s)>>6u)<<4u);}
      let q=((byteAt(p+48u+(j/64u)*32u+j%32u)>>((s%2u)*4u))&15u)|(((byteAt(p+16u+j%32u)>>s)&1u)<<4u);
      return halfAt(p)*f32(sc*q)-halfAt(p+2u)*f32(mn);`,
    14: `let p=(i/256u)*210u;let j=i%256u;let segment=j/128u;let quarter=(j%128u)/32u;let lane=j%32u;
      let low=(byteAt(p+segment*64u+(quarter%2u)*32u+lane)>>((quarter/2u)*4u))&15u;
      let high=(byteAt(p+128u+segment*32u+lane)>>(quarter*2u))&3u;
      return halfAt(p+208u)*signedByte(p+192u+segment*8u+quarter*2u+lane/16u)*f32(i32(low|(high<<4u))-32);`,
  };
  if (!(type in code)) throw new Error(`No GPU kernel for tensor type ${type}.`);
  return `${bytes}\nfn weightAt(i:u32)->f32{${code[type]}}`;
}
export function matmulShader(type: number): string {
  return `
@group(0) @binding(0) var<storage,read> w:array<u32>;
@group(0) @binding(1) var<storage,read> x:array<f32>;
@group(0) @binding(2) var<storage,read_write> y:array<f32>;
struct P{cols:u32,rows:u32,stride:u32,rowBase:u32,inputOffset:u32,pad0:u32,pad1:u32,pad2:u32}
@group(0) @binding(3) var<uniform> p:P;
${weightFunction(type)} ${reduction}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) g:vec3<u32>){
  var sum=0.0;
  for(var c=lane;c<p.cols;c+=64u){sum+=weightAt(g.x*p.cols+c)*x[(g.y+p.inputOffset)*p.cols+c];}
  let result=reduceSum(lane,sum);
  if(lane==0u){y[g.y*p.stride+p.rowBase+g.x]=result;}
}`;
}
export const normShader = `
@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(1) var<storage,read> w:array<f32>;
@group(0) @binding(2) var<storage,read_write> y:array<f32>;
struct P{n:u32,epsilon:f32,pad0:u32,pad1:u32}
@group(0) @binding(3) var<uniform> p:P;
${reduction}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) g:vec3<u32>){
  let base=g.x*p.n; var sum=0.0;
  for(var i=lane;i<p.n;i+=64u){sum+=x[base+i]*x[base+i];}
  let inverse=inverseSqrt(reduceSum(lane,sum)/f32(p.n)+p.epsilon);
  for(var i=lane;i<p.n;i+=64u){y[base+i]=x[base+i]*inverse*w[i];}
}`;
export const ropeShader = `
@group(0) @binding(0) var<storage,read_write> x:array<f32>;
@group(0) @binding(1) var<storage,read> w:array<f32>;
struct P{heads:u32,hd:u32,start:u32,theta:f32,scale:f32,epsilon:f32,rotaryDim:u32,pad1:u32}
@group(0) @binding(2) var<uniform> p:P;
${reduction}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) g:vec3<u32>){
  let base=(g.y*p.heads+g.x)*p.hd;var sum=0.0;
  for(var i=lane;i<p.hd;i+=64u){sum+=x[base+i]*x[base+i];}
  let inverse=inverseSqrt(reduceSum(lane,sum)/f32(p.hd)+p.epsilon);
  let rot=select(p.rotaryDim,p.hd,p.rotaryDim==0u);
  for(var i=lane;i<rot/2u;i+=64u){
    let angle=f32(p.start+g.y)*p.scale/pow(p.theta,f32(2u*i)/f32(rot));
    let a=x[base+i]*w[i]*inverse;let b=x[base+i+rot/2u]*w[i+rot/2u]*inverse;
    x[base+i]=a*cos(angle)-b*sin(angle);x[base+i+rot/2u]=a*sin(angle)+b*cos(angle);
  }
  for(var i=rot+lane;i<p.hd;i+=64u){x[base+i]*=w[i]*inverse;}
}`;
export function attentionShader(context: number): string {
  return `
@group(0) @binding(0) var<storage,read> q:array<f32>;
@group(0) @binding(1) var<storage,read> k:array<f32>;
@group(0) @binding(2) var<storage,read> v:array<f32>;
@group(0) @binding(3) var<storage,read_write> y:array<f32>;
struct P{heads:u32,kvHeads:u32,hd:u32,start:u32,scale:f32,window:u32,pad0:u32,pad1:u32}
@group(0) @binding(4) var<uniform> p:P;
var<workgroup> scores:array<f32,${context}>;
${reduction}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) g:vec3<u32>){
  let end=p.start+g.y+1u;let begin=end-min(end,p.window);let head=g.x/(p.heads/p.kvHeads);
  let qb=(g.y*p.heads+g.x)*p.hd;let kvStride=p.kvHeads*p.hd;
  var maximum=-3.402823e38;
  for(var t=begin+lane;t<end;t+=64u){
    var dot=0.0;for(var d=0u;d<p.hd;d++){dot+=q[qb+d]*k[t*kvStride+head*p.hd+d];}
    scores[t]=dot*p.scale;maximum=max(maximum,scores[t]);
  }
  sums[lane]=maximum;workgroupBarrier();
  for(var s=32u;s>0u;s/=2u){if(lane<s){sums[lane]=max(sums[lane],sums[lane+s]);}workgroupBarrier();}
  let maxScore=sums[0];workgroupBarrier();var total=0.0;
  for(var t=begin+lane;t<end;t+=64u){scores[t]=exp(scores[t]-maxScore);total+=scores[t];}
  let divisor=reduceSum(lane,total);workgroupBarrier();
  for(var d=lane;d<p.hd;d+=64u){var sum=0.0;for(var t=begin;t<end;t++){sum+=scores[t]/divisor*v[t*kvStride+head*p.hd+d];}y[qb+d]=sum;}
}`;
}
export const addShader = `
@group(0) @binding(0) var<storage,read> delta:array<f32>;
@group(0) @binding(1) var<storage,read_write> state:array<f32>;
struct P{n:u32,pad0:u32,pad1:u32,pad2:u32}
@group(0) @binding(2) var<uniform> p:P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>){if(g.x<p.n){state[g.x]+=delta[g.x];}}`;
export const geluShader = `
@group(0) @binding(0) var<storage,read> gate:array<f32>;
@group(0) @binding(1) var<storage,read> up:array<f32>;
@group(0) @binding(2) var<storage,read_write> y:array<f32>;
struct P{n:u32,pad0:u32,pad1:u32,pad2:u32}
@group(0) @binding(3) var<uniform> p:P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>){
  if(g.x<p.n){let x=gate[g.x];y[g.x]=0.5*x*(1.0+tanh(0.7978845608*(x+0.044715*x*x*x)))*up[g.x];}
}`;
