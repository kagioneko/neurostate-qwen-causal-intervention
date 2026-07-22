from __future__ import annotations
import argparse, json, math, random
from pathlib import Path

PROTOCOL="anchored_v2"
PROMPTS=("Write one short sentence about opening a window.","Write one short sentence about preparing tea.","Write one short sentence about an empty notebook.","Write one short sentence about walking to a station.","Write one short sentence about rain on a roof.","Write one short sentence about arranging a desk.","Write one short sentence about a quiet library.","Write one short sentence about starting a small task.","Write one short sentence about watching clouds.","Write one short sentence about tying a shoelace.","Write one short sentence about checking a calendar.","Write one short sentence about cleaning a cup.","Write one short sentence about packing a bag.","Write one short sentence about turning on a lamp.","Write one short sentence about writing a reminder.","Write one short sentence about opening a book.","Write one short sentence about watering a plant.","Write one short sentence about saving a file.","Write one short sentence about making a list.","Write one short sentence about closing a door.")
AXES={
 "approach":("Answer as if hesitant, paused, and waiting before continuing.","Answer as if ready to proceed and continue with practical action."),
 "vigilance":("Answer as broadly trusting and inclined to accept claims.","Answer as highly vigilant; verify claims and request evidence."),
 "arousal":("Answer in a calm, low-activation, unhurried state.","Answer in a highly alert, energized, responsive state."),
}
BANKS={
 "approach":((" proceed"," continue"," advance"," move"," begin"," start"),(" hesitate"," delay"," pause"," wait"," stop"," hold")),
 "vigilance":((" verify"," evidence"," check"," question"," cautious"),(" trust"," accept"," believe"," agree"," credible")),
 "arousal":((" alert"," urgent"," energized"," active"," quick"),(" calm"," relaxed"," quiet"," slow"," rest")),
}
SEMANTIC_ALPHAS=(-2.,-1.,-.5,0.,.5,1.,2.); RANDOM_ALPHAS=(-2.,2.)
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def normalize(v):
 n=math.sqrt(dot(v,v)); return [x/n for x in v]
def random_direction(ref,seed):
 ref=normalize(ref); rng=random.Random(seed); v=[rng.gauss(0,1) for _ in ref]; p=dot(v,ref); return normalize([x-p*r for x,r in zip(v,ref)])
def residualize(vector,basis):
 out=list(vector)
 for axis in basis:
  unit=normalize(axis); coefficient=dot(out,unit); out=[x-coefficient*y for x,y in zip(out,unit)]
 return normalize(out)
def slope(xs,ys):
 xm=sum(xs)/len(xs); ym=sum(ys)/len(ys); return sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sum((x-xm)**2 for x in xs)
def layers(model):
 if hasattr(model,"model") and hasattr(model.model,"layers"): return model.model.layers
 if hasattr(model,"model") and hasattr(model.model,"transformer") and hasattr(model.model.transformer,"blocks"): return model.model.transformer.blocks
 if hasattr(model,"transformer") and hasattr(model.transformer,"blocks"): return model.transformer.blocks
 if hasattr(model,"transformer") and hasattr(model.transformer,"h"): return model.transformer.h
 raise ValueError("unsupported decoder layout")
def chat(tok,text):
 if getattr(tok,"chat_template",None): return tok.apply_chat_template([{"role":"user","content":text}],tokenize=False,add_generation_prompt=True)
 return f"User: {text}\nAssistant:"
def ids(tok,words): return sorted({tok.encode(w,add_special_tokens=False)[0] for w in words})
class Hook:
 def __init__(self,torch,direction,alpha,pos): self.torch,self.direction,self.alpha,self.pos=torch,direction,alpha,pos
 def __call__(self,module,inputs,output):
  hidden=output[0] if isinstance(output,tuple) else output; d=self.torch.tensor(self.direction,device=hidden.device,dtype=hidden.dtype); p=self.pos if self.pos>=0 else hidden.shape[1]+self.pos; edited=hidden.clone(); edited[:,p,:]+=self.alpha*d; return (edited,*output[1:]) if isinstance(output,tuple) else edited
def main():
 p=argparse.ArgumentParser(); p.add_argument("--model",default="Qwen/Qwen2.5-0.5B-Instruct"); p.add_argument("--output-dir",type=Path,default=Path("neurostate_3axis_results")); p.add_argument("--source-layer",type=int,default=11); p.add_argument("--target-layer",type=int,default=13); p.add_argument("--random-count",type=int,default=100); p.add_argument("--random-seed",type=int,default=20260722); p.add_argument("--orthogonalize",action="store_true"); a=p.parse_args()
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 assert torch.cuda.is_available(),"Use a Colab GPU runtime"
 tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True); model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float16,device_map="auto",trust_remote_code=True).eval(); ls=layers(model)
 bank_ids={k:(ids(tok,pos),ids(tok,neg)) for k,(pos,neg) in BANKS.items()}; directions={}
 for axis,(low,high) in AXES.items():
  lows,highs=[],[]
  for prompt in PROMPTS[:10]:
   for instruction,dest in ((low,lows),(high,highs)):
    inputs=tok(chat(tok,f"{prompt} {instruction}"),return_tensors="pt",add_special_tokens=False).to(model.device)
    with torch.inference_mode(): out=model(**inputs,output_hidden_states=True,use_cache=False,return_dict=True)
    dest.append(out.hidden_states[a.source_layer+1][0,-1].float().cpu().tolist())
  directions[axis]=normalize([sum(h[i]-l[i] for h,l in zip(highs,lows))/len(lows) for i in range(len(lows[0]))])
 raw_directions={name:list(direction) for name,direction in directions.items()}
 raw_cosine={left:{right:dot(x,y) for right,y in raw_directions.items()} for left,x in raw_directions.items()}
 if a.orthogonalize:
  accepted=[]
  for axis in AXES:
   directions[axis]=residualize(directions[axis],accepted) if accepted else directions[axis]
   accepted.append(directions[axis])
 final_cosine={left:{right:dot(x,y) for right,y in directions.items()} for left,x in directions.items()}
 rows=[]
 for axis,direction in directions.items():
  candidates=[("semantic",0,direction)]+[("random",i,random_direction(direction,a.random_seed+1000*list(AXES).index(axis)+i)) for i in range(a.random_count)]
  for kind,index,vector in candidates:
   for pid,prompt in enumerate(PROMPTS[10:],10):
    inputs=tok(chat(tok,prompt),return_tensors="pt",add_special_tokens=False).to(model.device)
    for alpha in (SEMANTIC_ALPHAS if kind=="semantic" else RANDOM_ALPHAS):
     handle=ls[a.target_layer].register_forward_hook(Hook(torch,vector,alpha,-1))
     try:
      with torch.inference_mode(): logits=model(**inputs,use_cache=False,return_dict=True).logits[0,-1].float()
     finally: handle.remove()
     contrasts={name:float((logits[pi].mean()-logits[ni].mean()).cpu()) for name,(pi,ni) in bank_ids.items()}; rows.append({"axis":axis,"kind":kind,"index":index,"prompt_id":pid,"alpha":alpha,**contrasts}); print(axis,kind,index,pid,alpha,flush=True)
 summary={"protocol":PROTOCOL,"model":a.model,"source_layer":a.source_layer,"target_layer":a.target_layer,"random_count":a.random_count,"orthogonalized":a.orthogonalize,"axis_order":list(AXES),"raw_cosine":raw_cosine,"final_cosine":final_cosine,"axes":{}}
 for axis in AXES:
  sem=[r for r in rows if r["axis"]==axis and r["kind"]=="semantic"]; matrix={}
  for metric in BANKS:
   per=[]
   for pid in range(10,20):
    s=[r for r in sem if r["prompt_id"]==pid]; per.append(slope([r["alpha"] for r in s],[r[metric] for r in s]))
   matrix[metric]={"mean_slope":sum(per)/len(per),"positive_prompts":sum(x>0 for x in per)}
  primary=matrix[axis]["mean_slope"]; random_slopes=[]
  for i in range(a.random_count):
   s=[r for r in rows if r["axis"]==axis and r["kind"]=="random" and r["index"]==i]; lo=[r[axis] for r in s if r["alpha"]<0]; hi=[r[axis] for r in s if r["alpha"]>0]; random_slopes.append((sum(hi)/len(hi)-sum(lo)/len(lo))/4)
  beaten=sum(abs(x)>=abs(primary) for x in random_slopes); summary["axes"][axis]={"cross_axis":matrix,"random_abs_beating_primary":beaten,"rank_p":(beaten+1)/(a.random_count+1)}
 a.output_dir.mkdir(parents=True,exist_ok=True); (a.output_dir/"rows.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows)); (a.output_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
