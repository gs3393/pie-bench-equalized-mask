"""둘째 글용 예시 그림 3장. 흰 배경, 라벨 포함. 선택은 CSV 로 결정적으로 한다."""
import os, json, numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont
R="results"; S="PIE-bench-150"; O="figures"; os.makedirs(O, exist_ok=True)
M=["structure_distance","psnr_unedit_part","lpips_unedit_part","mse_unedit_part","ssim_unedit_part","clip_similarity_target_image","clip_similarity_target_image_edit_part"]
def load(name):
    df=pd.read_csv(os.path.join(R,name+".csv")); df["file_id"]=df["file_id"].astype(str).str.zfill(12)
    tag=[c.split("|")[0] for c in df.columns if "|" in c][0]
    out=df[["file_id"]].copy()
    for k in M: out[k]=pd.to_numeric(df[tag+"|"+k],errors="coerce")
    return out.set_index("file_id")
mapping=json.load(open(os.path.join(S,"mapping_file.json")))
def stem_of(fid): return os.path.splitext(os.path.basename(mapping[fid]["image_path"]))[0]
def src_img(fid): return Image.open(os.path.join(S,"annotation_images",mapping[fid]["image_path"])).convert("RGB")
def edit_img(cond, fid):
    g=Image.open("outputs/%s/%s_edited.png"%(cond,stem_of(fid))).convert("RGB"); W,H=g.size
    return g.crop((W-512,H-512,W,H)).resize((512,512))
def mask_img(fid): return Image.open(os.path.join(S,"mask_gt",stem_of(fid)+"_mask.png")).convert("L")
def overlay(img, mask, color=(255,60,60), alpha=0.45):
    a=np.asarray(img).astype(np.float32); m=(np.asarray(mask.resize(img.size))>127)[:,:,None]
    col=np.array(color,np.float32)[None,None,:]
    out=np.where(m, a*(1-alpha)+col*alpha, a)
    return Image.fromarray(out.astype(np.uint8))
try:
    FONT=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    FONT_S=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
except Exception:
    FONT=ImageFont.load_default(); FONT_S=FONT
def wrap(text, width):
    words=text.split(" "); lines=[]; cur=""
    for w in words:
        if len(cur)+len(w)+1>width and cur: lines.append(cur); cur=w
        else: cur=(cur+" "+w).strip()
    if cur: lines.append(cur)
    return lines
def panel(images, labels, title, size=384, pad=16):
    n=len(images); W=n*size+(n+1)*pad
    tl=wrap(title, max(60, W//12)); H=size+pad*2+28*len(tl)+8+3*22
    canvas=Image.new("RGB",(W,H),(255,255,255)); d=ImageDraw.Draw(canvas)
    for j,t in enumerate(tl): d.text((pad,pad+j*28),t,fill=(0,0,0),font=FONT)
    y0=pad+28*len(tl)+8
    for i,(im,lb) in enumerate(zip(images,labels)):
        x=pad+i*(size+pad)
        canvas.paste(im.resize((size,size),Image.LANCZOS),(x,y0))
        for j,t in enumerate(lb.split("  ")): d.text((x,y0+size+6+j*22),t,fill=(40,40,40),font=FONT_S)
    return canvas

# --- Fig 1: largest CLIP-edit drop under GT mask ---
best=None
for meth,nm,gm in [("FlowAlign","flowalign_nomask","flowalign_gtmask"),("DirectEdit","nomask","gtmask"),("DNAEdit","dnaedit_nomask","dnaedit_gtmask"),("FTEdit","ftedit_nomask","ftedit_gtmask"),("FlowEdit","flowedit_nomask","flowedit_gtmask")]:
    a=load(nm); b=load(gm); d=(a["clip_similarity_target_image_edit_part"]-b["clip_similarity_target_image_edit_part"])
    fid=d.idxmax()
    if best is None or d.max()>best[0]: best=(d.max(),meth,nm,gm,fid,a.loc[fid],b.loc[fid])
drop,meth,nm,gm,fid,ra,rb=best
it=mapping[fid]
fig=panel([src_img(fid), overlay(src_img(fid),mask_img(fid)), edit_img(nm,fid), edit_img(gm,fid)],
          ["source","GT mask (red = editable)","%s, no mask  CLIP-edit %.1f"%(meth,ra["clip_similarity_target_image_edit_part"]),"%s, GT mask  CLIP-edit %.1f"%(meth,rb["clip_similarity_target_image_edit_part"])],
          "Largest CLIP-edit drop from pasting: %s -> %s  |  id %s"%(it["original_prompt"].replace("[","").replace("]",""), it["editing_prompt"].replace("[","").replace("]",""), fid)[:220])
fig.save(os.path.join(O,"fig1-clip-drop.png")); print("fig1", meth, fid, "drop %.2f"%drop, "bg psnr %.2f -> %.2f"%(ra["psnr_unedit_part"],rb["psnr_unedit_part"]))

# --- Fig 2: paste hurt Structure most (seam candidate), mask area 0.1-0.5 ---
def area(fid):
    return (np.asarray(mask_img(fid))>127).mean()
best=None
for meth,nm,gm in [("DirectEdit","nomask","gtmask"),("FlowAlign","flowalign_nomask","flowalign_gtmask"),("DNAEdit","dnaedit_nomask","dnaedit_gtmask"),("FTEdit","ftedit_nomask","ftedit_gtmask"),("FlowEdit","flowedit_nomask","flowedit_gtmask")]:
    a=load(nm); b=load(gm); d=(b["structure_distance"]-a["structure_distance"])
    for fid in d.sort_values(ascending=False).index[:15]:
        ar=area(fid)
        if 0.1<=ar<=0.5:
            if best is None or d[fid]>best[0]: best=(d[fid],meth,nm,gm,fid,a.loc[fid],b.loc[fid],ar)
            break
dS,meth,nm,gm,fid,ra,rb,ar=best
it=mapping[fid]
m=np.asarray(mask_img(fid))>127; ys,xs=np.nonzero(m); x0,x1,y0,y1=max(0,xs.min()-40),min(512,xs.max()+40),max(0,ys.min()-40),min(512,ys.max()+40)
side=max(x1-x0,y1-y0); cx,cy=(x0+x1)//2,(y0+y1)//2; x0,y0=max(0,cx-side//2),max(0,cy-side//2); x1,y1=min(512,x0+side),min(512,y0+side)
crop=lambda im: im.crop((x0,y0,x1,y1))
fig=panel([overlay(src_img(fid),mask_img(fid)), edit_img(nm,fid), edit_img(gm,fid), crop(edit_img(gm,fid))],
          ["source + GT mask","%s, no mask  Struct %.1f"%(meth,ra["structure_distance"]),"%s, GT mask  Struct %.1f  bg-PSNR %.1f"%(meth,rb["structure_distance"],rb["psnr_unedit_part"]),"GT mask, zoom at the boundary"],
          "Paste raised Structure Distance: %s -> %s  |  id %s"%(it["original_prompt"].replace("[","").replace("]",""), it["editing_prompt"].replace("[","").replace("]",""), fid)[:220])
fig.save(os.path.join(O,"fig2-seam.png")); print("fig2", meth, fid, "dStruct %+.2f area %.2f"%(dS,ar))

# --- Fig 3: FlowAlign fixed-range vs min/max save, largest PSNR gap ---
a=load("flowalign_nomask"); b=load("flowalign_minmax151_nomask"); d=(a["psnr_unedit_part"]-b["psnr_unedit_part"]).dropna()
fid=d.idxmax(); it=mapping[fid]
fig=panel([src_img(fid), edit_img("flowalign_nomask",fid), edit_img("flowalign_minmax151_nomask",fid)],
          ["source","FlowAlign edit, fixed [-1,1] save  bg-PSNR %.1f"%a.loc[fid,"psnr_unedit_part"],"same edit, per-image min/max save  bg-PSNR %.1f"%b.loc[fid,"psnr_unedit_part"]],
          "Same FlowAlign output, two save paths  |  %s -> %s  |  id %s"%(it["original_prompt"].replace("[","").replace("]",""), it["editing_prompt"].replace("[","").replace("]",""), fid)[:220])
fig.save(os.path.join(O,"fig3-flowalign-save.png")); print("fig3", fid, "psnr gap %.2f"%d.max(), "mean gap %.2f"%d.mean())
