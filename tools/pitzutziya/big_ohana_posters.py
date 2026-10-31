import sys, random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
src, dst = sys.argv[1], sys.argv[2]
S = 2  # upscale factor (keeps the crunchy pixel look but makes text legible)
base = Image.open(src).convert('RGBA')
im = base.resize((base.width*S, base.height*S), Image.NEAREST)
B = '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf'
def F(px): return ImageFont.truetype(B, px*S//2 if False else px)
random.seed(7)

def photo(w, h, wide=1.0, cap=False):
    """grainy photocopied portrait of a big broad guy"""
    p = Image.new('RGB', (w, h), (120,120,120))
    d = ImageDraw.Draw(p)
    cx = w//2
    # shoulders / body - very wide
    d.ellipse((cx-int(w*0.62*wide), int(h*0.55), cx+int(w*0.62*wide), int(h*1.45)), fill=(28,28,28))
    # thick neck
    d.rectangle((cx-int(w*0.17), int(h*0.42), cx+int(w*0.17), int(h*0.65)), fill=(36,36,36))
    # big round head
    d.ellipse((cx-int(w*0.27), int(h*0.10), cx+int(w*0.27), int(h*0.62)), fill=(40,40,40))
    if cap:
        d.chord((cx-int(w*0.29), int(h*0.04), cx+int(w*0.29), int(h*0.40)), 180, 360, fill=(18,18,18))
        d.rectangle((cx-int(w*0.29), int(h*0.21), cx+int(w*0.45), int(h*0.26)), fill=(18,18,18))
    # sunglasses
    d.rectangle((cx-int(w*0.21), int(h*0.30), cx+int(w*0.21), int(h*0.36)), fill=(10,10,10))
    # gold chain
    for i in range(9):
        t = i/8; x = cx-int(w*0.2)+int(t*w*0.4); y = int(h*0.66 + 0.06*h*(1-(2*t-1)**2))
        d.ellipse((x-2,y-2,x+2,y+2), fill=(150,150,150))
    # photocopy grain
    px = p.load()
    for y in range(h):
        for x in range(w):
            r = px[x,y][0] + random.randint(-28,28)
            r = max(0,min(255,r)); px[x,y] = (r,r,r)
    return p

def text_c(d, cx, y, s, size, fill):
    f = ImageFont.truetype(B, size)
    w = d.textlength(s, font=f, direction='rtl', language='he') if any('֐'<=c<='׿' for c in s) else d.textlength(s, font=f)
    kw = dict(direction='rtl', language='he') if any('֐'<=c<='׿' for c in s) else {}
    d.text((cx - w/2, y), s, font=f, fill=fill, **kw)

def poster(x0, y0, w, h, variant):
    x0*=S; y0*=S; w*=S; h*=S
    p = Image.new('RGBA', (w, h), (238,236,226,255))
    d = ImageDraw.Draw(p)
    cx = w//2
    if variant == 0:
        d.rectangle((0,0,w,int(h*0.18)), fill=(196,24,24))
        text_c(d, cx, 2, 'נעדר', 22, (255,255,255))
        ph = photo(int(w*0.56), int(h*0.38), wide=1.15)
        p.paste(ph, (cx-ph.width//2-1, int(h*0.21)-1)); d.rectangle((cx-ph.width//2-2, int(h*0.21)-2, cx+ph.width//2+1, int(h*0.21)+ph.height), outline=(20,20,20))
        y = int(h*0.62)
        text_c(d, cx, y, 'דה ביג אוחנה', 14, (15,15,15)); y += 16
        text_c(d, cx, y, 'THE BIG OHANA', 10, (15,15,15)); y += 13
        text_c(d, cx, y, 'נראה לאחרונה בלילה', 9, (40,40,40)); y += 11
        text_c(d, cx, y, 'בפיצוצייה', 9, (40,40,40)); y += 12
        text_c(d, cx, y, '03-6812457', 10, (196,24,24))
    else:
        p = Image.new('RGBA', (w, h), (222,208,170,255)); d = ImageDraw.Draw(p)
        text_c(d, cx, 3, 'נעדר', 22, (150,70,40))
        ph = photo(int(w*0.56), int(h*0.38), wide=1.25, cap=True)
        p.paste(ph, (cx-ph.width//2-1, int(h*0.21)-1)); d.rectangle((cx-ph.width//2-2, int(h*0.21)-2, cx+ph.width//2+1, int(h*0.21)+ph.height), outline=(70,60,45))
        y = int(h*0.62)
        text_c(d, cx, y, 'ראיתם את', 10, (60,50,40)); y += 12
        text_c(d, cx, y, 'הביג אוחנה', 14, (60,50,40)); y += 17
        text_c(d, cx, y, 'יצא לקנות סיגריות', 9, (90,80,65)); y += 11
        text_c(d, cx, y, 'ולא חזר', 9, (90,80,65)); y += 12
        text_c(d, cx, y, 'ליד הקיוסק', 10, (60,50,40))
        # aged / stained paper
        st = Image.new('RGBA', (w,h), (0,0,0,0)); sd = ImageDraw.Draw(st)
        for _ in range(6):
            rx, ry, rr = random.randint(0,w), random.randint(0,h), random.randint(6,16)
            sd.ellipse((rx-rr,ry-rr,rx+rr,ry+rr), fill=(120,90,40,40))
        p = Image.alpha_composite(p, st.filter(ImageFilter.GaussianBlur(4)))
    # crumple noise + tape strips
    px = p.load()
    for yy in range(h):
        for xx in range(w):
            r,g,b,a = px[xx,yy]; n = random.randint(-6,6); px[xx,yy] = (max(0,min(255,r+n)),max(0,min(255,g+n)),max(0,min(255,b+n)),255)
    d = ImageDraw.Draw(p)
    d.rectangle((2,1,14,5), fill=(230,226,190,255)); d.rectangle((w-15,1,w-3,5), fill=(230,226,190,255))
    im.paste(p, (x0, y0))

poster(128, 0, 64, 96, 0)
poster(192, 0, 64, 96, 1)
im.save(dst, optimize=True)
