import json, struct, sys, math
import numpy as np
SRC, PAPER, DST = sys.argv[1], sys.argv[2], sys.argv[3]
d = open(SRC,'rb').read()
jl = struct.unpack('<I', d[12:16])[0]
J = json.loads(d[20:20+jl]); b0 = 20+jl+8
bin_old = d[b0:b0+struct.unpack('<I', d[20+jl:24+jl])[0]]
MAT = {m['name']: i for i, m in enumerate(J['materials'])}
NODE = {n.get('name'): i for i, n in enumerate(J['nodes'])}

# ---------- rebuild binary (swap the paper texture) ----------
paper_img = next(i for i, im in enumerate(J['images']) if im['name'] == 'T_Paper')
new_bin = bytearray()
for i, bv in enumerate(J['bufferViews']):
    chunk = bin_old[bv.get('byteOffset', 0): bv.get('byteOffset', 0)+bv['byteLength']]
    if i == J['images'][paper_img]['bufferView']:
        chunk = open(PAPER, 'rb').read()
    while len(new_bin) % 4: new_bin += b'\0'
    bv['byteOffset'] = len(new_bin); bv['byteLength'] = len(chunk); bv['buffer'] = 0
    new_bin += chunk

def add_view(arr, target):
    global new_bin
    while len(new_bin) % 4: new_bin += b'\0'
    off = len(new_bin); b = arr.tobytes(); new_bin += b
    J['bufferViews'].append({'buffer': 0, 'byteOffset': off, 'byteLength': len(b), 'target': target})
    return len(J['bufferViews'])-1

def add_acc(arr, typ, target):
    arr = np.ascontiguousarray(arr)
    ct = 5126 if arr.dtype == np.float32 else 5125
    a = {'bufferView': add_view(arr, target), 'componentType': ct, 'count': len(arr), 'type': typ}
    if typ == 'VEC3': a['min'] = arr.min(0).tolist(); a['max'] = arr.max(0).tolist()
    J['accessors'].append(a); return len(J['accessors'])-1

# ---------- geometry helpers ----------
class MB:
    def __init__(s): s.p = {}
    def quad(s, mat, pts, uvs=None, density=1.0):
        P = [np.array(p, float) for p in pts]
        n = np.cross(P[1]-P[0], P[2]-P[0]); n /= np.linalg.norm(n)
        if uvs is None:
            ax = np.argmax(np.abs(n))
            a, b = [(2, 1), (0, 2), (0, 1)][ax]
            uvs = [(p[a]*density, -p[b]*density) for p in P]
        L = s.p.setdefault(mat, ([], [], [], []))
        base = len(L[0])
        for p, uv in zip(P, uvs): L[0].append(p); L[1].append(n); L[2].append(uv)
        L[3].extend([base, base+1, base+2, base, base+2, base+3])
    def hexa(s, mat, c, density=1.0, matfn=None):
        # c: 8 corners in box order: (x0y0z0,x1y0z0,x1y1z0,x0y1z0, x0y0z1,x1y0z1,x1y1z1,x0y1z1)
        faces = [(0,3,2,1), (4,5,6,7), (0,4,7,3), (1,2,6,5), (0,1,5,4), (3,7,6,2)]
        for f in faces:
            pts = [c[i] for i in f]
            P = [np.array(p, float) for p in pts]
            n = np.cross(P[1]-P[0], P[2]-P[0]); n /= np.linalg.norm(n)
            m = matfn(n, np.mean(P, 0)) if matfn else mat
            if m is not None: s.quad(m, pts, density=density)
    def box(s, mat, x0, y0, z0, x1, y1, z1, density=1.0, matfn=None):
        c = [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]
        s.hexa(mat, c, density, matfn)
    def build(s, name):
        prims = []
        for mat, (P, N, U, I) in s.p.items():
            prims.append({'attributes': {
                'POSITION': add_acc(np.array(P, np.float32), 'VEC3', 34962),
                'NORMAL': add_acc(np.array(N, np.float32), 'VEC3', 34962),
                'TEXCOORD_0': add_acc(np.array(U, np.float32), 'VEC2', 34962)},
                'indices': add_acc(np.array(I, np.uint32), 'SCALAR', 34963), 'material': MAT[mat]})
        J['meshes'].append({'name': name, 'primitives': prims}); return len(J['meshes'])-1

def node(name, mesh=None, t=None, r=None, s=None, children=None, ext=None):
    n = {'name': name}
    if mesh is not None: n['mesh'] = mesh
    if t: n['translation'] = list(t)
    if r: n['rotation'] = list(r)
    if s: n['scale'] = [s, s, s] if isinstance(s, (int, float)) else list(s)
    if children: n['children'] = children
    if ext: n['extensions'] = ext
    J['nodes'].append(n); return len(J['nodes'])-1

def yrot(deg):
    a = math.radians(deg)/2; return (0, math.sin(a), 0, math.cos(a))

# ---------- layout ----------
FL = 3.30            # upper floor level (top of existing roof)
TOP = 5.90           # wall top
X0, X1 = 1.20, 5.20  # upper room outer footprint
Z0, Z1 = -8.20, -3.60
T = 0.20             # wall thickness
CX, CZ = (X0+X1)/2, (Z0+Z1)/2
DOOR_Z0, DOOR_Z1, DOOR_H = -7.42, -6.60, 2.12

def wallmat(n, c):   # outward faces get facade plaster, inward faces interior plaster
    if abs(n[1]) > 0.5: return 'M_Facade_Plaster'
    out = (c[0]-CX)*n[0] + (c[2]-CZ)*n[2]
    return 'M_Facade_Plaster' if out > 0 else 'M_Wall_Plaster'

# --- walls, floor, ceiling, roof ---
W = MB()
def wall_x(x0, x1, z0, z1, holes):  # wall running along X, holes = [(a0,a1,y0,y1)] in x
    segs = sorted(holes); cur = x0
    for a0, a1, y0, y1 in segs:
        W.box(None, cur, FL, z0, a0, TOP, z1, matfn=wallmat)
        if y0 > FL: W.box(None, a0, FL, z0, a1, y0, z1, matfn=wallmat)
        W.box(None, a0, y1, z0, a1, TOP, z1, matfn=wallmat)
        cur = a1
    W.box(None, cur, FL, z0, x1, TOP, z1, matfn=wallmat)
def wall_z(z0, z1, x0, x1, holes):
    segs = sorted(holes); cur = z0
    for a0, a1, y0, y1 in segs:
        W.box(None, x0, FL, cur, x1, TOP, a0, matfn=wallmat)
        if y0 > FL: W.box(None, x0, FL, a0, x1, y0, a1, matfn=wallmat)
        W.box(None, x0, y1, a0, x1, TOP, a1, matfn=wallmat)
        cur = a1
    W.box(None, x0, FL, cur, x1, TOP, z1, matfn=wallmat)

WIN_BACK = (2.60, 3.80, 4.25, 5.20)
WIN_FRONT = (2.00, 3.20, 4.25, 5.20)
WIN_LEFT = (-6.60, -5.40, 4.25, 5.20)
wall_x(X0, X1, Z0, Z0+T, [WIN_BACK])
wall_x(X0, X1, Z1-T, Z1, [WIN_FRONT])
wall_z(Z0+T, Z1-T, X0, X0+T, [WIN_LEFT])
wall_z(Z0+T, Z1-T, X1-T, X1, [(DOOR_Z0, DOOR_Z1, FL, FL+DOOR_H)])
# roof slab + parapet lip
W.box('M_Facade_Plaster', X0-0.05, TOP, Z0-0.05, X1+0.05, TOP+0.12, Z1+0.05)
W.quad('M_Asphalt', [(X0, TOP+0.121, Z1), (X1, TOP+0.121, Z1), (X1, TOP+0.121, Z0), (X0, TOP+0.121, Z0)], density=0.667)
for (a, b, c_, d_) in [(X0-0.05, X1+0.05, Z0-0.05, Z0+0.07), (X0-0.05, X1+0.05, Z1-0.07, Z1+0.05),
                       (X0-0.05, X0+0.07, Z0-0.05, Z1+0.05), (X1-0.07, X1+0.05, Z0-0.05, Z1+0.05)]:
    W.box('M_Facade_Plaster', a, TOP+0.12, c_, b, TOP+0.32, d_)
# interior floor / ceiling
W.quad('M_Wood_Laminate', [(X0+T, FL+0.005, Z1-T), (X1-T, FL+0.005, Z1-T), (X1-T, FL+0.005, Z0+T), (X0+T, FL+0.005, Z0+T)], density=1.5)
W.quad('M_Ceiling_Tiles', [(X0+T, TOP-0.005, Z0+T), (X1-T, TOP-0.005, Z0+T), (X1-T, TOP-0.005, Z1-T), (X0+T, TOP-0.005, Z1-T)], density=1.667)
# skirting board / door threshold
W.box('M_Metal_Painted', X1-T, FL, DOOR_Z0, X1, FL+0.02, DOOR_Z1)
m_walls = W.build('SM_Upper_Walls')

# --- windows (frames + glass) ---
G = MB()
def window_x(x0, x1, y0, y1, zc):
    f = 0.05
    G.box('M_Aluminium', x0, y0, zc-0.05, x1, y0+f, zc+0.05); G.box('M_Aluminium', x0, y1-f, zc-0.05, x1, y1, zc+0.05)
    G.box('M_Aluminium', x0, y0, zc-0.05, x0+f, y1, zc+0.05); G.box('M_Aluminium', x1-f, y0, zc-0.05, x1, y1, zc+0.05)
    G.box('M_Aluminium', (x0+x1)/2-0.02, y0, zc-0.04, (x0+x1)/2+0.02, y1, zc+0.04)
    G.box('M_Aluminium', x0-0.05, y0-0.04, zc-0.14 if zc < CZ else zc-0.05, x1+0.05, y0, zc+0.05 if zc < CZ else zc+0.14)  # sill
    G.quad('GL_Glass', [(x0, y0, zc), (x1, y0, zc), (x1, y1, zc), (x0, y1, zc)], uvs=[(0, 1), (1, 1), (1, 0), (0, 0)])
def window_z(z0, z1, y0, y1, xc):
    f = 0.05
    G.box('M_Aluminium', xc-0.05, y0, z0, xc+0.05, y0+f, z1); G.box('M_Aluminium', xc-0.05, y1-f, z0, xc+0.05, y1, z1)
    G.box('M_Aluminium', xc-0.05, y0, z0, xc+0.05, y1, z0+f); G.box('M_Aluminium', xc-0.05, y0, z1-f, xc+0.05, y1, z1)
    G.box('M_Aluminium', xc-0.04, y0, (z0+z1)/2-0.02, xc+0.04, y1, (z0+z1)/2+0.02)
    G.box('M_Aluminium', xc-0.14, y0-0.04, z0-0.05, xc+0.05, y0, z1+0.05)
    G.quad('GL_Glass', [(xc, y0, z0), (xc, y0, z1), (xc, y1, z1), (xc, y1, z0)], uvs=[(0, 1), (1, 1), (1, 0), (0, 0)])
window_x(WIN_BACK[0], WIN_BACK[1], WIN_BACK[2], WIN_BACK[3], Z0+T/2)
window_x(WIN_FRONT[0], WIN_FRONT[1], WIN_FRONT[2], WIN_FRONT[3], Z1-T/2)
window_z(WIN_LEFT[0], WIN_LEFT[1], WIN_LEFT[2], WIN_LEFT[3], X0+T/2)
# door frame (outside)
G.box('M_Aluminium', X1-0.02, FL, DOOR_Z0-0.06, X1+0.04, FL+DOOR_H+0.06, DOOR_Z0)
G.box('M_Aluminium', X1-0.02, FL, DOOR_Z1, X1+0.04, FL+DOOR_H+0.06, DOOR_Z1+0.06)
G.box('M_Aluminium', X1-0.02, FL+DOOR_H, DOOR_Z0, X1+0.04, FL+DOOR_H+0.06, DOOR_Z1)
m_windows = G.build('SM_Upper_Windows')

# --- exterior steel stair on the right side + landing + alley ground ---
S = MB()
SX0, SX1 = 5.28, 6.18
NSTEP = 18; RISE = FL/NSTEP; RUN = 0.27
Z_START = -0.85
for i in range(NSTEP-1):
    y = RISE*(i+1); z = Z_START - RUN*i
    S.box('M_Metal_Painted', SX0+0.05, y-0.04, z-RUN-0.02, SX1-0.05, y, z)
Z_LAND = Z_START - RUN*(NSTEP-1)            # stair top edge = start of landing
S.box('M_Metal_Painted', SX0, FL-0.08, -7.80, SX1, FL, Z_LAND)  # landing deck
# stringers (sloped)
for xa in (SX0, SX1-0.05):
    xb = xa+0.05
    zt, zb = Z_LAND, Z_START+0.02
    c = [(xa, 0.0, zb), (xb, 0.0, zb), (xb, FL+0.02, zt), (xa, FL+0.02, zt), (xa, 0.0, zb-0.35), (xb, 0.0, zb-0.35), (xb, FL-0.30, zt), (xa, FL-0.30, zt)]
    S.hexa('M_Metal_Painted', [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
# landing posts
for (px, pz) in [(SX1-0.05, Z_LAND), (SX1-0.05, -7.75), (SX0, -7.75)]:
    S.box('M_Metal_Painted', px, 0, pz-0.05, px+0.05, FL-0.08, pz)
# handrails: outer stair rail + landing rails
HR = 0.95
for i in range(0, NSTEP, 3):
    y = RISE*(i+1); z = Z_START - RUN*i - RUN/2
    S.box('M_Metal_Painted', SX1-0.05, y, z-0.02, SX1-0.02, y+HR, z+0.02)
zt = Z_LAND; zb = Z_START
S.hexa('M_Metal_Painted', [(SX1-0.06, RISE+HR, zb), (SX1-0.01, RISE+HR, zb), (SX1-0.01, FL+HR, zt), (SX1-0.06, FL+HR, zt),
                           (SX1-0.06, RISE+HR+0.04, zb), (SX1-0.01, RISE+HR+0.04, zb), (SX1-0.01, FL+HR+0.04, zt), (SX1-0.06, FL+HR+0.04, zt)])
S.box('M_Metal_Painted', SX1-0.06, FL+HR, -7.80, SX1-0.01, FL+HR+0.04, Z_LAND)
S.box('M_Metal_Painted', SX0, FL+HR, -7.80, SX1, FL+HR+0.04, -7.75)
S.box('M_Metal_Painted', SX1-0.06, FL+0.45, -7.80, SX1-0.01, FL+0.48, Z_LAND)
for pz in (-7.2, -6.6, -6.0):
    S.box('M_Metal_Painted', SX1-0.05, FL, pz-0.02, SX1-0.02, FL+HR, pz+0.02)
m_stairs = S.build('SM_Upper_Stairs')

A = MB()  # alley ground beside the kiosk so the stair stands on something
A.box('M_Sidewalk', 5.20, -0.12, -8.20, 6.50, 0.0, 0.0)
A.box('M_Facade_Plaster', 6.50, -0.12, -8.20, 6.65, 1.10, 0.0)
A.box('M_Facade_Plaster', 6.47, 1.10, -8.20, 6.68, 1.16, 0.0)
m_alley = A.build('SM_Alley')

# --- rooftop solar water heater ("dud shemesh") on the upper roof ---
H = MB()
H.box('M_Metal_Painted', 2.0, TOP+0.12, -7.6, 2.08, TOP+0.75, -7.52); H.box('M_Metal_Painted', 3.2, TOP+0.12, -7.6, 3.28, TOP+0.75, -7.52)
H.box('M_Enamel', 1.95, TOP+0.75, -7.85, 3.35, TOP+1.25, -7.35)                         # tank
H.hexa('GL_Glass', [(1.9, TOP+0.15, -6.0), (3.4, TOP+0.15, -6.0), (3.4, TOP+0.95, -7.2), (1.9, TOP+0.95, -7.2),
                    (1.9, TOP+0.20, -6.0), (3.4, TOP+0.20, -6.0), (3.4, TOP+1.00, -7.2), (1.9, TOP+1.00, -7.2)])
H.box('M_Plastic_Dark', 1.9, TOP+0.12, -6.05, 3.4, TOP+0.95, -6.0)
m_heater = H.build('SM_Upper_SolarHeater')

# --- interior bits: mattress, wall of posters ---
I = MB()
I.box('M_Burlap', 1.45, FL, -7.95, 2.35, FL+0.18, -6.05, density=2)
# Big Ohana posters taped on the inside walls (UV regions of the two posters in T_Paper)
P0 = [(0.751953, 0.373047), (0.501953, 0.373047), (0.501953, 0.001953), (0.751953, 0.001953)]   # red header poster
P1 = [(0.998047, 0.373047), (0.751953, 0.373047), (0.751953, 0.001953), (0.998047, 0.001953)]   # beige poster
def poster_back(x, y, uv):   # on back wall inner face (z = Z0+T), facing +z
    z = Z0+T+0.006; w, h = 0.40, 0.60
    I.quad('M_Paper', [(x+w, y, z), (x, y, z), (x, y+h, z), (x+w, y+h, z)], uvs=[uv[0], uv[1], uv[2], uv[3]])
def poster_left(z, y, uv):   # on left wall inner face, facing +x
    x = X0+T+0.006; w, h = 0.40, 0.60
    I.quad('M_Paper', [(x, y, z), (x, y, z-w), (x, y+h, z-w), (x, y+h, z)], uvs=[uv[1], uv[0], uv[3], uv[2]])
poster_back(1.55, 4.15, P0); poster_back(2.05, 4.30, P1); poster_back(4.05, 4.10, P0); poster_back(4.35, 4.75, P1)
poster_left(-4.15, 4.20, P1); poster_left(-4.65, 4.05, P0); poster_left(-6.95, 4.30, P0); poster_left(-7.40, 4.00, P1)
m_int = I.build('SM_Upper_Interior')

# ---------- nodes ----------
kids = []
kids.append(node('SM_Upper_Walls', m_walls))
kids.append(node('SM_Upper_Windows', m_windows))
kids.append(node('SM_Upper_Stairs', m_stairs))
kids.append(node('SM_Alley', m_alley))
kids.append(node('SM_Upper_SolarHeater', m_heater))
kids.append(node('SM_Upper_Interior', m_int))
def reuse(name, src, t, r=None, s=None):
    n = J['nodes'][NODE[src]]
    kids.append(node(name, n['mesh'], t, r or n.get('rotation'), s or n.get('scale')))
reuse('PROP_Upper_Chair', 'PROP_Owner_Chair', (4.2, FL, -4.6), yrot(200))
reuse('PROP_Upper_Crate_0', 'PROP_Crate_0', (3.3, FL, -7.6), yrot(0))
reuse('PROP_Upper_Crate_1', 'PROP_Crate_1', (3.3, FL+0.2*0.67, -7.6), yrot(6))
reuse('PROP_Upper_Radio', 'PROP_Radio', (3.3, FL+0.27, -7.6), yrot(10))
reuse('PROP_Upper_Teddy', 'PROP_Teddy', (1.9, FL+0.18, -7.6), yrot(150))
reuse('PROP_Upper_Bin', 'PROP_Bin_In', (4.7, FL, -4.2), yrot(30))
reuse('PROP_Upper_Ashtray', 'PROP_Ashtray', (3.5, FL+0.27, -7.55))
reuse('PROP_Upper_Butt', 'PROP_Butt_0', (3.52, FL+0.28, -7.55))
reuse('PROP_Upper_Clock', 'PROP_Clock', (X0+T, 5.15, -5.0), (0,0,0,1))
reuse('PROP_Upper_Flashlight', 'PROP_Flashlight', (2.6, FL, -4.4), yrot(70))
# bulb + light (copy of the WC bulb / light)
wc_bulb = J['nodes'][NODE['SM_WC_Bulb']]
kids.append(node('SM_Upper_Bulb', wc_bulb['mesh'], (-1.0, TOP-2.6, 1.2)))
J['extensions']['KHR_lights_punctual']['lights'].append(dict(J['extensions']['KHR_lights_punctual']['lights'][14], name='PL_Upper_Bulb'))
li = len(J['extensions']['KHR_lights_punctual']['lights'])-1
pl = node('PL_Upper_Bulb', None, None, (-0.7071068, 0, 0, 0.7071068), ext={'KHR_lights_punctual': {'light': li}})
kids.append(node('LGT_Upper_Bulb', None, (3.2, 5.45, -5.9), children=[pl]))
grp = node('GRP_Upper_Floor', children=kids)
J['nodes'][NODE['Pitzutziya']]['children'].append(grp)
# door: copy of the WC door, hinged on the right wall
wc = J['meshes'][J['nodes'][NODE['DOOR_WC']]['mesh']]
keep = {MAT['M_Door_WC'], MAT['M_Wood_Laminate'], MAT['M_Aluminium']}
J['meshes'].append({'name': 'DOOR_Upper', 'primitives': [dict(pr) for pr in wc['primitives'] if pr['material'] in keep]})
door = node('DOOR_Upper', len(J['meshes'])-1, (X1-T/2, FL, DOOR_Z1-0.01), yrot(-90))
J['nodes'][NODE['GRP_Moving']]['children'].append(door)

# ---------- write ----------
while len(new_bin) % 4: new_bin += b'\0'
J['buffers'] = [{'byteLength': len(new_bin)}]
js = json.dumps(J, separators=(',', ':')).encode()
while len(js) % 4: js += b' '
out = struct.pack('<III', 0x46546C67, 2, 12+8+len(js)+8+len(new_bin)) + struct.pack('<II', len(js), 0x4E4F534A) + js + struct.pack('<II', len(new_bin), 0x004E4942) + new_bin
open(DST, 'wb').write(out)
print('wrote', DST, len(out))
