"""- lee logs ULog de PX4 y Hivemind, detecta fallas y compara si son del mismo vuelo.
Solo usa numpy (viene en Pyodide). Lector propio y tolerante: repara cabeceras dañadas y logs truncados."""
import struct, re, json
import numpy as np

NP = {'int8_t':'i1','uint8_t':'u1','int16_t':'i2','uint16_t':'u2','int32_t':'i4','uint32_t':'u4',
      'int64_t':'i8','uint64_t':'u8','float':'f4','double':'f8','bool':'?','char':'S1'}
NAV = {0:'MANUAL',1:'ALTCTL',2:'POSCTL',3:'MISSION',4:'LOITER',5:'RTL',10:'ACRO',12:'DESCEND',13:'TERMINATION',
       14:'OFFBOARD',15:'STAB',17:'TAKEOFF',18:'LAND',19:'FOLLOW',20:'PRECLAND',21:'ORBIT'}
LVL = ['EMERG','ALERT','CRIT','ERR','WARN','NOTICE','INFO','DEBUG']

# ---------------------------------------------------------------- lector ULog
def _kv(b):
    kl = b[0]; k = b[1:1+kl].decode(errors='ignore'); v = b[1+kl:]
    ty, name = k.split(' ', 1)
    if ty.startswith('char'): val = v.decode(errors='ignore').strip('\0')
    elif ty in ('int32_t','uint32_t') and len(v) == 4: val = struct.unpack('<i' if ty[0]=='i' else '<I', v)[0]
    elif ty == 'float' and len(v) == 4: val = struct.unpack('<f', v)[0]
    else: val = v
    return name, val

def parse(d):
    d = bytes(d); n = len(d)
    if d[:7] != b'ULog\x01\x12\x35': raise ValueError('No es un archivo ULog')
    R = dict(fmt={}, sub={}, rows={}, logs=[], info={}, par={}, drop=[], trunc=False, repaired=0, n=n)
    pos, last = 16, 0
    while pos + 3 <= n:
        sz, t = struct.unpack_from('<HB', d, pos); t = chr(t); q = pos + 3
        if t == 'F':  # reparación: el Hivemind a veces mete mensajes INFO entre la cabecera y el cuerpo del formato
            while q + 4 <= n and d[q+2:q+3] == b'I' and d[q+3] < 60 and b' ' in d[q+4:q+4+d[q+3]]:
                s2 = struct.unpack_from('<H', d, q)[0]
                try: k, v = _kv(d[q+3:q+3+s2]); R['info'][k] = v
                except Exception: break
                q += 3 + s2; R['repaired'] += 1
        if q + sz > n: R['trunc'] = True; break
        b = d[q:q+sz]; pos = q + sz
        if t == 'D':
            mid = struct.unpack_from('<H', b)[0]; R['rows'].setdefault(mid, []).append(b[2:])
            if len(b) >= 10: last = struct.unpack_from('<Q', b, 2)[0]
        elif t == 'F':
            name, rest = b.decode(errors='ignore').split(':', 1)
            R['fmt'][name] = [tuple(f.split(' ', 1)) for f in rest.split(';') if f]
        elif t == 'A': R['sub'][struct.unpack_from('<H', b, 1)[0]] = (b[3:].decode(), b[0])
        elif t == 'L': R['logs'].append((struct.unpack_from('<Q', b, 1)[0], b[0], b[9:].decode(errors='ignore').strip('\t\0 ')))
        elif t == 'I':
            try: k, v = _kv(b); R['info'][k] = v
            except Exception: pass
        elif t == 'P':
            try: k, v = _kv(b); R['par'][k] = v
            except Exception: pass
        elif t == 'O': R['drop'].append((last, struct.unpack_from('<H', b)[0]))
    return R

def _dtype(fm, name):
    out = []; fl = list(fm[name])
    while fl and fl[-1][1].startswith('_padding'): fl.pop()  # ULog no guarda el relleno final
    for i, (ty, nm) in enumerate(fl):
        base, cnt = re.match(r'(\w+)(?:\[(\d+)\])?', ty).groups(); cnt = int(cnt) if cnt else 1
        if nm.startswith('_padding'): nm = '_pad%d' % i
        if base == 'char': out.append((nm, 'S%d' % cnt))
        elif base in NP: out.append((nm, NP[base]) if cnt == 1 else (nm, NP[base], (cnt,)))
        else: out.append((nm, _dtype(fm, base)) if cnt == 1 else (nm, _dtype(fm, base), (cnt,)))
    return np.dtype(out)

def topic(R, name, m=0):
    mid = next((k for k, (nn, mm) in R['sub'].items() if nn == name and mm == m), None)
    if mid is None or name not in R['fmt']: return None
    dt = _dtype(R['fmt'], name); rows = [r for r in R['rows'].get(mid, []) if len(r) == dt.itemsize]
    if not rows: return None
    a = np.frombuffer(b''.join(rows), dtype=dt); out = {}
    for f in a.dtype.names:
        if f.startswith('_pad') or a[f].dtype.names: continue
        if a[f].ndim == 2:
            for i in range(a[f].shape[1]): out['%s[%d]' % (f, i)] = a[f][:, i]
        else: out[f] = a[f]
    return out

# ---------------------------------------------------------------- utilidades
def runs(ts, mask, minlen=0.0):
    d = np.diff(np.r_[0, np.asarray(mask).astype(int), 0]); s = np.where(d == 1)[0]; e = np.where(d == -1)[0] - 1
    return [(ts[a], ts[b]) for a, b in zip(s, e) if ts[b] - ts[a] >= minlen]

def L(a, step=1):
    a = np.asarray(a, float)[::step]
    return [None if not np.isfinite(v) else round(float(v), 3) for v in a]

def ds(x, *ys, k=1200):
    st = max(1, len(x) // k); return (x[::st],) + tuple(y[::st] for y in ys)

def ev(t, sev, cat, title, detail='', src='', approx=False):
    return dict(t=float(t), sev=sev, cat=cat, title=title, detail=detail, src=src, approx=approx)

def log_events(R, tf, src, w0, approx=False):
    """Mensajes de texto -> eventos. Agrupa repetidos. Antes del despegue (w0) se limita a 'warn'."""
    out = []; g = None
    for ts, lv, msg in R['logs']:
        lv = lv - 48 if lv >= 48 else lv; t = tf(ts)
        if re.search(r'progress|^\[cal\] calibration', msg) or not msg: continue
        key = re.sub(r'[\d.]+', '#', msg)
        if g and g['key'] == key and t - g['t1'] < 10: g['n'] += 1; g['t1'] = t; continue
        g = dict(key=key, msg=msg, lv=lv, t0=t, t1=t, n=1); out.append(g)
    res = []
    for g in out:
        pre = w0 is not None and g['t0'] < w0
        sev = 'crit' if g['lv'] <= 3 else 'warn' if (g['lv'] == 4 or re.search(r'Fail|Unknown|Error|error', g['msg'])) else 'info'
        if pre and sev == 'crit': sev = 'warn'
        cat = 'comms' if re.search(r'write error|Serial|serial|link', g['msg']) else ('preflight' if pre else 'mensaje')
        det = ('%d veces en %.1f s' % (g['n'], g['t1'] - g['t0'])) if g['n'] > 1 else ''
        res.append(ev(g['t0'], sev, cat, g['msg'][:110], det, src, approx))
    return res

def air_window(R, tf):
    """Ventana en el aire (UTC). PX4: vehicle_land_detected; si no hay, None."""
    ld = topic(R, 'vehicle_land_detected')
    if ld is None: return None
    ts = ld['timestamp'] / 1e6; idx = np.where(~ld['landed'].astype(bool))[0]
    if not len(idx): return None
    return tf(ld['timestamp'][idx[0]]), tf(ld['timestamp'][idx[-1]])

# ---------------------------------------------------------------- análisis PX4
def analyze_px4(R, name):
    g = topic(R, 'sensor_gps'); ok = g['time_utc_usec'] > 0 if g else None
    off = float(np.median(g['time_utc_usec'][ok] - g['timestamp'][ok])) if g is not None and ok.any() else None
    tf = (lambda ts: (ts + off) / 1e6) if off is not None else (lambda ts: ts / 1e6)
    F = dict(name=name, kind='PX4', size=R['n'], trunc=R['trunc'], repaired=R['repaired'], utc=off is not None,
             uuid=str(R['info'].get('sys_uuid') or '').strip(), hw=R['info'].get('ver_hw'), drops=[(tf(t) if t else None, ms) for t, ms in R['drop']])
    E = []; M = {}
    lp = topic(R, 'vehicle_local_position')
    F['t0'], F['t1'] = tf(lp['timestamp'][0]), tf(lp['timestamp'][-1])
    win = air_window(R, tf); F['win'] = win; w0, w1 = win if win else (None, None)
    F['logger'] = next((m.group(1) for _, _, s in R['logs'] for m in [re.search(r'log/(\d{4}-\d\d-\d\d/\d+_\d+_\d+)\.ulg', s)] if m), None)
    air = lambda t: w0 is not None and w0 <= t <= w1
    E += log_events(R, tf, 'PX4', w0)
    # modos de vuelo
    vs = topic(R, 'vehicle_status'); modes = []
    if vs is not None:
        ns = vs['nav_state']; ch = np.r_[0, np.where(np.diff(ns) != 0)[0] + 1]
        for i in ch:
            t = tf(vs['timestamp'][i]); modes.append((t, NAV.get(int(ns[i]), str(int(ns[i])))))
            if i and air(t):
                a, b = NAV.get(int(ns[i-1]), '?'), NAV.get(int(ns[i]), '?')
                sev = 'crit' if b == 'TERMINATION' else 'warn' if b in ('LAND', 'RTL', 'DESCEND', 'LOITER') else 'info'
                E.append(ev(t, sev, 'modo', 'Cambio de modo %s -> %s en vuelo' % (a, b), 'altura %.0f m' % -np.interp(vs['timestamp'][i], lp['timestamp'], lp['z']), 'PX4'))
    # failsafes
    ff = topic(R, 'failsafe_flags')
    if ff is not None:
        tt = ff['timestamp']
        for k, v in ff.items():
            if k == 'timestamp' or k.startswith('mode_req') or v.dtype.kind not in 'b?u': continue
            for a, b in runs(tt, v.astype(bool)):
                if air(tf(a)) or k in ('battery_unhealthy',): E.append(ev(tf(a), 'crit' if not k.startswith('local_position_accuracy') else 'warn', 'failsafe', 'Failsafe: %s' % k, 'duración %.1f s' % (tf(b) - tf(a)), 'PX4'))
    # batería
    bs = topic(R, 'battery_status')
    if bs is not None:
        v, rem = bs['voltage_v'], bs['remaining']; M['bat'] = (float(v[0]), float(v[-1]), float(rem[0]) * 100, float(rem[-1]) * 100)
        for thr, sev in ((0.07, 'crit'), (0.15, 'warn')):
            i = np.where(rem < thr)[0]
            if len(i): E.append(ev(tf(bs['timestamp'][i[0]]), sev, 'bateria', 'Batería por debajo de %d%%' % (thr * 100), '', 'PX4')); break
        if np.all(bs['current_a'] == 0): E.append(ev(F['t0'], 'info', 'bateria', 'El sensor de corriente marca 0 A todo el vuelo', 'la capacidad se estima solo por voltaje', 'PX4'))
    # GPS
    if g is not None:
        for a, b in runs(g['timestamp'], g['fix_type'] < 3, 0.5):
            if air(tf(a)): E.append(ev(tf(a), 'crit', 'gps', 'GPS sin fix 3D', '%.1f s' % ((b - a) / 1e6), 'PX4'))
        if 'satellites_used' in g and air(F['t0']) is not None: M['sats'] = int(g['satellites_used'].min())
        for a, b in runs(g['timestamp'], g['jamming_state'] >= 2, 0.5): E.append(ev(tf(a), 'warn', 'gps', 'Posible interferencia (jamming) GPS', '', 'PX4'))
    # EKF
    tr = topic(R, 'estimator_innovation_test_ratios')
    if tr is not None:
        ks = [k for k in tr if k not in ('timestamp', 'timestamp_sample')]
        for k in ks:
            rr = [(a, b) for a, b in runs(tr['timestamp'], tr[k] > 1, 0.5) if air(tf(a))]
            if rr: E.append(ev(tf(rr[0][0]), 'warn', 'ekf', 'EKF rechaza mediciones: %s > 1' % k, 'pico %.1f, %d episodios' % (tr[k].max(), len(rr)), 'PX4'))
    # vibración / clipping
    for m in range(3):
        im = topic(R, 'vehicle_imu_status', m)
        if im is None: continue
        pk = float(im['accel_vibration_metric'].max()); clip = int(sum(im[k][-1] for k in im if 'accel_clipping' in k))
        if pk > 30: E.append(ev(F['t0'], 'warn', 'vibracion', 'Vibración alta IMU%d: %.0f m/s²' % (m, pk), 'umbral 30', 'PX4'))
        if clip > 0: E.append(ev(F['t0'], 'info', 'vibracion', 'IMU%d: %d muestras con clipping del acelerómetro' % (m, clip), '', 'PX4'))
    # motores
    am = topic(R, 'actuator_motors')
    if am is not None:
        ks = [k for k in am if k.startswith('control[') and np.isfinite(am[k]).any()]; mm = np.array([am[k] for k in ks]).T
        tm = np.array([tf(x) for x in am['timestamp']]); inair = (tm >= w0) & (tm <= w1) if win else np.ones(len(tm), bool)
        sat = (mm > 0.95).any(axis=1) & inair; fr = sat.sum() / max(1, inair.sum()); M['mot'] = (np.nanmean(mm[inair], 0), np.nanmax(mm[inair], 0))
        if fr > 0.02: E.append(ev(tm[sat][0], 'warn', 'motores', 'Motores saturados (>95%%) el %.1f%% del vuelo' % (fr * 100), '', 'PX4'))
    # seguimiento de posición
    sp = topic(R, 'vehicle_local_position_setpoint')
    if sp is not None:
        n = min(len(lp['x']), len(sp['x'])); e = np.hypot(lp['x'][:n] - sp['x'][:n], lp['y'][:n] - sp['y'][:n]); e = np.where(np.isfinite(e), e, 0)
        M['trk'] = float(e.max())
        for a, b in runs(lp['timestamp'][:n], e > 1.0, 0.5): E.append(ev(tf(a), 'crit' if e.max() > 3 else 'warn', 'control', 'Error de seguimiento XY > 1 m', 'pico %.1f m' % e.max(), 'PX4'))
    # huecos en offboard (descartando los del logger)
    oc = topic(R, 'offboard_control_mode')
    if oc is not None:
        t = oc['timestamp'] / 1e6; dr = [x[0] / 1e6 for x in R['drop'] if x[0]]
        for i in np.where(np.diff(t) > 0.5)[0]:
            if any(t[i] - 0.3 <= x <= t[i + 1] + 0.3 for x in dr): continue
            E.append(ev(tf(t[i] * 1e6), 'warn', 'comms', 'Se corta el stream offboard %.1f s' % (t[i + 1] - t[i]), 'no coincide con pérdida del logger', 'PX4'))
    for t, ms in R['drop']:
        if t: E.append(ev(tf(t), 'info', 'logger', 'El logger perdió %d ms de datos' % ms, 'no es una falla de vuelo', 'PX4'))
    # correlación: salida de modo precedida de errores de comunicación
    for e in E:
        if e['cat'] == 'modo' and e['sev'] == 'warn':
            pre = [x for x in E if x['cat'] == 'comms' and 0 <= e['t'] - x['t'] <= 6]
            if pre: e['sev'] = 'crit'; e['detail'] += ' | precedido por: ' + '; '.join(x['title'] + (' (%s)' % x['detail'] if x['detail'] else '') for x in pre)
    for e in E:
        if e['cat'] == 'failsafe' and 'offboard_control_signal_lost' in e['title'] and any(x['cat'] == 'modo' and 0 <= e['t'] - x['t'] <= 6 for x in E):
            e['sev'] = 'warn'; e['detail'] += ' | consecuencia de salir de OFFBOARD (COM_OF_LOSS_T)'
    # series para gráficos
    P = []
    x = np.array([tf(a) for a in lp['timestamp']]); P.append(('Altitud (m)', 'PX4 altitud', x, -lp['z']))
    if bs is not None: P.append(('Batería', 'PX4 voltaje (V)', np.array([tf(a) for a in bs['timestamp']]), bs['voltage_v']))
    if am is not None:
        for j, k in enumerate(ks): P.append(('Motores (0-1)', 'PX4 motor %d' % (j + 1), tm, am[k]))
    # resumen
    dur = (w1 - w0) if win else None; S = []
    if win:
        S.append('Vuelo PX4: %d:%02d min en el aire, altura máx. %.0f m, distancia máx. desde el despegue %.0f m.' % (dur // 60, dur % 60, -lp['z'].min(), np.hypot(lp['x'] - lp['x'][0], lp['y'] - lp['y'][0]).max()))
    modes = [m for m in modes if not win or m[0] <= w1 + 1]
    if modes: S.append('Modos: ' + ' -> '.join(m for _, m in modes) + '.')
    if 'bat' in M: S.append('Batería: %.1f V -> %.1f V (%.0f%% -> %.0f%%).' % M['bat'])
    if 'trk' in M: S.append('Error máx. de seguimiento XY: %.2f m.' % M['trk'])
    F['summary'] = S; F['win'] = win
    return F, E, P

# ---------------------------------------------------------------- análisis Hivemind
def analyze_hm(R, name):
    v = topic(R, 'Vehicle'); tf = lambda ts: ts / 1e6
    F = dict(name=name, kind='Hivemind', size=R['n'], trunc=R['trunc'], repaired=R['repaired'], utc=True, uuid=str(R['info'].get('UniqueID') or '').strip(),
             hw=None, drops=[], logger=None)
    t = v['timestamp'] / 1e6; z = v['localPos[2]']; F['t0'], F['t1'] = float(t[0]), float(t[-1])
    zok = np.where(np.abs(z) < 300, z, np.nan); ia = np.where(zok > 1)[0]
    w0, w1 = (float(t[ia[0]]), float(t[ia[-1]])) if len(ia) else (None, None); F['win'] = (w0, w1) if w0 else None
    F['logger'] = next((m.group(1) for _, _, s in R['logs'] for m in [re.search(r'log/(\d{4}-\d\d-\d\d/\d+_\d+_\d+)\.ulg', s)] if m), None)
    E = log_events(R, tf, 'Hivemind', w0 if w0 else F['t1'], approx=True)
    if R['repaired']: E.append(ev(F['t0'], 'info', 'archivo', 'Cabecera dañada reparada', 'el Hivemind escribió %d mensajes INFO dentro de un mensaje de formato' % R['repaired'], 'Hivemind'))
    if R['trunc']: E.append(ev(F['t1'], 'crit', 'archivo', 'Log del Hivemind incompleto: termina aquí', 'el archivo se corta a mitad de un mensaje', 'Hivemind'))
    bad = np.where(np.abs(z) >= 300)[0]
    if len(bad): E.append(ev(t[bad[0]], 'info', 'posicion', 'Posición local inválida en tierra (z hasta %.0f m)' % np.nanmax(np.abs(z)), 'antes de tener buena posición', 'Hivemind'))
    inw = (t >= w0) & (t <= w1) if w0 else np.zeros(len(t), bool)
    for a, b in runs(t, np.r_[np.diff(t) > max(1.0, 8 * np.median(np.diff(t))), False], 0): E.append(ev(a, 'warn', 'comms', 'Hueco en datos del Hivemind', '%.1f s' % (b - a), 'Hivemind'))
    if inw.any():
        if v['battery'][inw].min() < 20: E.append(ev(t[inw][0], 'warn', 'bateria', 'Batería Hivemind < 20%', '', 'Hivemind'))
        if v['deviation'][inw].max() > 0.5: E.append(ev(t[inw][np.argmax(v['deviation'][inw])], 'warn', 'control', 'Desviación de posición > 0.5 m', 'máx %.2f m' % v['deviation'][inw].max(), 'Hivemind'))
        rs = topic(R, 'RSSI')
        if rs is not None:
            tr = rs['timestamp'] / 1e6; m = (tr >= w0) & (tr <= w1)
            if m.any() and rs['rssi'][m].min() < -85: E.append(ev(tr[m][np.argmin(rs['rssi'][m])], 'warn', 'comms', 'RSSI bajo en vuelo', '%.0f dBm' % rs['rssi'][m].min(), 'Hivemind'))
        vb = topic(R, 'VehicleBrokerStatus')
        if vb is not None:
            tb = vb['timestamp'] / 1e6; m = (tb >= w0) & (tb <= w1)
            if m.sum() > 1 and vb['messagesDropped'][m][-1] > vb['messagesDropped'][m][0]: E.append(ev(tb[m][0], 'warn', 'comms', 'Mensajes perdidos en el broker del vehículo', '%d en vuelo' % (vb['messagesDropped'][m][-1] - vb['messagesDropped'][m][0]), 'Hivemind'))
            if m.any() and vb['missedRTCM'][m].max() > 0: E.append(ev(tb[m][0], 'warn', 'gps', 'RTCM perdidos', 'máx %d' % vb['missedRTCM'][m].max(), 'Hivemind'))
        pp = topic(R, 'PathPlanner')
        if pp is not None:
            tp = pp['timestamp'] / 1e6; m = (tp >= w0) & (tp <= w1)
            if m.any() and pp['errorSpatial'][m].max() > 1: E.append(ev(tp[m][np.argmax(pp['errorSpatial'][m])], 'warn', 'control', 'Error espacial del planificador > 1 m', 'máx %.2f m' % pp['errorSpatial'][m].max(), 'Hivemind'))
    P = [('Altitud (m)', 'Hivemind altitud', t, zok)]
    S = ['Log Hivemind: %d:%02d min de datos (%s).' % ((F['t1'] - F['t0']) // 60, (F['t1'] - F['t0']) % 60, 'INCOMPLETO' if R['trunc'] else 'completo')]
    if w0: S.append('En el aire: %.0f s, altura máx. %.1f m.' % (w1 - w0, np.nanmax(zok)))
    F['summary'] = S
    return F, E, P

# ---------------------------------------------------------------- ¿mismo vuelo?
def compare(pa, ha, Pa, Ha):
    """pa/ha = metadatos de PX4 y Hivemind; Pa/Ha = (t, altura). Devuelve veredicto y evidencias."""
    C = []; ov = min(pa['t1'], ha['t1']) - max(pa['t0'], ha['t0'])
    uu = pa['uuid'] and ha['uuid'] and pa['uuid'].strip() == ha['uuid'].strip()
    C.append(('Mismo dron (UUID)', bool(uu), '%s / %s' % (pa['uuid'], ha['uuid'])))
    C.append(('Los tiempos se solapan (hora GPS/UTC)', ov > 0, 'solape %.0f s' % max(ov, 0)))
    if pa['logger'] and ha['logger']: C.append(('El Hivemind registró el nombre del log PX4', pa['logger'] == ha['logger'], '%s vs %s' % (pa['logger'], ha['logger'])))
    else: C.append(('El Hivemind registró el nombre del log PX4', None, 'no hay dato' if not ha['logger'] else 'el PX4 no lo trae'))
    err = None
    if ov > 0 and pa['win']:
        lo, hi = max(pa['win'][0], ha['t0']), min(pa['win'][1], ha['t1']); tt = np.arange(lo, hi, 0.5)
        if len(tt) > 10:
            hz = np.interp(tt, Ha[0][np.isfinite(Ha[1])], Ha[1][np.isfinite(Ha[1])]); pz = np.interp(tt, Pa[0], Pa[1]); err = float(np.mean(np.abs(hz - pz)))
    C.append(('Los perfiles de altura coinciden', None if err is None else err < 3, 'sin tramo en común' if err is None else 'error medio %.1f m' % err))
    name_ok = [c[1] for c in C if c[0].startswith('El Hivemind')][0]
    same = bool(name_ok) or (ov > 0 and err is not None and err < 3)
    lvl = 'same' if same else ('drone' if uu else 'diff')
    txt = {'same': 'SON DEL MISMO VUELO', 'drone': 'Mismo dron, pero NO son del mismo vuelo', 'diff': 'No parecen relacionados'}[lvl]
    cov = None
    if same and pa['win']:
        end = pa['win'][1]; cov = max(0.0, min(ha['t1'], end) - pa['win'][0]) / max(1.0, end - pa['win'][0])
    return dict(px4=pa['name'], hm=ha['name'], level=lvl, verdict=txt, checks=C, coverage=cov, hm_end=ha['t1'], hm_trunc=ha['trunc'])

# ---------------------------------------------------------------- punto de entrada
def analyze(files):
    """files: lista de (nombre, bytes). Devuelve JSON (texto)."""
    F, Ev, Pl, errors = [], [], {}, []
    for name, data in files:
        try:
            R = parse(data)
            if 'vehicle_status' in {v[0] for v in R['sub'].values()}: f, e, p = analyze_px4(R, name)
            elif any(v[0] == 'Vehicle' for v in R['sub'].values()): f, e, p = analyze_hm(R, name)
            else: errors.append('%s: ULog reconocido pero no es de PX4 ni Hivemind' % name); continue
        except Exception as ex: errors.append('%s: %s' % (name, ex)); continue
        f['events'] = e; f['series'] = p; F.append(f)
    pairs = []
    for a in [f for f in F if f['kind'] == 'PX4']:
        for b in [f for f in F if f['kind'] == 'Hivemind']:
            pa = next(s for s in a['series'] if s[0].startswith('Altitud')); hb = next(s for s in b['series'] if s[0].startswith('Altitud'))
            pairs.append(compare(a, b, (pa[2], pa[3]), (hb[2], hb[3])))
    # referencia de tiempo: despegue del primer PX4 (o del primer log)
    ref = next((f['win'][0] for f in F if f['kind'] == 'PX4' and f['win']), None) or next((f['win'][0] for f in F if f['win']), F[0]['t0'] if F else 0)
    for p in pairs:  # eventos críticos fuera de la cobertura del Hivemind
        if p['level'] == 'same':
            px = next(f for f in F if f['name'] == p['px4']); out = [e for e in px['events'] if e['sev'] == 'crit' and e['t'] > p['hm_end'] + 1]
            p['uncovered'] = [e['title'] for e in out]
    for f in F:
        for title, nm, x, y in f['series']:
            xs, ys = ds(np.asarray(x, float) - ref, np.asarray(y, float))
            Pl.setdefault(title, []).append(dict(name=nm, x=L(xs), y=L(ys)))
        f['series'] = []
        for e in f['events']: e['rel'] = e['t'] - ref
        f['events'].sort(key=lambda e: e['t']); f['win'] = [w - ref for w in f['win']] if f['win'] else None
        f['t0'] -= ref; f['t1'] -= ref; f['drops'] = [(d - ref if d else None, ms) for d, ms in f['drops']]
    for p in pairs: p['hm_end'] -= ref
    return json.dumps(dict(files=F, pairs=pairs, plots=Pl, ref=ref, errors=errors), default=lambda o: o.item() if hasattr(o, 'item') else str(o))

if __name__ == '__main__':  # uso local: python analyzer.py log1.ulg log2.ulog
    import sys
    r = json.loads(analyze([(p.split('/')[-1], open(p, 'rb').read()) for p in sys.argv[1:]]))
    print('errores:', r['errors'])
    for p in r['pairs']: print('>>', p['px4'], '<->', p['hm'], ':', p['verdict'], p['checks'])
    for f in r['files']:
        print('\n==', f['name'], f['kind']); [print(' ', s) for s in f['summary']]
        for e in f['events']:
            if e['sev'] != 'info': print('  %-4s t=%8.1f %s %s' % (e['sev'], e['rel'], e['title'], e['detail']))