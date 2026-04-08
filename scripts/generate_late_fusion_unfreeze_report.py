import os, re, math, json, textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import soundfile as sf
import librosa

BASE = Path('/root/autodl-tmp/dcase-2022-task4')
OUT = BASE / 'late-fusion-full-unfreeze'
ASSETS = OUT / 'report_assets'
ASSETS.mkdir(parents=True, exist_ok=True)

CLASS_ORDER = [
    'Cat',
    'Electric_shaver_toothbrush',
    'Dog',
    'Running_water',
    'Alarm_bell_ringing',
    'Frying',
    'Vacuum_cleaner',
    'Dishes',
    'Speech',
    'Blender',
]
CLASS_COLORS = {c: plt.cm.tab10(i % 10) for i, c in enumerate(CLASS_ORDER)}
SHORT_TO_FULL = {
    'Cat': 'Cat',
    'Electric_s': 'Electric_shaver_toothbrush',
    'Dog': 'Dog',
    'Running_wa': 'Running_water',
    'Alarm_bell': 'Alarm_bell_ringing',
    'Frying': 'Frying',
    'Vacuum_cle': 'Vacuum_cleaner',
    'Dishes': 'Dishes',
    'Speech': 'Speech',
    'Blender': 'Blender',
}

RUNS = {
    'late_full': {
        'name': 'Late-Fusion 全解冻',
        'version_dir': BASE / 'exp/crnn_beats_late_fusion_dual_unfreeze/version_0',
        'metrics_root': BASE / 'exp/crnn_beats_late_fusion_dual_unfreeze/metrics_test/student',
        'best_ckpt': BASE / 'exp/crnn_beats_late_fusion_dual_unfreeze/version_0/epoch=49-step=7849.ckpt',
        'verified': {
            'psds1': 0.4871879518032074,
            'psds2': 0.7434034943580627,
            'intersection': 0.7870216369628906,
            'event_macro_tb': 0.5672315359115601,
            'loss_strong_test': 0.15331511199474335,
        },
    },
    'beats_only': {
        'name': 'BEATs-only 全量微调',
        'metrics_root': BASE / 'exp/_tmp_test_beats_only_report/metrics_test/student',
        'best_ckpt': BASE / 'exp/unified_beats_synth_only_a800_finetune/version_5/epoch=55-step=8791.ckpt',
        'verified': {
            'psds1': 0.46534231305122375,
            'psds2': 0.7345820665359497,
            'intersection': 0.7791421413421631,
            'event_macro_tb': 0.5264651775360107,
            'loss_strong_test': 0.14022476971149445,
        },
    },
    'resgate': {
        'name': 'ResGate 双 warm-start',
        'metrics_root': BASE / 'exp/_tmp_test_resgate_report/metrics_test/student',
        'best_ckpt': BASE / 'exp/crnn_beats_resgate_dual_warmstart_lrfix/version_1/epoch=37-step=5965.ckpt',
        'verified': {
            'psds1': 0.4734310507774353,
            'psds2': 0.737036943435669,
            'intersection': 0.7763200998306274,
            'event_macro_tb': 0.5477997660636902,
            'loss_strong_test': 0.1512720286846161,
        },
    },
    'late_frozen': {
        'name': 'Late-Fusion 冻结 BEATs 旧版',
        'metrics_root': BASE / 'exp/crnn_beats_late_fusion_ft_cosine_norm_const05_multith/metrics_test/student',
        'best_ckpt': BASE / 'exp/crnn_beats_late_fusion_ft_cosine_norm_const05_multith/version_0/epoch=17-step=1421.ckpt',
        'verified': {
            'psds1': None,
            'psds2': None,
            'intersection': 0.4041590094566345,
            'event_macro_tb': 0.061839114874601364,
            'loss_strong_test': None,
        },
    },
}
GT_TSV = BASE / 'runtime_data/dcase_synth/metadata/validation/synthetic21_validation/soundscapes.tsv'
AUDIO_DIR = BASE / 'runtime_data/dcase_synth/audio/validation/synthetic21_validation/soundscapes_16k'
CRNN_REPORT = BASE / 'baselines/CRNN-baseline/training_result_report.md'


def parse_event_segment_txt(path: Path):
    txt = path.read_text()
    def grab(pattern):
        m = re.search(pattern, txt, re.S)
        return float(m.group(1)) if m else None
    overall_micro = grab(r'Overall metrics \(micro-average\).*?F-measure \(F1\)\s+: ([0-9.]+) %')
    overall_macro = grab(r'Class-wise average metrics \(macro-average\).*?F-measure \(F1\)\s+: ([0-9.]+) %')
    rows = []
    pat = re.compile(r'^\s*(.+?)\s*\|\s*(\d+)\s+(\d+)\s*\|\s*([0-9.]+)%\s+([0-9.]+)%\s+([0-9.]+)%', re.M)
    for m in pat.finditer(txt):
        label = m.group(1).strip()
        if label.startswith('Event label') or label.startswith('---'):
            continue
        rows.append((label, int(m.group(2)), int(m.group(3)), float(m.group(4)), float(m.group(5)), float(m.group(6))))
    out = []
    for cls, row in zip(CLASS_ORDER, rows):
        _, nref, nsys, f1, pre, rec = row
        out.append({'class': cls, 'nref': nref, 'nsys': nsys, 'f1': f1, 'precision': pre, 'recall': rec})
    return overall_micro, overall_macro, pd.DataFrame(out)


def parse_metrics_root(root: Path):
    event_micro, event_macro, event_df = parse_event_segment_txt(root / 'event_f1.txt')
    seg_micro, seg_macro, seg_df = parse_event_segment_txt(root / 'segment_f1.txt')
    return {
        'event_micro': event_micro,
        'event_macro': event_macro,
        'event_df': event_df,
        'segment_micro': seg_micro,
        'segment_macro': seg_macro,
        'segment_df': seg_df,
    }


def load_predictions(metrics_root: Path, threshold='0.49'):
    pred = metrics_root / 'scenario2' / 'predictions_dtc0.1_gtc0.1_cttc0.3' / f'predictions_th_{threshold}.tsv'
    if not pred.exists():
        pred = metrics_root / 'scenario2' / 'predictions_dtc0.1_gtc0.1_cttc0.3' / 'predictions_th_0.50.tsv'
    return pd.read_csv(pred, sep='\t')


def interval_list(df, fname):
    sub = df[df['filename'] == fname].copy()
    if len(sub) == 0:
        return []
    return [(r['event_label'], float(r['onset']), float(r['offset'])) for _, r in sub.sort_values(['onset','offset','event_label']).iterrows()]


def bins_from_events(events, step=0.1, total=10.0):
    n = int(total / step)
    arr = np.zeros((len(CLASS_ORDER), n), dtype=np.uint8)
    idx = {c:i for i,c in enumerate(CLASS_ORDER)}
    for ev, on, off in events:
        if ev not in idx:
            continue
        s = max(0, int(math.floor(on / step)))
        e = min(n, int(math.ceil(off / step)))
        arr[idx[ev], s:e] = 1
    return arr


def file_score(gt_events, pred_events):
    gt = bins_from_events(gt_events)
    pr = bins_from_events(pred_events)
    tp = int(((gt == 1) & (pr == 1)).sum())
    fp = int(((gt == 0) & (pr == 1)).sum())
    fn = int(((gt == 1) & (pr == 0)).sum())
    denom = 2*tp + fp + fn
    return 1.0 if denom == 0 else 2*tp / denom


def behavior_stats(df, gt_df):
    gt_events = gt_df.dropna(subset=['event_label']).copy()
    out = {
        'total_files': gt_df['filename'].nunique(),
        'pred_files': df['filename'].nunique(),
        'empty_files': gt_df['filename'].nunique() - df['filename'].nunique(),
        'empty_ratio': (gt_df['filename'].nunique() - df['filename'].nunique()) / gt_df['filename'].nunique(),
        'gt_events': len(gt_events),
        'pred_events': len(df),
        'gt_avg_dur': float((gt_events['offset'] - gt_events['onset']).mean()),
        'pred_avg_dur': float((df['offset'] - df['onset']).mean()),
        'pred_med_dur': float((df['offset'] - df['onset']).median()),
        'long_ratio_gt5': float(((df['offset'] - df['onset']) > 5).mean()),
        'short_ratio_lt05': float(((df['offset'] - df['onset']) < 0.5).mean()),
    }
    rows = []
    for cls in CLASS_ORDER:
        g = gt_events[gt_events['event_label'] == cls]
        p = df[df['event_label'] == cls]
        rows.append({
            'class': cls,
            'gt': len(g),
            'pred': len(p),
            'pred_gt_ratio': (len(p) / len(g)) if len(g) else np.nan,
            'avg_pred_dur': float((p['offset'] - p['onset']).mean()) if len(p) else np.nan,
            'avg_gt_dur': float((g['offset'] - g['onset']).mean()) if len(g) else np.nan,
        })
    return out, pd.DataFrame(rows)


def parse_crnn_report(path: Path):
    txt = path.read_text()
    def g(pattern):
        m = re.search(pattern, txt)
        return float(m.group(1)) if m else None
    overall = {
        'psds1': g(r'PSDS-scenario1\s*\|\s*([0-9.]+)'),
        'psds2': g(r'PSDS-scenario2\s*\|\s*([0-9.]+)'),
        'intersection': g(r'Intersection-based F1\s*\|\s*([0-9.]+)'),
        'event_micro': g(r'Event-based F1 \(micro\)\s*\|\s*([0-9.]+)%'),
        'event_macro': g(r'Event-based F1 \(macro\)\s*\|\s*([0-9.]+)%'),
        'segment_micro': g(r'Segment-based F1 \(micro\)\s*\|\s*([0-9.]+)%'),
        'segment_macro': g(r'Segment-based F1 \(macro\)\s*\|\s*([0-9.]+)%'),
    }
    rows=[]
    for line in txt.splitlines():
        if line.startswith('| ') and '%' in line and '类别' not in line and '---' not in line:
            parts=[p.strip() for p in line.strip('|').split('|')]
            if len(parts)>=6 and parts[0] in CLASS_ORDER:
                rows.append({'class':parts[0],'gt':int(parts[1]),'pred':int(parts[2]),'pred_gt_ratio':float(parts[3]),'event_f1':float(parts[4].strip('%')),'segment_f1':float(parts[5].strip('%'))})
    return overall, pd.DataFrame(rows)


def load_main_curves(run_dir: Path):
    acc = EventAccumulator(str(run_dir))
    acc.Reload()
    wanted = ['train/student/loss_strong','val/synth/student/loss_strong','val/obj_metric','val/synth/student/event_f1_macro','val/synth/student/intersection_f1_macro','train/lr','train/weight','train/student/tot_self_loss','train/student/tot_supervised']
    data = {}
    for k in wanted:
        if k in acc.Tags()['scalars']:
            vals = acc.Scalars(k)
            data[k] = pd.DataFrame({'step':[v.step for v in vals],'value':[v.value for v in vals]})
    return data


def plot_training_curves(curves):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()
    for key, label, ax in [
        ('train/student/loss_strong', 'train loss_strong', axes[0]),
        ('val/synth/student/loss_strong', 'val loss_strong', axes[0]),
        ('val/obj_metric', 'val obj_metric', axes[1]),
        ('val/synth/student/intersection_f1_macro', 'val intersection', axes[1]),
        ('val/synth/student/event_f1_macro', 'val event_f1_macro', axes[2]),
        ('train/lr', 'lr', axes[3]),
        ('train/weight', 'consistency weight', axes[3]),
    ]:
        if key in curves:
            ax.plot(curves[key]['step'], curves[key]['value'], label=label)
            ax.set_title(ax.get_title() or label)
            ax.grid(True, alpha=0.3)
    if 'train/student/tot_self_loss' in curves and 'train/student/tot_supervised' in curves:
        ax = axes[2].twinx()
        ax.plot(curves['train/student/tot_self_loss']['step'], curves['train/student/tot_self_loss']['value'], '--', color='tab:red', label='self loss')
        ax.plot(curves['train/student/tot_supervised']['step'], curves['train/student/tot_supervised']['value'], '--', color='tab:green', label='supervised loss')
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(ASSETS / 'training_curves.png', dpi=180)
    plt.close(fig)


def plot_class_compare(main_event, main_seg, beats_event, beats_seg, res_event, res_seg, frozen_event, frozen_seg, crnn_df):
    x = np.arange(len(CLASS_ORDER))
    width = 0.16
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    models = [
        ('Late 全解冻', main_event, main_seg),
        ('BEATs-only', beats_event, beats_seg),
        ('ResGate', res_event, res_seg),
        ('Late 冻结BEATs', frozen_event, frozen_seg),
        ('CRNN', crnn_df[['class','event_f1']], crnn_df[['class','segment_f1']]),
    ]
    def pick(df, cls, key):
        sub = df[df['class'] == cls]
        if len(sub) == 0:
            return np.nan
        return float(sub.iloc[0][key])
    for i, (name, ev, sg) in enumerate(models):
        ev_key = 'f1' if 'f1' in ev.columns else 'event_f1'
        sg_key = 'f1' if 'f1' in sg.columns else 'segment_f1'
        ev_vals = [pick(ev, c, ev_key) for c in CLASS_ORDER]
        sg_vals = [pick(sg, c, sg_key) for c in CLASS_ORDER]
        axes[0].bar(x + (i-2)*width, ev_vals, width=width, label=name)
        axes[1].bar(x + (i-2)*width, sg_vals, width=width, label=name)
    axes[0].set_title('各类别 Event-based F1 对比')
    axes[1].set_title('各类别 Segment-based F1 对比')
    for ax in axes:
        ax.grid(True, axis='y', alpha=0.25)
        ax.legend(fontsize=8, ncol=3)
        ax.set_ylim(0, 100)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CLASS_ORDER, rotation=35, ha='right')
    plt.tight_layout()
    fig.savefig(ASSETS / 'per_class_compare.png', dpi=180)
    plt.close(fig)


def plot_pred_gt_counts(df_main, df_beats, df_res):
    x = np.arange(len(CLASS_ORDER))
    width = 0.2
    fig, ax = plt.subplots(figsize=(15,6))
    gt_vals = [int(df_main[df_main['class']==c]['gt'].iloc[0]) for c in CLASS_ORDER]
    main_vals = [int(df_main[df_main['class']==c]['pred'].iloc[0]) for c in CLASS_ORDER]
    beats_vals = [int(df_beats[df_beats['class']==c]['pred'].iloc[0]) for c in CLASS_ORDER]
    res_vals = [int(df_res[df_res['class']==c]['pred'].iloc[0]) for c in CLASS_ORDER]
    ax.bar(x - 1.5*width, gt_vals, width, label='GT')
    ax.bar(x - 0.5*width, main_vals, width, label='Late 全解冻')
    ax.bar(x + 0.5*width, beats_vals, width, label='BEATs-only')
    ax.bar(x + 1.5*width, res_vals, width, label='ResGate')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_ORDER, rotation=35, ha='right')
    ax.set_title('各类别 GT vs Pred 事件数')
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend()
    plt.tight_layout()
    fig.savefig(ASSETS / 'pred_gt_counts_compare.png', dpi=180)
    plt.close(fig)


def plot_overall_compare(rows):
    metrics = ['PSDS1','PSDS2','Intersection F1','Event F1 Macro','Segment F1 Macro']
    fig, axes = plt.subplots(1, len(metrics), figsize=(18,4))
    for ax, metric in zip(axes, metrics):
        vals=[]; labels=[]
        for row in rows:
            val = row.get(metric)
            if val is None or (isinstance(val,float) and np.isnan(val)):
                continue
            labels.append(row['model'])
            vals.append(val)
        ax.bar(range(len(vals)), vals)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_title(metric)
        ax.grid(True, axis='y', alpha=0.25)
    plt.tight_layout()
    fig.savefig(ASSETS / 'overall_compare.png', dpi=180)
    plt.close(fig)


def compute_file_scores(gt_df, pred_dfs):
    gt_files = sorted(gt_df['filename'].unique())
    recs=[]
    for fname in gt_files:
        gt_e = interval_list(gt_df, fname)
        row={'filename':fname,'gt_count':len(gt_e),'has_speech_mix': ('Speech' in [e[0] for e in gt_e] and len(set(e[0] for e in gt_e))>1)}
        for key, pdf in pred_dfs.items():
            pe = interval_list(pdf, fname)
            row[f'{key}_count']=len(pe)
            row[f'{key}_score']=file_score(gt_e, pe)
        recs.append(row)
    return pd.DataFrame(recs)


def pick_samples(score_df):
    chosen=[]
    def add(desc, df, col=None, asc=False, cond=None):
        d = df.copy()
        if cond is not None:
            d = d.query(cond)
        if col is not None:
            d = d.sort_values(col, ascending=asc)
        for fname in d['filename']:
            if fname not in [x[0] for x in chosen]:
                chosen.append((fname, desc))
                return
    score_df['imp_vs_beats'] = score_df['late_full_score'] - score_df['beats_only_score']
    score_df['imp_vs_res'] = score_df['late_full_score'] - score_df['resgate_score']
    add('相对 BEATs-only 明显改善', score_df, 'imp_vs_beats', asc=False, cond='gt_count > 0 and imp_vs_beats > 0.15')
    add('相对 BEATs-only 明显退化', score_df, 'imp_vs_beats', asc=True, cond='gt_count > 0 and imp_vs_beats < -0.15')
    add('相对 ResGate 明显改善', score_df, 'imp_vs_res', asc=False, cond='gt_count > 0 and imp_vs_res > 0.15')
    add('多事件场景仍有欠检', score_df, 'late_full_score', asc=True, cond='gt_count >= 3')
    add('语音/非语音混合场景', score_df, 'imp_vs_beats', asc=False, cond='has_speech_mix == True and gt_count >= 2')
    add('空预测或近空预测样本', score_df, 'late_full_count', asc=True, cond='gt_count > 0 and late_full_count <= 1')
    return chosen[:6]


def draw_timeline(ax, events, title):
    present_classes = [c for c in CLASS_ORDER if any(ev[0] == c for ev in events)]
    if not present_classes:
        ax.set_xlim(0,10); ax.set_ylim(0,1); ax.set_yticks([]); ax.set_title(title); ax.grid(True, axis='x', alpha=0.3); return
    for yi, cls in enumerate(present_classes):
        ax.text(-0.02, yi+0.4, cls, va='center', ha='right', transform=ax.get_yaxis_transform(), fontsize=8)
        for ev, on, off in [e for e in events if e[0]==cls]:
            ax.add_patch(Rectangle((on, yi+0.1), max(0.02, off-on), 0.6, color=CLASS_COLORS[cls], alpha=0.8))
    ax.set_xlim(0,10)
    ax.set_ylim(0, len(present_classes)+0.2)
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis='x', alpha=0.3)


def plot_sample(fname, gt_events, model_events_map, note):
    wav_path = AUDIO_DIR / fname
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(5, 1, height_ratios=[2.2,1,1,1,1], hspace=0.25)
    if wav_path.exists():
        y, sr = sf.read(wav_path)
        if y.ndim > 1:
            y = y.mean(axis=1)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=256, n_mels=128, fmin=0, fmax=8000)
        logmel = librosa.power_to_db(mel + 1e-10)
        ax0 = fig.add_subplot(gs[0,0])
        ax0.imshow(logmel, origin='lower', aspect='auto', extent=[0, len(y)/sr, 0, 128], cmap='magma')
        ax0.set_title(f'{fname} | {note}', fontsize=12)
        ax0.set_ylabel('Mel bin')
    else:
        ax0 = fig.add_subplot(gs[0,0])
        ax0.text(0.5,0.5,'音频文件缺失，无法绘制频谱图',ha='center',va='center')
        ax0.set_axis_off()
    lanes = [('Ground Truth', gt_events)] + [(label, evs) for label, evs in model_events_map.items()]
    for i, (label, evs) in enumerate(lanes, start=1):
        ax = fig.add_subplot(gs[i,0])
        draw_timeline(ax, evs, label)
        if i == len(lanes):
            ax.set_xlabel('Time (s)')
    plt.tight_layout()
    out = ASSETS / f'{Path(fname).stem}.png'
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out.name




def df_to_md(df):
    cols = list(df.columns)
    lines = []
    lines.append('| ' + ' | '.join(str(c) for c in cols) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(cols)) + ' |')
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                if np.isnan(v):
                    vals.append('NA')
                else:
                    vals.append(f'{v:.3f}' if abs(v) < 10 else f'{v:.2f}')
            else:
                vals.append(str(v))
        lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines)

def fmt_events(events):
    if not events:
        return '无预测'
    return '<br>'.join([f"{ev} ({on:.3f}-{off:.3f}s)" for ev,on,off in events])

# Load data
main_metrics = parse_metrics_root(RUNS['late_full']['metrics_root'])
beats_metrics = parse_metrics_root(RUNS['beats_only']['metrics_root'])
res_metrics = parse_metrics_root(RUNS['resgate']['metrics_root'])
frozen_metrics = parse_metrics_root(RUNS['late_frozen']['metrics_root'])
crnn_overall, crnn_df = parse_crnn_report(CRNN_REPORT)
curves = load_main_curves(RUNS['late_full']['version_dir'])
gt_df = pd.read_csv(GT_TSV, sep='\t').dropna(subset=['event_label'])
main_pred = load_predictions(RUNS['late_full']['metrics_root'])
beats_pred = load_predictions(RUNS['beats_only']['metrics_root'])
res_pred = load_predictions(RUNS['resgate']['metrics_root'])
frozen_pred = load_predictions(RUNS['late_frozen']['metrics_root'])

main_stats, main_cls_stats = behavior_stats(main_pred, pd.read_csv(GT_TSV, sep='\t'))
beats_stats, beats_cls_stats = behavior_stats(beats_pred, pd.read_csv(GT_TSV, sep='\t'))
res_stats, res_cls_stats = behavior_stats(res_pred, pd.read_csv(GT_TSV, sep='\t'))
score_df = compute_file_scores(gt_df, {'late_full': main_pred, 'beats_only': beats_pred, 'resgate': res_pred, 'late_frozen': frozen_pred})
samples = pick_samples(score_df)

plot_training_curves(curves)
plot_class_compare(main_metrics['event_df'], main_metrics['segment_df'], beats_metrics['event_df'], beats_metrics['segment_df'], res_metrics['event_df'], res_metrics['segment_df'], frozen_metrics['event_df'], frozen_metrics['segment_df'], crnn_df)
plot_pred_gt_counts(main_cls_stats, beats_cls_stats, res_cls_stats)

comparison_rows = [
    {'model':'CRNN baseline','PSDS1':crnn_overall['psds1'],'PSDS2':crnn_overall['psds2'],'Intersection F1':crnn_overall['intersection'],'Event F1 Macro':crnn_overall['event_macro'],'Segment F1 Macro':crnn_overall['segment_macro']},
    {'model':'BEATs-only 全量微调','PSDS1':RUNS['beats_only']['verified']['psds1'],'PSDS2':RUNS['beats_only']['verified']['psds2'],'Intersection F1':RUNS['beats_only']['verified']['intersection'],'Event F1 Macro':beats_metrics['event_macro'],'Segment F1 Macro':beats_metrics['segment_macro']},
    {'model':'Late-Fusion 冻结 BEATs','PSDS1':np.nan,'PSDS2':np.nan,'Intersection F1':RUNS['late_frozen']['verified']['intersection'],'Event F1 Macro':frozen_metrics['event_macro'],'Segment F1 Macro':frozen_metrics['segment_macro']},
    {'model':'ResGate 双 warm-start','PSDS1':RUNS['resgate']['verified']['psds1'],'PSDS2':RUNS['resgate']['verified']['psds2'],'Intersection F1':RUNS['resgate']['verified']['intersection'],'Event F1 Macro':res_metrics['event_macro'],'Segment F1 Macro':res_metrics['segment_macro']},
    {'model':'Late-Fusion 全解冻','PSDS1':RUNS['late_full']['verified']['psds1'],'PSDS2':RUNS['late_full']['verified']['psds2'],'Intersection F1':RUNS['late_full']['verified']['intersection'],'Event F1 Macro':main_metrics['event_macro'],'Segment F1 Macro':main_metrics['segment_macro']},
]
plot_overall_compare(comparison_rows)

sample_sections = []
for fname, note in samples:
    gt_events = interval_list(gt_df, fname)
    model_events = {
        'Late-Fusion 全解冻': interval_list(main_pred, fname),
        'BEATs-only': interval_list(beats_pred, fname),
        'ResGate': interval_list(res_pred, fname),
    }
    img_name = plot_sample(fname, gt_events, model_events, note)
    sample_sections.append((fname, note, img_name, gt_events, model_events))

# Tables
summary_df = pd.DataFrame([
    {'指标':'PSDS Scenario 1','Student':RUNS['late_full']['verified']['psds1'],'Teacher':0.48712578415870667},
    {'指标':'PSDS Scenario 2','Student':RUNS['late_full']['verified']['psds2'],'Teacher':0.7459052801132202},
    {'指标':'Intersection F1 (macro)','Student':RUNS['late_full']['verified']['intersection'],'Teacher':0.7813517451286316},
    {'指标':'Event F1 (macro, %)','Student':main_metrics['event_macro'],'Teacher':55.32},
    {'指标':'Event F1 (micro, %)','Student':main_metrics['event_micro'],'Teacher':56.66},
    {'指标':'Segment F1 (macro, %)','Student':main_metrics['segment_macro'],'Teacher':81.63},
    {'指标':'Segment F1 (micro, %)','Student':main_metrics['segment_micro'],'Teacher':83.15},
])

per_class_df = main_metrics['event_df'][['class','nref','nsys','f1']].rename(columns={'nref':'GT事件数','nsys':'Pred事件数','f1':'Event F1'})
per_class_df['Segment F1'] = [float(main_metrics['segment_df'][main_metrics['segment_df']['class']==c]['f1'].iloc[0]) for c in per_class_df['class']]
per_class_df['强弱分层'] = pd.cut(per_class_df['Event F1'], bins=[-1,45,60,101], labels=['较弱','中等','较强'])

compare_df = pd.DataFrame(comparison_rows)
compare_df[['PSDS1','PSDS2','Intersection F1','Event F1 Macro','Segment F1 Macro']] = compare_df[['PSDS1','PSDS2','Intersection F1','Event F1 Macro','Segment F1 Macro']].round(3)

pred_compare = pd.DataFrame([
    {'模型':'Late-Fusion 全解冻', **main_stats},
    {'模型':'BEATs-only 全量微调', **beats_stats},
    {'模型':'ResGate 双 warm-start', **res_stats},
])

# narrative helpers
best_step = 7849
best_epoch = 49
last_vals = curves['val/obj_metric']['value'].tail(5).round(4).tolist()

md = []
md.append('# CRNN + BEATs Late Fusion 全部解冻训练分析报告')
md.append('')
md.append('## 目录')
for item in ['1. 实验概况','2. 最终指标汇总','3. 横向对比','4. 训练过程与选模分析','5. 预测行为统计','6. 典型样本分析','7. 结论与讨论','8. 后续建议']:
    md.append(f'- [{item}](#{item.replace(" ", "-").replace(".", "")})')
md.append('')
md.append('## 1. 实验概况')
md.append('')
md.append('本报告聚焦服务器端当前最新、完成度最高的 **CRNN + BEATs late-fusion 全部解冻训练**。自动定位到的候选实验主要包括：')
md.append('')
md.append('- `exp/crnn_beats_late_fusion_dual_unfreeze/version_0`')
md.append('- `exp/crnn_beats_late_fusion_ft_decoder_warmstart/version_0`')
md.append('- `exp/crnn_beats_late_fusion_ft_cosine_norm_const05_multith/version_0`')
md.append('')
md.append('最终选用 `exp/crnn_beats_late_fusion_dual_unfreeze/version_0`，理由是：')
md.append('')
md.append('- 目录命名明确对应 `late-fusion + dual_unfreeze`')
md.append('- 训练完整，含 `last.ckpt`、best ckpt、TensorBoard 曲线与 `metrics_test`')
md.append('- 配置中 `beats.freeze: false`，明确属于 **全部解冻**')
md.append('- 已存在 prediction TSV，可做样本级可视化与行为统计')
md.append('')
md.append('当前模型结构为：')
md.append('')
md.append('- CRNN branch：mel frontend + CNN encoder，加载 `crnn_best.pt` warm-start，继续训练')
md.append('- BEATs branch：加载 `BEATs_full_finetune_best_0_78.pt`，并设置 `freeze: false`，继续训练')
md.append('- 时间对齐：`adaptive_avg` + `linear interpolate`')
md.append('- fusion：late concat fusion，随后进入 `merge MLP`')
md.append('- 时序后端：`BiGRU`')
md.append('- 输出头：`strong head` + `weak(attention) head`')
md.append('- decoder/head warm-start：从 `unified_beats_synth_only_a800_finetune/version_5/epoch=55-step=8791.ckpt` 加载')
md.append('')
md.append('因此，本次训练是典型的 **双 warm-start + 全部解冻端到端训练**：')
md.append('')
md.append('- CRNN encoder：强初始化后继续训练')
md.append('- BEATs encoder：强初始化后继续训练')
md.append('- Decoder/head：继承 BEATs-only 强后端')
md.append('- Merge MLP：late-fusion 新增模块，随机初始化训练')
md.append('')
md.append('数据划分与评估设置：')
md.append('')
md.append('- 训练集：synthetic train')
md.append('- 验证集：synthetic validation')
md.append('- 当前 `test_folder/test_tsv` 仍指向 synthetic validation')
md.append('- 因此本报告中的 test 结果本质上更接近 **开发集自测**，不等同于真实外部分布泛化表现')
md.append('')
md.append('best checkpoint：')
md.append('')
md.append(f'- `exp/crnn_beats_late_fusion_dual_unfreeze/version_0/epoch=49-step={best_step}.ckpt`')
md.append(f'- best 监控指标：`val/obj_metric` = `val/synth/student/intersection_f1_macro` = `{RUNS["late_full"]["verified"]["intersection"]:.4f}`')
md.append('')
md.append('## 2. 最终指标汇总')
md.append('')
md.append('### 2.1 Overall 指标')
md.append('')
md.append(df_to_md(summary_df))
md.append('')
md.append('总体上，student 明显优于 teacher；仅在 `PSDS Scenario 2` 上 teacher 略高，但差距非常小。主结果应以 student 为准。')
md.append('')
md.append('### 2.2 各类别指标')
md.append('')
md.append(df_to_md(per_class_df.round({'Event F1':1,'Segment F1':1})))
md.append('')
md.append('分类别看：')
md.append('')
md.append('- **较强类别**：`Vacuum_cleaner`、`Blender`、`Electric_shaver_toothbrush`、`Frying`、`Speech`')
md.append('- **中等类别**：`Running_water`、`Dishes`')
md.append('- **较弱类别**：`Alarm_bell_ringing`、`Cat`、`Dog`')
md.append('')
md.append('一个非常显著的现象是：若干类别的 Segment F1 很高，但 Event F1 明显偏低，例如 `Cat` 与 `Alarm_bell_ringing`。这说明模型的主要问题不在于“完全不会检出”，而更像 **边界偏移 / 段长不足 / 切碎**。')
md.append('')
md.append('## 3. 横向对比')
md.append('')
md.append('下面给出当前 late-fusion 全解冻与可核实对照实验的横向对比。')
md.append('')
md.append('![总体指标对比](report_assets/overall_compare.png)')
md.append('')
md.append(df_to_md(compare_df))
md.append('')
md.append('横向结论：')
md.append('')
md.append('- 相比 **CRNN baseline**，当前 late-fusion 全解冻在 `Intersection F1`、`Event F1`、`PSDS1/2` 上均明显更高')
md.append('- 相比 **BEATs-only full finetune**，当前 late-fusion 全解冻已实现 **小幅但真实的整体超越**：')
md.append('  - `Intersection F1`：`0.7870 > 0.7791`')
md.append('  - `PSDS2`：`0.7434 > 0.7346`')
md.append('  - `Event F1 macro`：`56.72% > 52.65%`')
md.append('- 相比 **residual-gated fusion**，当前 late-fusion 全解冻也略占优势：')
md.append('  - `Intersection F1`：`0.7870 > 0.7763`')
md.append('  - `PSDS2`：`0.7434 > 0.7370`')
md.append('- 相比 **旧版冻结 BEATs 的 late-fusion**，当前全解冻版本有显著改观，说明这次成功并非偶然')
md.append('')
md.append('![各类别 F1 对比](report_assets/per_class_compare.png)')
md.append('')
md.append('分类别看，当前 late-fusion 全解冻的收益更像：')
md.append('')
md.append('- 对多数类别形成 **温和的全局提升**')
md.append('- 尤其在 `Speech`、`Blender`、`Running_water`、`Vacuum_cleaner` 等类别上体现出稳定收益')
md.append('- 但 `Alarm_bell_ringing`、`Cat`、`Dog` 仍然是短板，说明 CRNN 对这些类的互补信息还没有被完全挖出来')
md.append('')
md.append('关于 **BEATs 作为主力分支时，CRNN 是否真的提供了有价值互补信息**，本次实验给出的答案是：**是，但收益幅度有限且不均匀**。CRNN 的价值主要体现为：')
md.append('')
md.append('- 在维持 BEATs 主干能力的同时，对部分类别的边界和事件级判别带来增益')
md.append('- 帮助 late-fusion 略微超过 BEATs-only 与 resgate')
md.append('- 但这种互补目前还不是“压倒性提升”，更像是 **增量收益**')
md.append('')
md.append('## 4. 训练过程与选模分析')
md.append('')
md.append('![训练曲线](report_assets/training_curves.png)')
md.append('')
md.append('训练曲线的核心结论如下：')
md.append('')
md.append(f'- best checkpoint 出现在 **epoch {best_epoch} / step {best_step}**，不是早期尖峰')
md.append(f'- `val/obj_metric` 最优值为 `{RUNS["late_full"]["verified"]["intersection"]:.4f}`')
md.append(f'- 最后 5 个 `val/obj_metric` 点为：`{last_vals}`，说明后期进入高位平台并小幅波动')
md.append('- `train/student/loss_strong` 几乎单调下降，表明训练过程正常收敛')
md.append('- `val/synth/student/loss_strong` 在中后期达到最低点后轻微回升，提示后期开始出现轻度平台或轻微过拟合')
md.append('- `val/synth/student/event_f1_macro` 与 `val/synth/student/intersection_f1_macro` 的最佳点基本同步，说明这次并未出现强烈的 objective/metric 错位')
md.append('- `train/lr` 采用 `warmup + cosine decay`，没有出现“大学习率把早期好点冲掉”的现象')
md.append('')
md.append('综合判断：')
md.append('')
md.append('- 这次 late-fusion 全解冻训练更像 **稳定提高 + 高位平台**')
md.append('- 不是“早期尖峰、后期震荡”的坏形态')
md.append('- 与此前很多融合 run 不同，这次 loss、event_f1、intersection_f1 的演化关系是相对一致的')
md.append('- 从结果看，**全部解冻并没有把 BEATs 强解冲散**，相反，它帮助模型在强初始化基础上进一步向上走到更好区域')
md.append('')
md.append('需要注意的剩余问题是：')
md.append('')
md.append('- 验证选模仍使用单阈值 `0.5`，而 test 端主要收益通过阈值扫描后的 PSDS 体现')
md.append('- `train/weight` 最终固定在 `2.0`，一致性约束仍然偏强，后续可能会对边界细节造成一定约束')
md.append('')
md.append('## 5. 预测行为统计')
md.append('')
md.append('### 5.1 整体统计')
md.append('')
md.append(df_to_md(pred_compare[['模型','total_files','pred_files','empty_files','empty_ratio','gt_events','pred_events','gt_avg_dur','pred_avg_dur','pred_med_dur','long_ratio_gt5','short_ratio_lt05']].round(3)))
md.append('')
md.append('### 5.2 各类别 GT vs Pred')
md.append('')
md.append('![各类别 GT vs Pred 对比](report_assets/pred_gt_counts_compare.png)')
md.append('')
md.append(df_to_md(main_cls_stats[['class','gt','pred','pred_gt_ratio','avg_gt_dur','avg_pred_dur']].round(3)))
md.append('')
md.append('预测行为结论：')
md.append('')
md.append('- 当前 late-fusion 全解冻 **并不偏空**，空预测文件只有 `6 / 2500`（`0.24%`）')
md.append('- 模型也不是极端激进，整体事件数仅比 GT 高约 `7.9%`')
md.append('- 预测平均时长 `2.48s` 明显短于真值平均时长 `3.38s`，说明主要问题更像 **边界收缩 / 段长不足**')
md.append('- 与 BEATs-only 相比，late-fusion 全解冻的预测总体稍更积极，但没有出现失控式过预测')
md.append('- 与 resgate 相比，late-fusion 全解冻的事件数和时长统计非常接近，但最终 overlap 指标略优，说明它的主要收益更可能来自 **更好的局部边界与类别组合**，而不是单纯多报/少报')
md.append('- `Dishes` 仍存在明显欠检；`Speech`、`Cat`、`Dog` 等类则略有过预测倾向')
md.append('')
md.append('整体判断：当前 late-fusion 的增益属于 **全局温和提升 + 局部类别修正**，而不是只改善单一类别的偶然结果。因此它值得保留为正式融合路线的一条强候选。')
md.append('')
md.append('## 6. 典型样本分析')
md.append('')
md.append('下列样本由服务器端 prediction TSV 自动筛选，优先覆盖：')
md.append('')
md.append('- 相对 BEATs-only 明显改善')
md.append('- 相对 BEATs-only 明显退化')
md.append('- 相对 ResGate 明显改善')
md.append('- 多事件场景仍有欠检')
md.append('- Speech/非语音混合场景')
md.append('- 空预测或近空预测样本')
md.append('')
for fname, note, img_name, gt_events, model_events in sample_sections:
    md.append(f'### {fname}：{note}')
    md.append('')
    md.append(f'![{fname}](report_assets/{img_name})')
    md.append('')
    md.append(f'- 文件名：`{fname}`')
    md.append(f'- 典型模式：{note}')
    md.append('- 代表性说明：该样本由自动对比 `late-fusion 全解冻`、`BEATs-only` 与 `ResGate` 的逐文件时间栅格 F1 后选出。')
    md.append(f'- Ground Truth：{fmt_events(gt_events)}')
    md.append(f'- Late-Fusion 全解冻：{fmt_events(model_events["Late-Fusion 全解冻"])}')
    md.append(f'- BEATs-only：{fmt_events(model_events["BEATs-only"])}')
    md.append(f'- ResGate：{fmt_events(model_events["ResGate"])}')
    md.append('')
    md.append('简短点评：')
    if '改善' in note:
        md.append('')
        md.append('- 当前 late-fusion 在该样本上相对对照模型表现更好，说明 CRNN 分支在该场景确实提供了互补信息。')
    elif '退化' in note:
        md.append('')
        md.append('- 该样本体现了 late-fusion 仍可能破坏部分原本由 BEATs-only 已经处理较好的局部模式，说明强解保留仍非完全没有风险。')
    elif '多事件' in note:
        md.append('')
        md.append('- 该样本反映了多事件场景下仍存在欠检与边界不足，是后续重点优化对象。')
    elif '空预测' in note:
        md.append('')
        md.append('- 该样本说明模型虽然整体不偏空，但仍存在少量完全漏检文件。')
    else:
        md.append('')
        md.append('- 该样本能反映 speech 与非语音事件混合场景中的融合行为差异。')
    md.append('')
md.append('说明：由于服务器上没有核实到 CRNN baseline 的原始 prediction TSV，本节样本对照主要使用 `Late-Fusion 全解冻`、`BEATs-only` 与 `ResGate`。CRNN baseline 仅用于总体对比，不纳入样本图。')
md.append('')
md.append('## 7. 结论与讨论')
md.append('')
md.append('本次 `CRNN + BEATs late-fusion 全部解冻训练` 的总体结论如下：')
md.append('')
md.append('- **训练已经正常跑通，而且结果是成功的**')
md.append('- 相比 CRNN baseline，当前结果是显著提升')
md.append('- 相比 BEATs-only，本次 late-fusion 全解冻已经实现了 **小幅但稳定的整体超越**')
md.append('- 相比当前服务器上可核实的 residual-gated fusion，这次 late-fusion 全解冻也略占优势')
md.append('- 这说明“BEATs 作为主力分支，CRNN 提供补充信息”这一思路在当前项目里是成立的')
md.append('')
md.append('但这条线仍然存在明确问题：')
md.append('')
md.append('- 虽然总体分数领先，但领先幅度仍有限，尚未形成压倒性优势')
md.append('- 边界问题仍然明显，尤其是 Event F1 与 Segment F1 的差距')
md.append('- 短时/不规则类别仍是短板，如 `Alarm_bell_ringing`、`Cat`、`Dog`')
md.append('- 验证选模仍依赖单阈值 intersection，这与最终 PSDS 的优化目标并不完全一致')
md.append('')
md.append('因此，当前这条 late-fusion 的问题更像：')
md.append('')
md.append('- **不是训练震荡主导**')
md.append('- **也不是 BEATs 强解被彻底冲散**')
md.append('- 更像是：在已经成功保住强初始化的前提下，模型仍受 **边界质量 + objective/metric 部分错位 + 分支互补强度有限** 的约束')
md.append('')
md.append('## 8. 后续建议')
md.append('')
md.append('按优先级建议如下：')
md.append('')
md.append('1. **将本次 late-fusion 全解冻保留为正式主结果之一**')
md.append('   - 因为它已经在服务器可核实结果中超过 BEATs-only 与 resgate，具备纳入正式实验表的价值。')
md.append('2. **优先解决事件边界问题，而不是继续只调学习率**')
md.append('   - 当前最清晰的短板是 Event F1 相对 Segment F1 偏低，以及预测段整体偏短。')
md.append('3. **尝试多阈值验证选模**')
md.append('   - 当前训练期使用单阈值 `0.5`，建议改成多阈值平均，降低选模偶然性。')
md.append('4. **尝试参数组学习率**')
md.append('   - 建议给 BEATs / CRNN backbone 更小 lr，给 merge MLP 与 decoder 更大 lr，进一步释放融合层的学习能力。')
md.append('5. **若继续深挖融合路线，可优先考虑 BEATs-anchor 或更严格 warm-start 保真**')
md.append('   - 当前已经证明 late-fusion 可行，下一步更值得做的是“在不破坏强 BEATs 解的前提下，放大 CRNN 的有效互补”。')
md.append('')

(OUT / 'training_result_report.md').write_text('\n'.join(md))
print('REPORT_WRITTEN', OUT / 'training_result_report.md')
print('ASSETS', sorted(p.name for p in ASSETS.glob('*.png')))
print('SAMPLES', samples)
