"""Visualize actual ACT CNN and projected-image features on a verified rollout.

Replay saved observations with the original temporal ensemble and RNG state.
This is inference replay of the same successful execution, not a new rollout.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw
from .visualize_attention import _font,_overlay,_colors


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def feature_statistics(values,maximum):
    row,col=np.unravel_index(np.argmax(values),values.shape)
    return dict(minimum=float(values.min()),maximum=float(values.max()),mean=float(values.mean()),
                argmax_row=int(row),argmax_col=int(col),clipped_cells=int(np.sum(values>maximum)),
                total_cells=int(values.size))


def feature_tile(rgb,weights,maximum):
    tile=Image.fromarray(_overlay(rgb,weights,maximum)).resize((320,320),Image.Resampling.BILINEAR)
    row,col=np.unravel_index(np.argmax(weights),weights.shape)
    h,w=weights.shape
    ImageDraw.Draw(tile).rectangle((col*320/w,row*320/h,(col+1)*320/w-1,(row+1)*320/h-1),outline='white',width=2)
    return tile


def render_frame(data,row,scales,cameras,*,position,success_time):
    canvas=Image.new('RGB',(688,1328),(20,25,34));draw=ImageDraw.Draw(canvas)
    white,muted,accent='#edf1f7','#b5c1d0','#65d8cc'
    draw.text((16,12),'ACT | image encoder feature magnitude',font=_font(23),fill=white)
    draw.text((16,48),f"InD {position} | successful {success_time:.1f}s run | t = {row['time_s']:.1f}s | TE m = {row['coefficient']:g}",font=_font(16),fill=accent)
    for cam,key in enumerate(cameras):
        x=16+336*cam;name=key.rsplit('.',1)[-1];rgb=data[f'image_{cam}']
        draw.text((x,83),f'INPUT RGB / {name}',font=_font(18),fill=white)
        canvas.paste(Image.fromarray(rgb).resize((320,320)),(x,110))
        for stage,y,label in [('backbone',474,'CNN LAYER4 RMS'),('projected',862,'PROJECTED TOKEN RMS')]:
            weight=data[f'{stage}_rms_{cam}'];limit=scales[stage]['maximum']
            draw.text((x,y-27),f'{label} / {name}',font=_font(16),fill=accent)
            canvas.paste(feature_tile(rgb,weight,limit),(x,y))
    gradient=np.repeat(_colors(np.linspace(0,1,180))[None],12,axis=0).astype(np.uint8)
    for stage,y in [('backbone',808),('projected',1196)]:
        canvas.paste(Image.fromarray(gradient),(16,y))
        draw.text((208,y-5),f"0 to {scales[stage]['maximum']:.4g} RMS | fixed across cameras / time",font=_font(14),fill=muted)
    draw.text((16,1230),'White box = highest-RMS grid cell; not a learned keypoint.',font=_font(15),fill=white)
    draw.text((16,1258),f"Executed action: {row['contributors']} predictions | latest {row['latest_weight']:.1%} | history {1-row['latest_weight']:.1%}",font=_font(15),fill=accent)
    draw.text((16,1285),'Current-image features. Not attention probabilities or causal action importance.',font=_font(13),fill=muted)
    return np.asarray(canvas)


def decode_video(path):
    p=subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(path),'-map','0:v:0','-pix_fmt','rgb24','-f','framemd5','-'],capture_output=True,check=True)
    if p.stderr.strip():raise RuntimeError(p.stderr.decode())
    lines=[s for s in p.stdout.decode().splitlines() if s and not s.startswith('#')]
    return dict(frames=len(lines),framemd5_sha256=hashlib.sha256('\n'.join(lines).encode()).hexdigest(),sha256=digest(path))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--torch-threads',type=int,default=1)
    parser.add_argument('--display-percentile',type=float,default=99.5)
    args=parser.parse_args(argv)
    source=args.source_run.expanduser().resolve();out=args.output.expanduser().resolve()
    if out.exists():raise FileExistsError(out)
    if not 0<args.display_percentile<=100:raise ValueError('Display percentile must be in (0,100]')
    if args.torch_threads<1:raise ValueError('torch-threads must be positive')
    report=json.loads((source/'report.json').read_text());validation=json.loads((source/'validation.json').read_text())
    roll=report['rollout'];metadata=report['policy'];coefficient=metadata['temporal_ensemble_coeff']
    if not (validation['passed'] and roll['task_success'] and roll['stop_reason']=='success' and roll['runtime_error'] is None):raise ValueError('Require independently verified successful source rollout')
    if metadata['policy_type']!='act' or coefficient is None or metadata['n_action_steps']!=1:raise ValueError('Require an ACT source with temporal ensembling and n_action_steps=1')
    if metadata['requires_depth']:raise ValueError('This RGB visualization currently requires the RGB ACT source')
    if args.device!=metadata['device']:raise ValueError('Use original source device for exact action comparison')
    frames=report['attention']['frames'];chunks=report['attention']['chunks'];n=len(frames)
    if n!=len(chunks) or n!=roll['completed_control_steps']:raise ValueError('Source frames/chunks/actions mismatch')
    checkpoint=Path(metadata['checkpoint'])
    protected_paths=[source/'report.json',source/'validation.json',source/'rng_states.npz',source/'prediction_actions.npz',source/'rollout.mp4',source/'scene.xml']
    protected_paths += [source/c['weights'] for c in chunks]
    protected_paths += [p for p in checkpoint.iterdir() if p.is_file()]
    protected={str(p):digest(p) for p in protected_paths}
    with np.load(source/'rng_states.npz',allow_pickle=False) as z:rng={k:z[k].copy() for k in z.files}
    with np.load(source/'prediction_actions.npz',allow_pickle=False) as z:expected_predictions=z['normalized_actions'].copy()
    expected_actions=np.asarray([r['action'] for r in frames],dtype=np.float32)
    import torch
    import imageio.v2 as imageio
    from mujoco_tools.lerobot_policy import LeRobotPolicy
    from mujoco_tools.act_features import ACTFeatureTrace
    torch.set_num_threads(args.torch_threads)
    adapter=LeRobotPolicy(checkpoint,device=args.device,n_action_steps=1,temporal_ensemble_coeff=coefficient)
    adapter.reset();torch.set_rng_state(torch.from_numpy(rng['initial_cpu']))
    torch.cuda.set_rng_state(torch.from_numpy(rng['initial_cuda']),device=args.device)
    out.mkdir(parents=True);(out/'features').mkdir();(out/'previews').mkdir()
    output_predictions=[]
    def capture_prediction(module,inputs,output):
        output_predictions.append(output[0].detach().cpu().numpy().copy())
    prediction_hook=adapter.policy.model.register_forward_hook(capture_prediction)
    records=[];maps={'backbone':[],'projected':[]};all_actions=[];cameras=tuple(adapter.policy.config.image_features)
    try:
        with ACTFeatureTrace(adapter.policy.model) as trace:
            for step,(frame,chunk) in enumerate(zip(frames,chunks,strict=True)):
                if frame['chunk_index']!=step or chunk['source_step']!=step or frame['query_index']!=0 or frame['source_age_s']!=0:raise ValueError('Source is not current-image n1 inference')
                if tuple(chunk['cameras'])!=cameras:raise ValueError('Source camera order mismatch')
                with np.load(source/chunk['weights'],allow_pickle=False) as z:
                    obs={'observation.state':z['source_state'].copy()}
                    for ci,key in enumerate(cameras):obs[key]=z[f'image_{ci}'].copy()
                action=adapter.select_action(obs);all_actions.append(action.copy())
                if not np.array_equal(action,expected_actions[step]):raise AssertionError(f'Action differs from successful source at step {step}')
                if not np.array_equal(output_predictions[-1][0],expected_predictions[step]):raise AssertionError(f'64-step prediction differs at step {step}')
                if trace.capture_count!=step+1 or trace.latest is None:raise AssertionError('Missing feature capture')
                capture=trace.latest;arrays={'source_state':obs['observation.state'],'action':action,'time_s':np.array(frame['time_s'])}
                for ci,key in enumerate(cameras):
                    arrays[f'image_{ci}']=obs[key]
                    arrays[f'backbone_{ci}']=capture.camera_backbone_features[key][0]
                    arrays[f'projected_{ci}']=capture.camera_projected_features[key][0]
                    arrays[f'backbone_rms_{ci}']=capture.backbone_rms[key]
                    arrays[f'projected_rms_{ci}']=capture.projected_rms[key]
                    for stage in maps:maps[stage].append(arrays[f'{stage}_rms_{ci}'].copy())
                name=f'features/frame_{step:04d}.npz';np.savez_compressed(out/name,**arrays)
                te=frame['temporal_ensemble']
                records.append(dict(frame=step,time_s=frame['time_s'],source_frame=step,source_age_s=0.,features=name,
                                    grid_shapes={k:list(v) for k,v in capture.grid_shapes.items()},
                                    coefficient=coefficient,contributors=len(te['contributions']),latest_weight=te['current_prediction_weight'],
                                    source_npz_sha256=protected[str(source/chunk['weights'])]))
                if (step+1)%50==0:print(json.dumps({'event':'captured','frames':step+1,'total':n}),flush=True)
    finally:prediction_hook.remove()
    cpu_equal=np.array_equal(torch.get_rng_state().numpy(),rng['final_cpu'])
    cuda_equal=np.array_equal(torch.cuda.get_rng_state(device=args.device).cpu().numpy(),rng['final_cuda'])
    if not(cpu_equal and cuda_equal):raise AssertionError('RNG state differs from successful source')
    scales={}
    for stage,values in maps.items():
        joined=np.stack(values)
        maximum=float(np.percentile(joined,args.display_percentile))
        if not math.isfinite(maximum) or maximum<=0:raise ValueError('Invalid feature display scale')
        scales[stage]=dict(minimum=0.,maximum=maximum,percentile=args.display_percentile,
                           scope='all frames and both cameras for this layer',raw_minimum=float(joined.min()),raw_maximum=float(joined.max()),
                           clipped_cells=int(np.sum(joined>maximum)),total_cells=int(joined.size))
    with imageio.get_writer(str(out/'encoder_features.mp4'),fps=roll['control_hz'],codec='libx264',quality=8,macro_block_size=1,ffmpeg_log_level='error') as writer:
        for row in records:
            with np.load(out/row['features'],allow_pickle=False) as z:data={k:z[k] for k in z.files}
            row['statistics']={stage:{key:feature_statistics(data[f'{stage}_rms_{ci}'],scales[stage]['maximum']) for ci,key in enumerate(cameras)} for stage in maps}
            image=render_frame(data,row,scales,cameras,position=report['position'],success_time=roll['simulation_seconds']);writer.append_data(image)
            if row['frame']%10==0:Image.fromarray(image).save(out/'previews'/f"frame_{row['frame']:04d}.png")
    shutil.copy2(source/'rollout.mp4',out/'rollout.mp4')
    np.savez_compressed(out/'replayed_actions.npz',actions=np.stack(all_actions),normalized_predictions=np.concatenate(output_predictions,axis=0))
    for p,expected in protected.items():
        if digest(p)!=expected:raise AssertionError('Protected source changed: '+p)
    video=decode_video(out/'encoder_features.mp4')
    if video['frames']!=n:raise AssertionError('Feature video frame count mismatch')
    result=dict(scope='feature extraction from verified successful temporal-ensemble rollout via exact fixed-observation inference replay; no new physics rollout',
                source_run=str(source),source_report_sha256=protected[str(source/'report.json')],position=report['position'],seed=report['seed'],
                source_task_success=True,source_simulation_seconds=roll['simulation_seconds'],source_success_validation=str(source/'validation.json'),
                policy=adapter.metadata(),cameras=list(cameras),frames=records,display_scales=scales,
                method=dict(backbone_layer='ACT model.backbone layer4 feature_map',projection_layer='ACT model.encoder_img_feat_input_proj',
                            spatial_statistic='sqrt(mean_c(feature_c ** 2))',feature_storage='float32 [channels,height,width] before spatial flattening',
                            projection_timing='1x1 projection output before Transformer encoder mixing; positional embeddings travel separately',
                            native_grid='measured from actual forward',color_map='fixed blue/cyan/yellow/red; bilinear interpolation for display only',
                            boxes='highest RMS spatial grid cell, row-major argmax; not a trained keypoint',
                            temporal_ensemble='Original coefficient and complete history preserved; maps show current visual features, not a fused historical feature map',
                            interpretation='Feature amplitude, not SpatialSoftmax probability, action-query attention, object detection, or causal importance. Layer scales are separate; do not rank camera importance from amplitude.'),
                verification=dict(actions_bitwise_equal=True,full_64_predictions_bitwise_equal=True,final_cpu_rng_equal=cpu_equal,final_cuda_rng_equal=cuda_equal,
                                  source_and_checkpoint_unchanged=True,protected_sha256=protected,video=video),
                video='encoder_features.mp4',rollout_video='rollout.mp4')
    (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(out),'source_success':True,'frames':n,'actions_bitwise_equal':True,'display_scales':scales}),flush=True)


if __name__=='__main__':main()
