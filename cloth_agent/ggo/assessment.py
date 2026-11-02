"""Auditable body-section curves and finite pose-perturbation sensitivity."""
import numpy as np


def read_grid(grid, xy, origin, step):
    ij=np.floor((xy-origin)/step).astype(int)
    ok=(ij[:,0]>=0)&(ij[:,0]<grid.shape[1])&(ij[:,1]>=0)&(ij[:,1]<grid.shape[0])
    value=np.zeros(len(xy),bool);value[ok]=grid[ij[ok,1],ij[ok,0]]
    return value,ok


def section_edges(mask,observed,u,v,frame,origin,cfg,R,t):
    """Require a single observed cloth interval and measured cloth/background transitions.

    No nearest-contour search, no interpolation through missing observations.
    """
    rows=[];support=max(2,int(np.ceil(cfg.boundary_support_mm/cfg.grid_mm)))
    for offset in (-cfg.grid_mm,0.,cfg.grid_mm):
        xy=frame['center']+u[:,None]*frame['ex']+(v+offset)*frame['ey']
        xy=xy@R.T+t
        cloth,inbounds=read_grid(mask,xy,origin,cfg.grid_mm)
        measured,_=read_grid(observed,xy,origin,cfg.grid_mm);measured &= inbounds
        ids=np.flatnonzero(cloth)
        if not len(ids):return dict(status='UNMEASURABLE',reason='no cloth intersection at this section')
        splits=np.flatnonzero(np.diff(ids)>1)
        if len(splits):return dict(status='UNMEASURABLE',reason='multiple/disconnected cloth intervals; boundary identity ambiguous')
        left,right=ids[0],ids[-1];result={}
        for side,index,sign in [('left',left,-1),('right',right,1)]:
            inside=index-sign*np.arange(support);outside=index+sign*np.arange(1,support+1)
            allids=np.r_[inside,outside]
            valid=bool(allids.min()>=0 and allids.max()<len(u))
            valid=valid and bool(measured[allids].all() and cloth[inside].all() and (~cloth[outside]).all())
            result[side]=float(u[index]+sign*cfg.grid_mm/2) if valid else None
        rows.append(result)
    result=dict(status='MEASURED',reason=None)
    for side in ('left','right'):
        values=[row[side] for row in rows]
        valid=all(x is not None for x in values)
        valid=valid and max(values)-min(values)<=cfg.boundary_support_mm
        result[side]=float(np.median(values)) if valid else None
        result[side+'_reason']=None if valid else 'edge lacks continuous measured cloth/background support across adjacent sections'
    return result


def boundary_profiles(refmask,current,observed,origin,frame,refpoints,cfg,R,t):
    ys,xs=np.where(refmask);xy=np.c_[xs,ys]*cfg.grid_mm+origin+cfg.grid_mm/2
    uu=(xy-frame['center'])@frame['ex'];vv=(xy-frame['center'])@frame['ey']
    hems=np.array([refpoints[k][:2] for k in ('HL','HR')])
    length=float(np.mean((hems-frame['center'])@frame['ey']))
    u=np.arange(uu.min()-40,uu.max()+40,cfg.grid_mm)
    samples=[]
    for v in np.arange(cfg.section_step_mm,max(cfg.section_step_mm,length),cfg.section_step_mm):
        ref=section_edges(refmask,np.ones_like(refmask),u,v,frame,origin,cfg,np.eye(2),np.zeros(2))
        cur=section_edges(current,observed,u,v,frame,origin,cfg,R,t)
        row=dict(v_mm=float(v),v_fraction=float(v/length))
        for side in ('left','right'):
            a,b=ref.get(side),cur.get(side)
            valid=a is not None and b is not None
            # x increases garment-left to garment-right; inward is positive on both sides.
            delta=(b-a if side=='left' else a-b) if valid else None
            row[side]=dict(status='MEASURED' if valid else 'UNMEASURABLE',reference_u_mm=a,current_u_mm=b,
                           inward_mm=delta,inward_beyond_tolerance_mm=max(0.,delta-cfg.tolerance_mm) if valid else None,
                           reason=None if valid else ('reference: '+str(ref.get(side+'_reason',ref.get('reason'))) if a is None else 'current: '+str(cur.get(side+'_reason',cur.get('reason')))))
        samples.append(row)
    summaries={}
    for side in ('left','right'):
        values=[s[side]['inward_mm'] for s in samples if s[side]['inward_mm'] is not None]
        run=longest=0
        for s in samples:
            d=s[side]['inward_mm'];run=run+1 if d is not None and d>cfg.tolerance_mm else 0;longest=max(run,longest)
        summaries[side]=dict(measured_sections=len(values),total_sections=len(samples),
                            median_inward_mm=float(np.median(values)) if values else None,
                            max_inward_mm=max(values) if values else None,
                            longest_consecutive_inward_sections=longest,
                            longest_measured_inward_span_mm=max(0,longest-1)*cfg.section_step_mm,
                            note='Unmeasurable sections break continuity; finite samples do not prove material-edge continuity.')
    return dict(frame='garment-local: u left-to-right; v armpit-line toward hem',
                origin_table_mm=frame['center'].tolist(),u_axis=frame['ex'].tolist(),v_axis=frame['ey'].tolist(),
                delta_left='u_current_left - u_reference_left',delta_right='u_reference_right - u_current_right',
                positive='inward displacement',step_mm=cfg.section_step_mm,tolerance_mm=cfg.tolerance_mm,
                reference_body_length_mm=length,pose_scope='Profile evaluated at nominal accepted pose; sensitivity bounds describe area asymmetry, not per-section confidence intervals.',samples=samples,summary=summaries)


def summarize_asymmetry(trials,margin):
    differences=[];areas=[]
    for trial in trials:
        regs=trial['regions'];left=regs['LEFT_LOWER_BODY'];right=regs['RIGHT_LOWER_BODY']
        if left['missing_ratio'] is None or right['missing_ratio'] is None:continue
        differences.append(right['missing_ratio']-left['missing_ratio'])
        areas.append(right['missing_area_mm2']-left['missing_area_mm2'])
    complete=len(differences)==len(trials) and bool(trials)
    if not complete:status='UNKNOWN';relation='UNKNOWN'
    elif min(differences)>margin:status='STABLE';relation='RIGHT_GREATER_THAN_LEFT'
    elif max(differences)<-margin:status='STABLE';relation='LEFT_GREATER_THAN_RIGHT'
    elif min(differences)>=-margin and max(differences)<=margin:status='NO_CLEAR_ASYMMETRY';relation='WITHIN_MARGIN'
    else:status='UNCERTAIN';relation='PERTURBATION_SENSITIVE'
    return dict(status=status,relation=relation,comparison_metric='right_missing_ratio - left_missing_ratio',ratio_margin=margin,valid_trials=len(differences),total_trials=len(trials),
                right_minus_left_ratio_range=[min(differences),max(differences)] if differences else None,
                right_minus_left_area_mm2_range=[min(areas),max(areas)] if areas else None,
                scope='Only the explicitly sampled bounded perturbations, not every possible pose or a statistical confidence interval.')


def assessment(report):
    """Small structured facts for a future one-shot input, separate from full audit."""
    deform=report.get('deformation',{});regions=deform.get('regions',{});sensitivity=deform.get('registration_uncertainty',{})
    asymmetry=sensitivity.get('asymmetry',dict(status='UNKNOWN',relation='UNKNOWN'))
    result=dict(registration=dict(status='ACCEPTED' if report.get('registration',{}).get('status')=='OK' else 'UNRELIABLE',
                                 rmse_mm=report.get('registration',{}).get('rmse_mm'),
                                 perturbation_bounds=sensitivity.get('bounds')),
                lower_body={},asymmetry=asymmetry,
                interpretation='INSUFFICIENT_OR_POSE_SENSITIVE_EVIDENCE',
                semantic_corners=report.get('corners',{}),
                limitations='Projected deficits and visible silhouette shifts are not material shortening or verified hem identities. No DONE rule.')
    result['section_curve']=dict(unit='mm',positive='inward',pose='nominal accepted pose; not a pose confidence interval',
        unmeasurable=None,samples=[dict(v_mm=row['v_mm'],left_inward_mm=row['left']['inward_mm'],right_inward_mm=row['right']['inward_mm'])
        for row in report.get('body_cross_sections',{}).get('samples',[])])
    for side,key in [('left','LEFT_LOWER_BODY'),('right','RIGHT_LOWER_BODY')]:
        region=regions.get(key,{})
        values=sensitivity.get('regional_ranges',{}).get(key,{})
        ratio=region.get('missing_ratio')
        interval=values.get('missing_ratio',{})
        classify=lambda x:'LOW' if x<.05 else 'HIGH' if x>=report['settings']['contraction_ratio'] else 'MODERATE'
        complete=interval.get('valid_trials',0)>0 and interval.get('valid_trials')==interval.get('total_trials')
        level='UNKNOWN'
        if ratio is not None and complete:
            lo,hi=classify(interval['min']),classify(interval['max'])
            level=lo if lo==hi else 'UNCERTAIN'
        result['lower_body'][side]=dict(projected_deficit=level,missing_ratio=ratio,missing_area_mm2=region.get('missing_area_mm2'),
                                       tolerance_mm=report.get('settings',{}).get('tolerance_mm'),perturbed_ranges=values,structural_state=region.get('state','UNKNOWN'),
                                       boundary_profile=report.get('body_cross_sections',{}).get('summary',{}).get(side))
    if result['registration']['status']=='ACCEPTED' and asymmetry['status']=='STABLE':
        dominant='right' if asymmetry['relation']=='RIGHT_GREATER_THAN_LEFT' else 'left'
        if result['lower_body'][dominant]['projected_deficit']=='HIGH':
            result['interpretation']='LOWER_BODY_ASYMMETRY_NOT_EXPLAINED_BY_SAMPLED_UPPER_BODY_RIGID_ALIGNMENT'
    if asymmetry['status']=='NO_CLEAR_ASYMMETRY' and all(v['projected_deficit']=='LOW' for v in result['lower_body'].values()):
        result['interpretation']='NO_SIGNIFICANT_LOWER_BODY_PROJECTED_DEFICIT'
    return result


def assessment_text(value):
    import json
    return 'Geometry Assessment\n'+json.dumps(value,ensure_ascii=False,indent=2)


def write_outputs(folder, report):
    """Machine report + short prompt text + exact samples + gap-preserving plot."""
    import csv,json
    from pathlib import Path
    folder=Path(folder);value=assessment(report)
    (folder/'geometry_assessment.json').write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
    (folder/'geometry_assessment.txt').write_text(assessment_text(value))
    profile=report.get('body_cross_sections',{})
    samples=profile.get('samples',[])
    fields=['v_mm','v_fraction','side','status','reference_u_mm','current_u_mm','inward_mm','inward_beyond_tolerance_mm','reason']
    with (folder/'body_cross_sections.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in samples:
            for side in ('left','right'):writer.writerow(dict(v_mm=row['v_mm'],v_fraction=row['v_fraction'],side=side,**row[side]))
    if samples:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(15,5),sharey=True)
        v=[row['v_mm'] for row in samples]
        for side,color,ax in [('left','tab:blue',axes[0]),('right','tab:orange',axes[1])]:
            series=lambda key:[row[side][key] if row[side][key] is not None else np.nan for row in samples]
            ax.plot(series('reference_u_mm'),v,'--',color='gray',label='reference')
            # Mask current points if reference/current comparison was unmeasurable.
            current=[row[side]['current_u_mm'] if row[side]['status']=='MEASURED' else np.nan for row in samples]
            ax.plot(current,v,'o-',color=color,label='measured current')
            ax.set_title(side+' body boundary');ax.set_xlabel('garment-local u (mm)');ax.legend();ax.grid(alpha=.25)
            axes[2].plot(series('inward_mm'),v,'o-',color=color,label=side)
        axes[0].set_ylabel('garment-local v from armpits (mm)');axes[0].invert_yaxis()
        axes[2].axvspan(-profile['tolerance_mm'],profile['tolerance_mm'],color='gray',alpha=.15)
        axes[2].axvline(0,color='gray');axes[2].set_xlabel('inward displacement (mm); positive=inward')
        axes[2].set_title('Unmeasurable sections remain gaps');axes[2].legend();axes[2].grid(alpha=.25)
        fig.tight_layout();fig.savefig(folder/'body_cross_sections.png',dpi=150);plt.close(fig)
    return value
