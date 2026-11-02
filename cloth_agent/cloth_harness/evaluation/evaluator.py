"""Fixed evaluator: UNKNOWN is abstention, not a correct label. No agent scores."""
LABELS={'grasp_status':{'SUCCESS','MISS','SLIPPED','UNKNOWN'},'area_change':{'INCREASED','UNCHANGED','DECREASED','UNKNOWN'},
        'motion_type':{'RIGID_DOMINANT','NONRIGID','MIXED','UNKNOWN'},'local_shape_change':{'CHANGED','NO_CLEAR_CHANGE','UNKNOWN'},
        'visible_unfolding_progress':{'IMPROVED','NO_CLEAR_PROGRESS','WORSE','UNKNOWN'}}

def validate_label(record):
    if set(record.get('labels',{}))!=set(LABELS):raise ValueError('oracle labels must use fixed five-field schema')
    for k,v in record['labels'].items():
        if v not in LABELS[k]:raise ValueError('invalid oracle label '+k)
    if record.get('confirmed_by') and not record.get('evidence'):raise ValueError('confirmed labels require evidence references')

def classification_metrics(truth,prediction):
    eligible=[k for k,v in truth.items() if v!='UNKNOWN'];confident=[k for k in eligible if prediction.get(k,'UNKNOWN')!='UNKNOWN']
    errors=sum(prediction[k]!=truth[k] for k in confident)
    return {'labelled':len(eligible),'confident':len(confident),'errors':errors,
            'coverage':len(confident)/len(eligible) if eligible else None,
            'accuracy_when_confident':1-errors/len(confident) if confident else None,
            'false_confident_rate':errors/len(eligible) if eligible else None}

def evaluate_outcomes(outcomes,labels):
    truth={};pred={}
    for key,out in outcomes.items():
        label=labels.get(key)
        if not label or not label.get('confirmed_by'):continue
        validate_label(label)
        predicted={'grasp_status':out['execution']['grasp_status'],'area_change':out['coverage']['status'] if out['coverage']['status'] in LABELS['area_change'] else 'UNKNOWN',
                   'motion_type':'UNKNOWN','local_shape_change':'UNKNOWN','visible_unfolding_progress':'UNKNOWN'}
        for field,value in label['labels'].items():truth[key+'/'+field]=value;pred[key+'/'+field]=predicted[field]
    return classification_metrics(truth,pred)
