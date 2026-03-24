screen_question_prompt = {
    "anxiety": 'Predicting Anxiety-level',
    "depression": 'Predicting Depression-level',
    "satisfaction": 'Predicting Life-satisfaction',
    "stress": 'Predicting Stress-level'
}

assessment_question_prompt = {
    "re_suicidal": 'Predicting Suicidal Ideation level',
    "re_obsessive": 'Predicting Obsessive Symptom level',
    "re_psychotic": 'Predicting Psychotic Symptom level',
    "re_mania": 'Predicting Mania level',
    "re_ADHD": 'Predicting ADHD level',
}

screen_label_candidate = {
    "anxiety": ['No anxiety', 'Mild anxiety', 'Severe anxiety'],
    "depression": ['No depression', 'Mild depression', 'Severe depression'],
    "satisfaction": ['Low life satisfaction', 'High life satisfaction'],
    "stress": ['No life stress', 'Low life stress', 'High life stress']
}

assessment_label_candidate = {
    "re_suicidal": ['Mild suicidal ideation', 'Severe suicidal ideation'],
    "re_obsessive": ['Mild obsessive symptoms', 'Severe obsessive symptoms'],
    "re_psychotic": ['Mild psychotic symptoms', 'Severe psychotic symptoms'],
    "re_mania": ['Mild mania', 'Severe mania'],
    "re_ADHD": ['Mild ADHD', 'Severe ADHD']
}


screen_candidate_class = [
    [[2822, 18547], [44, 699, 18547], [1542, 19846, 18547]],
    [[2822, 18710], [44, 699, 18710], [1542, 19846, 18710]],
    [[25162, 2324, 24617], [12243, 2324, 24617]],
    [[2822, 2324, 8631], [25162, 2324, 8631], [12243, 2324, 8631]]
]

screen_candidate_token = [
    [2822, 44, 1542],
    [2822, 44, 1542],
    [25162, 12243],
    [2822, 25162, 12243]
]

assessment_candidate_class = [
    [[44, 699, 66153, 2679, 367], [1542, 19846, 66153, 2679, 367]],
    [[44, 699, 84132, 13803], [1542, 19846, 84132, 13803]],
    [[44, 699, 94241, 13803], [1542, 19846, 94241, 13803]],
    [[44, 699, 893, 689], [1542, 19846, 893, 689]],
    [[44, 699, 52564], [1542, 19846, 52564]]
]

assessment_candidate_token = [
    [44, 1542],
    [44, 1542],
    [44, 1542],
    [44, 1542],
    [44, 1542],
]

question_prompt_mapping = {
    "Screen": screen_question_prompt,
    "Assessment": assessment_question_prompt,
}

label_candidate_mapping = {
    "Screen": screen_label_candidate,
    "Assessment": assessment_label_candidate,
}

candidate_class_mapping = {
    "Screen": screen_candidate_class,
    "Assessment": assessment_candidate_class,
}

candidate_token_mapping = {
    "Screen": screen_candidate_token,
    "Assessment": assessment_candidate_token,
}

def get_question_prompt(task_type: str):
    if task_type not in question_prompt_mapping:
        raise ValueError(f"Unknown task_type: {task_type}")
    return question_prompt_mapping[task_type]

def get_label_candidate(task_type: str):
    if task_type not in label_candidate_mapping:
        raise ValueError(f"Unknown task_type: {task_type}")
    return label_candidate_mapping[task_type]

def get_candidate_class(task_type: str):
    if task_type not in candidate_class_mapping:
        raise ValueError(f"Unknown task_type: {task_type}")
    return candidate_class_mapping[task_type]

def get_candidate_token(task_type: str):
    if task_type not in candidate_token_mapping:
        raise ValueError(f"Unknown task_type: {task_type}")
    return candidate_token_mapping[task_type]