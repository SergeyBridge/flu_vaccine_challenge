params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'thread_count': -1,
    'task_type': 'CPU', #  if get_gpu_device_count() > 0 else 'CPU',
    # 'task_type': 'GPU' if torch.cuda.is_available() else 'CPU',
    'grow_policy': 'Lossguide',
    'auto_class_weights': 'Balanced',
    'langevin': True,  # CPU only
    'iterations': 25000,
    'learning_rate': 4e-3,
    'random_seed': 0,
    'l2_leaf_reg': 1e-1,
    'depth': 16,
    # 'max_leaves': 10,
    'border_count': 128,
    'verbose': 100,
    'od_type': 'Iter',
    'od_wait': 100,

    # random control
    'random_strength': 0.001,
    'rsm': 1,

}



ordinal = [
    'h1n1_concern',
    'h1n1_knowledge',
    'opinion_h1n1_vacc_effective',
    'opinion_h1n1_risk',
    'opinion_h1n1_sick_from_vacc',
    'opinion_seas_vacc_effective',
    'opinion_seas_risk',
    'opinion_seas_sick_from_vacc',
    'age_group',
    'education',
    'income_poverty',
    'household_adults',
    'household_children',
]

categorical = [
    'behavioral_antiviral_meds',
    'behavioral_avoidance',
    'behavioral_face_mask',
    'behavioral_wash_hands',
    'behavioral_large_gatherings',
    'behavioral_outside_home',
    'behavioral_touch_face',
    'doctor_recc_h1n1',
    'doctor_recc_seasonal',
    'chronic_med_condition',
    'child_under_6_months',
    'health_worker',
    'health_insurance',
    'race',
    'sex',
    'marital_status',
    'rent_or_own',
    'employment_status',
    'hhs_geo_region',
    'census_msa',
    'employment_industry',
    'employment_occupation',
]

ordinal_to_replace = {
    # age_groups
    '18 - 34 Years': 1,
    '35 - 44 Years': 2,
    '45 - 54 Years': 3,
    '55 - 64 Years': 4,
    '65+ Years': 5,

    # educations = {
    '-1': -1,
    '< 12 Years': 1,
    'Some College': 2,
    '12 Years': 3,
    'College Graduate': 4,

    # income_poverties
    '-1': -1,
    'Below Poverty': 1,
    '<= $75,000, Above Poverty': 2,
    '> $75,000': 3,
}