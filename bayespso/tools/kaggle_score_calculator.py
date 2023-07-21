import pandas
import math


def setwise_score_calculator(merged_df, kaggleSet):
    signal = sum(merged_df.loc[
        (merged_df['Class']=='s') & \
        (merged_df['KaggleSet']==kaggleSet) & \
        (merged_df['Label']=='s'), 'KaggleWeight'
    ])
    background = sum(merged_df.loc[
        (merged_df['Class']=='s') & \
        (merged_df['KaggleSet']==kaggleSet) & \
        (merged_df['Label']=='b'), 'KaggleWeight'
    ])
    return AMS(signal, background)


def AMS(s, b):
    """ Approximate Median Significance defined as:
        AMS = sqrt(
                2 { (s + b + b_r) log[1 + (s/(b+b_r))] - s}
              )        
    where b_r = 10, b = background, s = signal, log is natural logarithm """
    
    br = 10.0
    radicand = 2 *( (s+b+br) * math.log (1.0 + s/(b+br)) -s)
    if radicand < 0:
        print('radicand is negative. Exiting')
        exit()
    else:
        return math.sqrt(radicand)


def calculate_ams_scores(path_to_truth, path_to_submission):
    full_data = pandas.read_csv(path_to_truth)
    submission = pandas.read_csv(path_to_submission)
    validation_data = full_data.loc[(full_data['KaggleSet']=='v') | (full_data['KaggleSet']=='b')]
    merged_df = pandas.merge(validation_data, submission, on='EventId')
    private_ams = setwise_score_calculator(merged_df, kaggleSet='v')
    public_ams = setwise_score_calculator(merged_df, kaggleSet='b')
    print('Private AMS = %s' %private_ams)
    print('Public AMS = %s' %public_ams)
    return private_ams, public_ams

