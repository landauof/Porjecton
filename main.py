import FeatureExtractor as fe
import FeatureSelection
import Classifier
from pathlib import Path

# Consts
out_0_path = 'MFDCA-DATA/FraudedFeatureOutputs/output0.csv'
FEATURE_SELECTION_MOST_COMMON = 1
CLASSIFIER_ONE_CLASS_SVM = 1


def run(is_changing_n=False, number_to_ngram=3, number_of_features=220, fs_method=FEATURE_SELECTION_MOST_COMMON,
        c_method=CLASSIFIER_ONE_CLASS_SVM):

    # set param for final calcs
    all_ans = []

    # Feature extraction
    out_0_file = Path(out_0_path)
    if is_changing_n:
        fe.export_to_csv_all_users(number_to_ngram)
    if not out_0_file.exists():
        fe.export_to_csv_all_users(number_to_ngram)

    for user_number in range(0, 5):
        # Feature selection
        FeatureSelection.select_features(number_of_features, fs_method, user_number)

        # Classifier
        ans = Classifier.classify(number_of_features, c_method)
        all_ans.append(ans)

    print("""
    ** FINAL SCORE : {} **
    """.format(sum(all_ans)/len(all_ans)))


run(False, 3, 220, FEATURE_SELECTION_MOST_COMMON, CLASSIFIER_ONE_CLASS_SVM)
