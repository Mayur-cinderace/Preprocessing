import traceback
import json
import pandas as pd
import numpy as np
import sys
import os

# ensure project root is on sys.path so imports inside hinglish_pipeline resolve
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from hinglish_pipeline.combined_preprocessing_pipeline import build_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import mutual_info_classif


def main():
    out = {}
    try:
        df = pd.read_csv('hinglish_pipeline/hinglish.csv')
        out['rows'] = int(len(df))
    except Exception:
        out['error_loading_csv'] = traceback.format_exc()
        print(json.dumps(out, indent=2))
        return

    try:
        proc = build_pipeline(phonetic_normalization=True,
                              balanced_tokenization=True,
                              language_identification_tagging=True)
        processed = proc.process_dataframe(df, text_col='text', output_col='processed_text')
    except Exception:
        out['error_pipeline'] = traceback.format_exc()
        print(json.dumps(out, indent=2))
        return

    texts = processed['processed_text'].astype(str).values
    labels = df['label'].values
    out['label_counts'] = dict(pd.Series(labels).value_counts().to_dict())

    # Try TF-IDF + mutual_info
    try:
        vec = TfidfVectorizer(max_features=20000)
        X = vec.fit_transform(texts)
        out['tfidf_shape'] = X.shape
        out['tfidf_nnz'] = int(X.nnz)
        try:
            mi = mutual_info_classif(X, labels, discrete_features=False, random_state=0)
            out['mi_mean'] = float(np.mean(mi))
            top_idx = np.argsort(mi)[-10:][::-1]
            features = vec.get_feature_names_out()
            out['mi_top_k'] = [ [features[i], float(mi[i])] for i in top_idx ]
        except Exception:
            out['mi_error'] = traceback.format_exc()
            # fallback to CountVectorizer
            try:
                vec2 = CountVectorizer(max_features=5000)
                X2 = vec2.fit_transform(texts)
                out['count_shape'] = X2.shape
                out['count_nnz'] = int(X2.nnz)
                mi2 = mutual_info_classif(X2, labels, discrete_features=True, random_state=0)
                out['mi_mean_countvec'] = float(np.mean(mi2))
                top_idx2 = np.argsort(mi2)[-10:][::-1]
                features2 = vec2.get_feature_names_out()
                out['mi_top_k_countvec'] = [ [features2[i], float(mi2[i])] for i in top_idx2 ]
            except Exception:
                out['mi_countvec_error'] = traceback.format_exc()
    except Exception:
        out['tfidf_error'] = traceback.format_exc()

    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
