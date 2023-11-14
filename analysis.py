import re

ref = "data/test_10min_tw_hk/text"
pred = "exp/asr_train_asr_s3prl_hubertlargecmn_tw_hk_1h/decode_asr_asr_model_valid.loss.ave/test_10min_tw_hk/text"

def CER(reference, hypothesis):
    """
    Calculate Character Error Rate (CER) between reference and hypothesis strings.

    Args:
        reference (str): The ground truth/reference string.
        hypothesis (str): The hypothesis/generated string.

    Returns:
        float: The Character Error Rate (CER) between reference and hypothesis.
    """
    # Remove any leading/trailing whitespaces and convert to lowercase
    reference = reference.strip().lower()
    hypothesis = hypothesis.strip().lower()

    # Calculate Levenshtein distance (edit distance) between reference and hypothesis
    distance = [[0 for _ in range(len(hypothesis) + 1)] for _ in range(len(reference) + 1)]
    for i in range(len(reference) + 1):
        distance[i][0] = i
    for j in range(len(hypothesis) + 1):
        distance[0][j] = j

    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                cost = 0
            else:
                cost = 1

            distance[i][j] = min(
                distance[i - 1][j] + 1,
                distance[i][j - 1] + 1,
                distance[i - 1][j - 1] + cost
            )

    # CER is the normalized Levenshtein distance
    cer = distance[len(reference)][len(hypothesis)] / max(len(reference), len(hypothesis))

    return cer

def WER(reference, hypothesis):
    """
    Calculate Word Error Rate (WER) between reference and hypothesis sentences.

    Args:
        reference (str): The ground truth/reference sentence.
        hypothesis (str): The hypothesis/generated sentence.

    Returns:
        float: The Word Error Rate (WER) between reference and hypothesis.
    """
    # Tokenize the reference and hypothesis sentences into words
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # Initialize the dynamic programming matrix
    dp = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    # Fill the matrix
    for i in range(len(ref_words) + 1):
        for j in range(len(hyp_words) + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            else:
                insertion = dp[i][j - 1] + 1
                deletion = dp[i - 1][j] + 1
                substitution = dp[i - 1][j - 1] + (0 if ref_words[i - 1] == hyp_words[j - 1] else 1)
                dp[i][j] = min(insertion, deletion, substitution)

    # Compute Word Error Rate (WER)
    wer = dp[len(ref_words)][len(hyp_words)] / len(ref_words)

    return wer

def process_txt(txt):
    txt = ' '.join(txt.split()[1:])
    # remove integers and multiple spaces
    txt = re.sub(r'\d+', '', txt)
    txt = re.sub(r'\s+', ' ', txt)
    return txt

def process(ref_file, pred_file):
    with open(ref_file) as f:
        refs = f.readlines()
    with open(pred_file) as f:
        preds = f.readlines()

    ERs = []
    for i in range(len(refs)):
        ref, pred = refs[i], preds[i]
        ref, pred = process_txt(ref), process_txt(pred)
        er = WER(ref, pred) 
        ERs.append(er)

    avg_er = sum(ERs)/len(ERs)
    print(avg_er)

process(ref, pred)
