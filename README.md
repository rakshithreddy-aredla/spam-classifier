# SMS spam classification

Multinomial Naive Bayes with TF-IDF features on the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — 5,572 real messages.

```
              precision   recall   f1-score   support
     ham         0.97      1.00       0.98      1206
    spam         1.00      0.78       0.88       187
accuracy 0.9706
```

The two numbers worth reading together: recall on spam is 0.78, and precision on ham is 1.00. The model misses about a fifth of spam, but it has never once deleted a legitimate message in the test set.

For this problem that's the right way round. Missing a spam message costs the user a second of attention; deleting a real one costs them something they can't get back. Lowering the decision threshold buys spam recall and spends ham precision, and on a dataset where the classes are this lopsided that's a worse trade than the headline accuracy suggests.

Naive Bayes suits the job for a specific reason: it's fast enough to retrain per message, handles thousands of sparse features without complaint, and the independence assumption that makes it naive is a poor fit for this data that hasn't cost it much.

```bash
pip install -r requirements.txt
python spam_classifier.py
```

## Files

```
spam_classifier.py     # training, evaluation, confusion matrix
data/SMSSpamCollection # UCI dataset
requirements.txt
```