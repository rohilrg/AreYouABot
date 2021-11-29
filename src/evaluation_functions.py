from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def evaluation_metrics(y_true, y_pred, suffix_for_file='validation'):
    cm = confusion_matrix(y_true, y_pred, labels=[0.0, 1.0])
    df_cm = pd.DataFrame(cm, index=[0.0, 1.0],
                         columns=[0.0, 1.0])
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt='g')
    plt.xlabel('True')
    plt.ylabel('Prediction')
    plt.title('Confusion Matrix')
    plt.savefig(f'plots/confusion_matrix_for_{suffix_for_file}_set.png')
    plt.show()
    print('Test accuracy: {:.3} '.format(accuracy_score(y_true, y_pred)))
    print("Recall-score on test set: {:.3}".format(recall_score(y_true, y_pred)))
    print("Precision-score on test set: {:.3}".format(precision_score(y_true, y_pred)))
    print("F1-score on test set: {:.3}".format(f1_score(y_true, y_pred)))