from sklearn.metrics import classification_report, confusion_matrix,accuracy_score, roc_auc_score


def print_report(output, target, name='Training', epoch=None):
    print('-------------{} Report----------------'.format(name))
    print(classification_report(target, output, digits=4))

    #print('\nConfusion Matrix')
    num_classes = max(output)

    matrices = confusion_matrix(target, output, labels=list(range(num_classes+1)))
    tn, fp, fn, tp = confusion_matrix(target, output, labels=[0,1]).ravel()
    print('TN, FP, FN, TP',tn,fp,fn,tp)
	
    acc = accuracy_score(target, output)
    if epoch is not None:
        print('Epoch {0}, {1} acc: {2:.4f}'.format(epoch, name, acc))
    else:
        print('{0} acc: {1:.4f}'.format(name, acc))

    print(matrices)