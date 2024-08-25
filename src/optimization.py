import torch
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score
from collections import defaultdict
import tqdm

def optimization_loop(model, train_dl, val_dl, epochs, criterion, optimizer, scheduler, device, project, entity, name, config, is_lopo):
    if entity:
        wandb_logging = True
        run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            reinit=True,
            config=config
        )
    else:
        wandb_logging = False

    names = ['low_grade', 'molecular']
    metrics = defaultdict(list)
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        print('-' * 10)

        metrics['lr'].append(optimizer.state_dict()['param_groups'][0]['lr'])
        model.train()

        running_loss_train = 0.0
        running_corrects_train = 0.0

        for item in tqdm.tqdm(train_dl, desc='Training...', total=len(train_dl)):
            inputs = item['image'].to(device)
            labels = item['label'].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs.squeeze(1), labels.float())
            loss.backward()
            optimizer.step()
            preds = torch.where(outputs > 0., 1., 0.)
            #_, preds = torch.max(outputs, 1)
            running_loss_train += loss.item() * inputs.size(0)
            running_corrects_train += torch.sum(preds.squeeze(1) == labels.data).item()

        epoch_loss_train = running_loss_train / len(train_dl.dataset)
        epoch_acc_train = running_corrects_train / len(train_dl.dataset)

        metrics['train/loss'].append(epoch_loss_train)
        metrics['train/acc'].append(epoch_acc_train)
        if not is_lopo:
            model.eval()
            raw_predictions = []
            true_labels = []

            for item in tqdm.tqdm(val_dl, desc='Validating...', total=len(val_dl)):
                inputs = item['image'].to(device)
                labels = item['label'].to(device)

                with torch.no_grad():
                    outputs = model(inputs)
                    raw_predictions.append(outputs.cpu().sigmoid().item())
                    true_labels.append(labels.item())

            predictions = np.hstack(raw_predictions).tolist()
            true_labels = np.hstack(true_labels).tolist()

            epoch_loss_val = torch.nn.functional.binary_cross_entropy(torch.tensor(predictions).float(), torch.tensor(true_labels).float()).item()
            epoch_acc_val = accuracy_score(true_labels, [1 if pred > 0.5 else 0 for pred in predictions])
            epoch_precision_val = precision_score(true_labels, [1 if pred > 0.5 else 0 for pred in predictions])
            epoch_recall_val = recall_score(true_labels, [1 if pred > 0.5 else 0 for pred in predictions])
            epoch_f1_val = f1_score(true_labels, [1 if pred > 0.5 else 0 for pred in predictions])
            epoch_roc_auc_val = roc_auc_score(true_labels, predictions)
            epoch_balanced_acc_val = balanced_accuracy_score(true_labels, [1 if pred > 0.5 else 0 for pred in predictions])

            metrics['val/loss'].append(epoch_loss_val)
            metrics['val/acc'].append(epoch_acc_val)
            metrics['val/precision'].append(epoch_precision_val)
            metrics['val/recall'].append(epoch_recall_val)
            metrics['val/f1'].append(epoch_f1_val)
            metrics['val/roc_auc'].append(epoch_roc_auc_val)
            metrics['val/balanced_acc'].append(epoch_balanced_acc_val)

            if wandb_logging:
                logs = {k: v[-1] for k, v in metrics.items()}
                logs.update(
                    {
                        "confusion_matrix": wandb.plot.confusion_matrix(
                            probs=None,
                            y_true=np.array(true_labels),
                            preds=np.array(
                                [round(prediction) for prediction in predictions]
                            ),
                            class_names=names,
                        )
                    }
                )
                wandb.log(logs, step=epoch)

            scheduler.step(epoch_loss_val)

    if is_lopo:
        model.eval()
        for item in val_dl:
            inputs = item['image'].to(device)
            labels = item['label'].to(device)

            with torch.no_grad():
                outputs = model(inputs)
                raw_prediction = outputs.cpu().sigmoid().item()
                true_label = labels.item()

    if wandb_logging:
        run.finish()
    if is_lopo:
        return model, raw_prediction, true_label
    else:
        return model, metrics


def compute_metrics(y_pred, y_true):

    loss = torch.nn.functional.binary_cross_entropy(torch.tensor(y_pred).float(), torch.tensor(y_true).float()).item()
    acc = accuracy_score(y_true, [1 if pred > 0.5 else 0 for pred in y_pred])
    precision = precision_score(y_true, [1 if pred > 0.5 else 0 for pred in y_pred])
    recall = recall_score(y_true, [1 if pred > 0.5 else 0 for pred in y_pred])
    f1 = f1_score(y_true, [1 if pred > 0.5 else 0 for pred in y_pred])
    roc_auc = roc_auc_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, [1 if pred > 0.5 else 0 for pred in y_pred])
    return {'loss': loss, 'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc, 'balanced_acc': balanced_acc}
    