function [acc, precision, recall, f1, cm] = computeMetrics(YTrue, YPred)
% YTrue, YPred: categorical with classes {'nospeech','speech'}

cm = confusionmat(YTrue, YPred);  % rows: true, cols: pred
% [nospeech, speech]
tn = cm(1,1);
fp = cm(1,2);
fn = cm(2,1);
tp = cm(2,2);

acc = (tp + tn) / sum(cm(:));
precision = tp / max(tp + fp, 1);
recall    = tp / max(tp + fn, 1);
f1        = 2 * precision * recall / max(precision + recall, 1e-8);
end
