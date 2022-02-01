

function ap = average_precision(pred, gt, poslabel, neglabel)

% compute precision/recall
[~, si] = sort(-pred);
tp = gt(si)==poslabel;  tp  = cumsum(tp);
fp = gt(si)==neglabel;  fp  = cumsum(fp);
rec = tp/sum(gt>0);     prec= tp./(fp+tp);

% compute average precision
ap = 0;
for t = 0:0.1:1
    p=max(prec(rec>=t));
    if isempty(p), p=0; end
    ap = ap+p/11;
end