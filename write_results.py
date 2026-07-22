import numpy as np
from sklearn.metrics import auc


LOW_SNR_SEQUENCE_NAMES = frozenset({
    'Sequence92',
    'Sequence47',
    'Sequence56',
    'Sequence59',
    'Sequence76',
    'Sequence101',
    'Sequence105',
    'Sequence119',
})
HIGH_SNR_SEQUENCE_NAMES = frozenset({
    'Sequence85',
    'Sequence86',
    'Sequence87',
    'Sequence88',
    'Sequence89',
    'Sequence90',
    'Sequence91',
    'Sequence93',
    'Sequence94',
    'Sequence95',
    'Sequence96',
    'Sequence97',
})


def _safe_divide(numerator, denominator):
    """Divide metrics without emitting NaN/Inf for empty targets or pixels."""
    numerator, denominator = np.broadcast_arrays(
        np.asarray(numerator, dtype=np.float64),
        np.asarray(denominator, dtype=np.float64),
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


def get_nudt_mirsdt_snr_indices(sequence_names):
    sequence_names = list(sequence_names)
    known_names = LOW_SNR_SEQUENCE_NAMES | HIGH_SNR_SEQUENCE_NAMES
    unknown_names = sorted(set(sequence_names) - known_names)
    if unknown_names:
        raise ValueError(
            'NUDT-MIRSDT SNR group is undefined for: %s'
            % ', '.join(unknown_names)
        )

    low_snr = np.array([
        index for index, name in enumerate(sequence_names)
        if name in LOW_SNR_SEQUENCE_NAMES
    ], dtype=np.int64)
    high_snr = np.array([
        index for index, name in enumerate(sequence_names)
        if name in HIGH_SNR_SEQUENCE_NAMES
    ], dtype=np.int64)
    if low_snr.size == 0 or high_snr.size == 0:
        raise ValueError('Both low- and high-SNR sequences are required for grouped metrics.')
    return low_snr, high_snr


def writeNUDTMIRSDT_ROC(FalseNumAll, TrueNumAll, TgtNumAll, pixelsNumber, total_intersection_mid, total_union_mid,
                        Th_Seg, TEST_DATASET, log_string):
    low_snr, high_snr = get_nudt_mirsdt_snr_indices(TEST_DATASET.seq_names)
    Pd_L = _safe_divide(np.sum(TrueNumAll[low_snr, :], axis=0), np.sum(TgtNumAll[low_snr, :], axis=0))
    Fa_L = _safe_divide(np.sum(FalseNumAll[low_snr, :], axis=0), pixelsNumber[low_snr].sum())
    auc_L = auc(Fa_L, Pd_L)
    Pd_H = _safe_divide(np.sum(TrueNumAll[high_snr, :], axis=0), np.sum(TgtNumAll[high_snr, :], axis=0))
    Fa_H = _safe_divide(np.sum(FalseNumAll[high_snr, :], axis=0), pixelsNumber[high_snr].sum())
    auc_H = auc(Fa_H, Pd_H)


    Pd_all = _safe_divide(np.sum(TrueNumAll[:, :], axis=0), np.sum(TgtNumAll[:, :], axis=0))
    Fa_all = _safe_divide(np.sum(FalseNumAll[:, :], axis=0), pixelsNumber.sum())
    auc_all = auc(Fa_all, Pd_all)
    for seq_i in range(len(TEST_DATASET)):
        seq_name = TEST_DATASET.seq_names[seq_i]
        log_string('%s results:\n' % seq_name)
        for seg_i in range(len(Th_Seg)):
            log_string('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\n' % (Th_Seg[seg_i],
                TrueNumAll[seq_i, seg_i], TgtNumAll[seq_i, seg_i], _safe_divide(TrueNumAll[seq_i, seg_i], TgtNumAll[seq_i, seg_i]),
                FalseNumAll[seq_i, seg_i], _safe_divide(FalseNumAll[seq_i, seg_i], pixelsNumber[seq_i])))


    log_string('Low SNR results:\tAUC:%.5f\n' % (auc_L))
    for th_i in range(len(Th_Seg)):
        log_string('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\n' % (Th_Seg[th_i],
                                                                      TrueNumAll[low_snr, th_i].sum(),
                                                                      TgtNumAll[low_snr, th_i].sum(),
                                                                      _safe_divide(TrueNumAll[low_snr, th_i].sum(), TgtNumAll[
                                                                          low_snr, th_i].sum()),
                                                                      FalseNumAll[low_snr, th_i].sum(),
                                                                      _safe_divide(FalseNumAll[low_snr, th_i].sum(), pixelsNumber[
                                                                          low_snr].sum())))
    log_string('High SNR results:\tAUC:%.5f\n' % (auc_H))
    for th_i in range(len(Th_Seg)):
        log_string('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\n' % (Th_Seg[th_i],
                                                                      TrueNumAll[high_snr, th_i].sum(),
                                                                      TgtNumAll[high_snr, th_i].sum(),
                                                                      _safe_divide(TrueNumAll[high_snr, th_i].sum(), TgtNumAll[
                                                                          high_snr, th_i].sum()),
                                                                      FalseNumAll[high_snr, th_i].sum(),
                                                                      _safe_divide(FalseNumAll[high_snr, th_i].sum(), pixelsNumber[
                                                                          high_snr].sum())))
    log_string('Final results:\tAUC:%.5f\n' % (auc_all))
    for th_i in range(len(Th_Seg)):
        log_string('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\n' % (Th_Seg[th_i],
                                                                      TrueNumAll[:, th_i].sum(), TgtNumAll[:, th_i].sum(),
                                                                      _safe_divide(TrueNumAll[:, th_i].sum(), TgtNumAll[:, th_i].sum()),
                                                                      FalseNumAll[:, th_i].sum(),
                                                                      _safe_divide(FalseNumAll[:, th_i].sum(), pixelsNumber.sum())))

    ############### log IoU results ###############
    mIoU_mid = _safe_divide(total_intersection_mid, total_union_mid)
    log_string('Eval avg class IoU of prediction: %f' % (mIoU_mid))

    return



def writeMIRST_ROC(FalseNumAll, TrueNumAll, TgtNumAll, pixelsNumber, total_intersection_mid, total_union_mid,
                        Th_Seg, TEST_DATASET, log_string):
    Pd_all = _safe_divide(np.sum(TrueNumAll[:, :], axis=0), np.sum(TgtNumAll[:, :], axis=0))
    Fa_all = _safe_divide(np.sum(FalseNumAll[:, :], axis=0), pixelsNumber.sum())
    auc_all = auc(Fa_all, Pd_all)
    for seq_i in range(len(TEST_DATASET)):
        seq_name = TEST_DATASET.seq_names[seq_i]
        log_string('%s results:\n' % seq_name)
        for seg_i in range(len(Th_Seg)):
            log_string('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\n' % (Th_Seg[seg_i],
                TrueNumAll[seq_i, seg_i], TgtNumAll[seq_i, seg_i], _safe_divide(TrueNumAll[seq_i, seg_i], TgtNumAll[seq_i, seg_i]),
                FalseNumAll[seq_i, seg_i], _safe_divide(FalseNumAll[seq_i, seg_i], pixelsNumber[seq_i])))

    log_string('Final results:\tAUC:%.5f\n' % (auc_all))
    for th_i in range(len(Th_Seg)):
        log_string('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\n' % (Th_Seg[th_i],
                    TrueNumAll[:, th_i].sum(), TgtNumAll[:, th_i].sum(), _safe_divide(TrueNumAll[:, th_i].sum(), TgtNumAll[:, th_i].sum()),
                    FalseNumAll[:, th_i].sum(), _safe_divide(FalseNumAll[:, th_i].sum(), pixelsNumber.sum())))

    ############### log IoU results ###############
    mIoU_mid = _safe_divide(total_intersection_mid, total_union_mid)
    log_string('Eval avg class IoU of prediction: %f' % (mIoU_mid))

    return

