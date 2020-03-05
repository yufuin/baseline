def bilou_span_decode(label_sequence, unk_label_symbols=None):
    """
    label_sequence := [label_0, label_1, label_2, ...]
    label_i := 'o' or '{bilu}-(labelClass)'
    unk_label_symbols := [unk_label_0, unk_label_1, ...] (or valid for 'in' operator). if label_i matches one of them, it is treated as 'o'.

    output : [span_0, span_1, ...]
    span_i := (labelClass, span_start, span_end). this denotes that label_sequence[span_start:span_end] is for labelClass.
    """
    assert type(label_sequence) in [list, tuple], f"type(label_sequence) must be either list or tuple, but given is {type(label_sequence)}."
    assert all(type(label) is str for label in label_sequence)

    outs = list()
    state = "stand_by" # state \in ["stand_by", "under_proc"]
    state_start = -1
    state_class = None

    for i,label in enumerate(label_sequence):
        if (unk_label_symbols is not None) and (label in unk_label_symbols):
            label = "o"
        e_bilou, *e_class = label.split("-")
        assert e_bilou in "bilou", label
        if len(e_class) == 0:
            assert e_bilou == "o", label
        elif len(e_class) == 1:
            assert e_bilou in "bilu"
            e_class = e_class[0]
        else:
            raise ValueError(label)

        if e_bilou in "il":
            if state_class != e_class:
                # if the incoming label is for the continuation that is different to the current state,
                # treat the incoming label as the new beginning label.
                if e_bilou == "i":
                    e_bilou = "b"
                elif e_bilou == "l":
                    e_bilou = "u"
                else:
                    raise ValueError(label)

        if e_bilou in "bou":
            if state == "under_proc":
                # terminate the current state
                outs.append((state_class, (state_start, i)))
                state = "stand_by"
                state_start = -1
                state_class = None

            if e_bilou == "b":
                state = "under_proc"
                state_start = i
                state_class = e_class
            elif e_bilou == "u":
                outs.append((e_class, (i, i+1)))
            elif e_bilou == "o":
                pass
            else:
                raise ValueError(label)
        elif e_bilou == "i":
            assert state_class == e_class
        elif e_bilou == "l":
            assert state_class == e_class
            outs.append((state_class, (state_start, i+1)))
            state = "stand_by"
            state_start = -1
            state_class = None
        else:
            raise ValueError(label)
    if state == "under_proc":
        outs.append((state_class, (state_start, len(label_sequence))))
        state = "stand_by"
        state_start = -1
        state_class = None
    return outs
