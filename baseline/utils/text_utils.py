def splitlines_with_positions(text, keepends=False):
    """
    input: string
    outputs: [(line_0, start_0, end_0), (line_1, start_1, end_1), ...]
    """
    lines = text.splitlines(keepends=True)
    if not keepends:
        lines_without_end = text.splitlines(keepends=False)

    outputs = list()
    prev_end = 0
    for l, line in enumerate(lines):
        start = prev_end + text[prev_end:].find(line)
        prev_end = start + len(line)
        if not keepends:
            line = lines_without_end[l]
        outputs.append((line, start, start+len(line)))
    return outputs

